import random
import torch
from torch import nn
import torch.nn.functional as f

# Use GPU if available
if torch.cuda.is_available():
    torch.set_default_device('cuda')

# Define perceptions
IDENTITY = torch.tensor([[0,0,0],[0,1,0],[0,0,0]], dtype=torch.float)       # Used to maintain the current state
SOBEL_X = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float)     # Detects horizontal edges
SOBEL_Y = SOBEL_X.T                                                         # Detects vertical edges

# Create perception layer, consisting of the identity, sobel x, sobel y and laplacian
PERCEPTIONS = torch.stack([IDENTITY, SOBEL_X, SOBEL_Y])
PERCEPTION_COUNT = PERCEPTIONS.shape[0]


class Grid:
    """
    Static substrate that the model can perceive but never writes to.

    Channel 0 doubles as an occupancy field: 0.0 = free space, 1.0 = blocked.
    GCA enforces this as a hard physical constraint (see GCA.apply_obstacle_mask)
    -- the overlay cannot exist on blocked cells, regardless of what the network
    has learned. A grid that is all zeros is simply "no obstacles": the overlay
    grows under pure self-organisation, same as if this plane didn't exist.

    Nothing in this codebase ever mutates `self.tensor` in place; it is
    read-only context for GCA. It's stored as a batch-of-1 tensor and
    auto-broadcast to whatever batch size the overlay has at forward() time,
    so you never need to manually expand it yourself.
    """

    def __init__(self, tensor: torch.Tensor):
        # Expected shape: (1, grid_channels, H, W), values typically in [0, 1]
        self.tensor = tensor

    @classmethod
    def from_image(cls, image_tensor: torch.Tensor):
        """
        Wrap a single (channels, H, W) image as a batch-of-1 Grid.
        Use this with e.g. the load_image() helpers elsewhere in the project.
        """
        return cls(image_tensor.unsqueeze(0))

    @classmethod
    def blank(cls, grid_size, channels=1, device=None):
        """An obstruction-free canvas -- overlay grows under pure self-organisation."""
        return cls(torch.zeros(1, channels, grid_size, grid_size, device=device))

    @classmethod
    def with_obstacles(cls, grid_size, obstacles, channels=1, device=None):
        """
        Build a grid with circular obstacles baked into channel 0.
        `obstacles` is a list of (centre_x, centre_y, radius) tuples in grid
        coordinates. Any extra channels beyond 0 are left at zero (free for
        you to use for other static context later, e.g. a background image).
        """
        tensor = torch.zeros(1, channels, grid_size, grid_size, device=device)
        yy, xx = torch.meshgrid(
            torch.arange(grid_size, device=device),
            torch.arange(grid_size, device=device),
            indexing="ij",
        )
        for cx, cy, radius in obstacles:
            blocked = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
            tensor[0, 0][blocked] = 1.0
        return cls(tensor)

    @classmethod
    def random_obstacles(cls, grid_size, count=4, min_radius=3, max_radius=6, margin=8, device=None):
        """
        Convenience generator: scatters `count` random circular obstacles,
        keeping a `margin`-radius clearing around the centre free so the
        seed always has room to start growing.
        """
        centre = grid_size // 2
        obstacles = []
        for _ in range(count):
            while True:
                cx = random.randint(0, grid_size - 1)
                cy = random.randint(0, grid_size - 1)
                if (cx - centre) ** 2 + (cy - centre) ** 2 > margin ** 2:
                    break
            obstacles.append((cx, cy, random.randint(min_radius, max_radius)))
        return cls.with_obstacles(grid_size, obstacles, channels=1, device=device)

    def to(self, device):
        self.tensor = self.tensor.to(device)
        return self

    @property
    def shape(self):
        return self.tensor.shape


class GCA(nn.Module):
    """
    Growing Cellular Automata model, split across two planes:

    - `grid`    : a STATIC substrate (see `Grid` above). The model perceives
                  it every step but never modifies it.
    - `overlay` : the DYNAMIC state. This is what grows, dies, and persists
                  across steps. It is the only thing the model ever writes to.

    Each forward() call perceives both planes together (so the growing
    pattern can react to whatever is on the static grid -- obstacles, a
    background image, etc.) but only produces a delta for the overlay.
    The grid is passed straight back through untouched; callers are
    responsible for keeping a stable reference to it across the rollout.
    """

    def __init__(self, overlay_channels=16, grid_channels=1, hidden_channels=96):
        super().__init__()
        self.overlay_channels = overlay_channels
        self.grid_channels = grid_channels

        # Perception is computed separately for grid and overlay, then concatenated,
        # so input width = (overlay_channels + grid_channels) * PERCEPTION_COUNT
        total_in_channels = (overlay_channels + grid_channels) * PERCEPTION_COUNT

        self.layers = nn.Sequential(
            nn.Conv2d(total_in_channels, hidden_channels, 1),
            nn.ReLU(),
            # Output channels = overlay_channels ONLY -- the model never produces
            # a delta for the grid plane.
            nn.Conv2d(hidden_channels, overlay_channels, 1, bias=False)
        )
        # Initialize weights of the last conv layer to zero, so the model starts
        # as a no-op (overlay delta = 0 everywhere) rather than scrambling the seed.
        nn.init.zeros_(self.layers[-1].weight)

    def forward(self, grid: torch.Tensor, overlay: torch.Tensor):
        """
        grid    : (1 or batch, grid_channels, H, W) -- static, read-only context
        overlay : (batch, overlay_channels, H, W)    -- the evolving state

        Returns the NEXT overlay state. Does not return or alter the grid --
        keep using the same `grid` tensor (or Grid.tensor) on every step.
        """
        # A single static grid is shared across an entire batch/pool, so it's
        # normally stored at batch size 1. Broadcast it here rather than
        # forcing every call site to expand it manually.
        if grid.shape[0] != overlay.shape[0]:
            grid = grid.expand(overlay.shape[0], -1, -1, -1)

        grid_perception = self.perception_conv(grid)
        overlay_perception = self.perception_conv(overlay)

        # Concatenate along channel dim so the dense layers see both planes at once
        combined_perception = torch.cat([grid_perception, overlay_perception], dim=1)

        # Pass through fully-connected layers -> delta is sized to overlay_channels
        delta = self.layers(combined_perception)
        # Stochastic update
        delta = self.mask(delta)

        # Apply delta to the OVERLAY only. The grid plane is never touched.
        new_overlay = overlay + delta
        # Alive cell masking is based on the overlay's own alpha channel -- the
        # grid plane has no concept of "alive" since it never changes.
        new_overlay = self.apply_alive_mask(new_overlay)
        # Hard physical constraint: the overlay cannot exist on blocked cells.
        # No-op when the grid is blank (occupancy field is all zero).
        new_overlay = self.apply_obstacle_mask(new_overlay, grid)
        return new_overlay

    def perception_conv(self, x):
        """Apply each perception convolution to a given plane's current state."""
        batches, channels, height, width = x.shape
        y = x.reshape(batches*channels, 1, height, width)
        # Circular pad the input to avoid losing information at the edges
        y = f.pad(y, [1, 1, 1, 1], 'circular')
        # Apply each perception convolution
        y = f.conv2d(y, PERCEPTIONS[:,None])
        # Reshape back to original shape
        return y.reshape(batches, -1, height, width)

    def mask(self, x, update_rate=0.5):
        """Stochastically mask updates to mimic the random updates found in biological cells."""
        batches, channels, height, width = x.shape
        mask = (torch.rand(batches, 1, height, width) + update_rate).floor()
        return x * mask

    def apply_alive_mask(self, overlay_state):
        """
        Applies alive mask to the overlay state ONLY.
        A cell is considered empty (zeroed out) if there is no mature
        (alpha > 0.1) overlay cell in its 3x3 neighbourhood. The grid plane
        is excluded entirely -- it doesn't have an alive/dead concept.
        """
        alive_mask = (
            f.max_pool2d(overlay_state[:, 3:4, :, :], kernel_size=3, stride=1, padding=1)
            > 0.1
        )
        return alive_mask * overlay_state

    def apply_obstacle_mask(self, overlay_state, grid):
        """
        Hard-enforces that the overlay cannot occupy obstructed cells.
        Grid channel 0 is treated as an occupancy field (> 0.5 = blocked).
        This holds before any training happens and regardless of what the
        network outputs -- it's a physical constraint on the simulation,
        not a learned preference. Note `grid` here is expected to already
        be batch-matched to `overlay_state` (forward() handles that).
        """
        free_mask = grid[:, 0:1, :, :] <= 0.5
        return overlay_state * free_mask

    def new_overlay_seed(self, batch_size, grid_size, device=None):
        """Single-pixel alive seed for the overlay plane, sized to overlay_channels."""
        seed = torch.zeros(batch_size, self.overlay_channels, grid_size, grid_size, device=device)
        seed[:, 3, grid_size // 2, grid_size // 2] = 1  # alpha channel = 1
        return seed

    def rgb(self, overlay):
        """Returns just the overlay's own RGB channels, clamped -- ignores the grid plane."""
        return overlay[:, :3].clamp(0, 1)


def composite(grid: torch.Tensor, overlay: torch.Tensor, obstacle_color=(0.15, 0.15, 0.15)):
    """
    Renders the overlay as a "filter" on top of the static grid for visualisation.
    The grid plane here represents occupancy (free/blocked), not colour content,
    so obstructed cells are painted `obstacle_color` as a dark block and
    everywhere else starts black; the overlay's own RGB is then alpha-blended
    on top using its own alpha channel, exactly like layering a sprite over a
    fixed background. (Obstructed cells should already read zero overlay alpha
    thanks to GCA.apply_obstacle_mask, but painting them explicitly keeps
    obstacles visible even before training, or if you inspect grid alone.)

    grid    : (1 or batch, grid_channels, H, W) -- channel 0 = occupancy field
    overlay : (batch, overlay_channels, H, W)
    Returns : (batch, 3, H, W) composited RGB image, values in [0, 1]
    """
    if grid.shape[0] != overlay.shape[0]:
        grid = grid.expand(overlay.shape[0], -1, -1, -1)

    overlay_rgb = overlay[:, :3].clamp(0, 1)
    overlay_alpha = overlay[:, 3:4].clamp(0, 1)

    obstacle_mask = (grid[:, 0:1] > 0.5).float()
    color = torch.tensor(obstacle_color, device=overlay.device, dtype=overlay.dtype).view(1, 3, 1, 1)
    background = color * obstacle_mask  # dark blocks where obstructed, black elsewhere

    return background * (1 - overlay_alpha) + overlay_rgb * overlay_alpha
