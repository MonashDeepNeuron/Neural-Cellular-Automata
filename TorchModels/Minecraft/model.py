import torch
from torch import nn
import torch.nn.functional as f

# Use GPU if available
if torch.cuda.is_available():
    torch.set_default_device("cuda")

# Define perceptions
SOBEL_X = torch.tensor(
    [
        [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
    ],
    dtype=torch.float32,
)
SOBEL_Y = -torch.rot90(SOBEL_X, k=-1, dims=[0, 1])  # GPT
SOBEL_Z = SOBEL_X.permute(2, 1, 0)  # GPT
IDENTITY = torch.zeros((3, 3, 3), dtype=torch.float32)
IDENTITY[1, 1, 1] = 1

# Create perception layer, consisting of the identity, sobel x, sobel y and laplacian
PERCEPTIONS = torch.stack([IDENTITY, SOBEL_X, SOBEL_Y, SOBEL_Z])
PERCEPTION_COUNT = PERCEPTIONS.shape[0]


class NCA_3D(nn.Module):
    def __init__(self, channels=16, hidden_channels=16):
        super().__init__()

        self.channels = channels

        # Define dense layers (2D convolutions with kernel size of 1)
        self.layers = nn.Sequential(
            # Input channels = channels * # of perceptions
            nn.Conv3d(channels * PERCEPTION_COUNT, hidden_channels, 1),
            nn.ReLU(),
            nn.Conv3d(hidden_channels, channels, 1, bias=False),
        )

        ## Initialise model parameters as much smaller numbers
        torch.nn.init.normal_(self.layers[0].weight, mean=0.0, std=0.001)
        torch.nn.init.normal_(self.layers[0].bias, mean=0.0, std=0.001)
        torch.nn.init.normal_(self.layers[2].weight, mean=0.0, std=0.001)

    def forward(self, x):
        # Create perception vector
        y = self.perception_conv(x)

        # Pass through fully-connected layers
        y = self.layers(y)

        # Stochastic update
        y = self.mask(y)

        # Input + stochastic delta
        y = x + y

        # Alive mask
        y = self.alive_mask(y)
        return y

    def perception_conv(self, x):
        """Apply each perception convolution to the current state."""
        # Reshape input to apply perception convolution
        batches, channels, height, width, depth = x.shape

        y = x.reshape(batches * channels, 1, height, width, depth)
        # Circular pad the input to avoid losing information at the edges
        y = f.pad(y, [1, 1,  1, 1, 1, 1], "circular")

        # Apply each perception convolution
        y = f.conv3d(y, PERCEPTIONS[:, None])

        # Reshape back to original shape
        return y.reshape(batches, -1, height, width, depth)

    def mask(self, x, update_rate=0.5):
        """Stochastically mask updates to mimic the random updates found in biological cells."""
        # Uniformly mask across all channels
        batches, channels, height, width, depth = x.shape
        mask = (torch.rand(batches, 1, height, width, depth) + update_rate).floor()

        # Apply mask
        return x * mask

    '''
    def seed(self, n=256, size=128):
        """Creates an initial state for the model."""
        return torch.zeros(n, self.channels, size, size)'''

    def rgb(self, x):
        """Converts the model output to an RGB image."""
        # Return the first 3 channels, clamped between 0 and 1
        return x[:, :3].clamp(0, 1)

    def alive_mask(self, x):
        """Applies alive mask to state_grid
        Does not use GPU to generate alive mask as doing max pooling
        A cell is considered empty (set rgba : 0) if there is no mature (alpha > 0.1) cell in its 3x3 neighbourhood
        """
        mask = (
            f.max_pool3d(x[:, 3:4, :, :, :], kernel_size=3, stride=1, padding=1) > 0.1
        )

        # Apply mask
        return x * mask
