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

SOBEL_X = SOBEL_X.unsqueeze(0) # needs shape (out_channels, in_channels, X, Y, Z). Repeat will be used in perception to handle the channels when channels != 1
SOBEL_Y = SOBEL_Y.unsqueeze(0)
SOBEL_Z = SOBEL_Z.unsqueeze(0)
# IDENTITY = torch.zeros((3, 3, 3), dtype=torch.float32)
# IDENTITY[1, 1, 1] = 1
# identity not really needed

# Create perception layer, consisting of the identity, sobel x, sobel y and laplacian
# PERCEPTIONS = torch.stack([SOBEL_X, SOBEL_Y, SOBEL_Z])
# PERCEPTION_COUNT = PERCEPTIONS.shape[0]


class NCA_3D(nn.Module):
    def __init__(self, n_channels=16, hidden_channels=32):
        super().__init__()

        self.channels = n_channels

        # Define dense layers (3D convolutions with kernel size of 1)
        # For simple model, use one hidden layer
        self.layers = nn.Sequential(
            nn.Conv3d(n_channels * 4, hidden_channels, 1), # need to multiply by 4, not 3, due to added dimension Z
            nn.ReLU(),
            nn.Conv3d(hidden_channels, n_channels, 1, bias=False),
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

        # Circular pad the input to avoid losing information at the edges
        padded_grid = f.pad(x, [1, 1, 1, 1, 1, 1], "circular")

        grad_x = f.conv3d(
            padded_grid,
            SOBEL_X.repeat(padded_grid.size(1), 1, 1, 1, 1), # It repeats the sobel_x for each channel
            stride=(1,1,1), # specify stride as a tuple/list for each dimension
            padding=0,
            groups=padded_grid.size(1), # again, useful when considering more than 1 channel
        ) 
        grad_y = f.conv3d(
            padded_grid,
            SOBEL_Y.repeat(padded_grid.size(1), 1, 1, 1, 1), # It repeats the sobel_y for each channel
            stride=(1,1,1),
            padding=0,
            groups=padded_grid.size(1), # again, useful when considering more than 1 channel
        )  
        grad_z = f.conv3d(
            padded_grid,
            SOBEL_Z.repeat(padded_grid.size(1), 1, 1, 1, 1), # It repeats the sobel_z for each channel
            stride=(1,1,1),
            padding=0,
            groups=padded_grid.size(1), # again, useful when considering more than 1 channel
        )

        perception_output = torch.cat([grad_x, grad_y, grad_z, x], dim=1) # just pass x, rather than applying identity. Output should be (BATCH_SIZE, 4*IN_CHANNELS, X, Y, Z)
        return perception_output

    def mask(self, x, update_rate=0.25):
        """Stochastically mask updates to mimic the random updates found in biological cells."""
        # Uniformly mask across all channels
        batches, channels, height, width, depth = x.shape
        mask = (torch.rand(batches, channels, height, width, depth) + update_rate).floor() # mask needs to be same shape

        # Apply mask
        return x * mask

    def alive_mask(self, x):
        """Applies alive mask to state_grid
        Does not use GPU to generate alive mask as doing max pooling
        A cell is considered empty (set rgba : 0) if there is no mature (alpha > 0.1) cell in its 3x3 neighbourhood
        """
        mask = (
            f.max_pool3d(x[:, 3, :, :, :], kernel_size=3, stride=1, padding=1) > 0.1
        )

        # Apply mask
        return x * mask
