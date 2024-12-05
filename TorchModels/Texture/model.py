import torch
from torch import nn
import torch.nn.functional as f

# Use GPU if available
if torch.cuda.is_available():
    torch.set_default_device('cuda')

# Define perceptions
IDENTITY = torch.tensor([[0,0,0],[0,1,0],[0,0,0]], dtype=torch.float)
SOBEL_X = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float)
SOBEL_Y = SOBEL_X.T
LAPLACIAN = torch.tensor([[1,2,1],[2,-12,2],[1,2,1]], dtype=torch.float)

# Create perception layer, consisting of the identity, sobel x, sobel y and laplacian
PERCEPTIONS = torch.stack([IDENTITY, SOBEL_X, SOBEL_Y, LAPLACIAN])
PERCEPTION_COUNT = PERCEPTIONS.shape[0]

class SelfOrganisingTexture(nn.Module):
    def __init__(self, channels=12, hidden_channels=96):
        super().__init__()

        self.channels = channels

        # Define dense layers (2D convolutions with kernel size of 1)
        self.layers = nn.Sequential(
            # Input channels = channels * # of perceptions
            nn.Conv2d(channels*PERCEPTION_COUNT, hidden_channels, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, channels, 1, bias=False)
        )

    def forward(self, x):
        # Create perception vector
        y = self.perception_conv(x)

        # Pass through fully-connected layers
        y = self.layers(y)

        # Stochastic update
        y = self.mask(y)

        # Return input + stochastic delta
        return x + y
    
    def perception_conv(self, x):
        """Apply each perception convolution to the current state."""
        # Reshape input to apply perception convolution
        batches, channels, height, width = x.shape
        y = x.reshape(batches*channels, 1, height, width)

        # Circular pad the input to avoid losing information at the edges
        y = f.pad(y, [1, 1, 1, 1], 'circular')

        # Apply each perception convolution
        y = f.conv2d(y, PERCEPTIONS[:,None])

        # Reshape back to original shape
        return y.reshape(batches, -1, height, width)
    
    def mask(self, x, update_rate = 0.5):
        """Stochastically mask updates to mimic the random updates found in biological cells."""
        # Uniformly mask across all channels
        batches, channels, height, width = x.shape
        mask = (torch.rand(batches, 1, height, width) + update_rate).floor()

        # Apply mask
        return x * mask
    
    def seed(self, n=256, size=128):
        """Creates an initial state for the model."""
        return torch.zeros(n, self.channels, size, size)
    
    def rgb(self, x):
        """Converts the model output to an RGB image."""
        # Return the first 3 channels, clamped between 0 and 1
        return x[:, :3].clamp(0, 1)
