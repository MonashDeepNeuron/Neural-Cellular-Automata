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
LAPLACIAN = torch.tensor([[1,2,1],[2,-12,2],[1,2,1]], dtype=torch.float)    # Detects sharp changes in intensity

# Create perception layer, consisting of the identity, sobel x, sobel y and laplacian
PERCEPTIONS = torch.stack([IDENTITY, SOBEL_X, SOBEL_Y, LAPLACIAN])
PERCEPTION_COUNT = PERCEPTIONS.shape[0]

class ImgCA(nn.Module):
    def __init__(self, input_channels=12, output_channels=12, hidden_channels=64, trainable_perception=False):
        super().__init__()

        self.channels = input_channels
        print(PERCEPTIONS.shape, PERCEPTION_COUNT)

        # self.bn0 = nn.BatchNorm2d(input_channels)
        # self.bn1 = nn.BatchNorm2d(input_channels * PERCEPTION_COUNT)
        # self.bn2 = nn.BatchNorm2d(hidden_channels)

        # Define dense layers (2D convolutions with kernel size of 1)
        self.layers = nn.Sequential(
            # Input channels = channels * # of perceptions
            # self.bn1,
            nn.Conv2d(input_channels*PERCEPTION_COUNT, hidden_channels, 1),
            nn.ReLU(),
            # self.bn2,
            nn.Conv2d(hidden_channels, output_channels, 1, bias=False)
        )

        # Initialize weights of the last conv layer to zero
        nn.init.zeros_(self.layers[-1].weight)

    def forward(self, x):
        # Create perception vector
        # y = self.bn0(x)

        y = self.perception_conv(x)

        # Pass through fully-connected layers
        y = self.layers(y)

        # Stochastic update
        y = self.mask(y)

        # Return input + stochastic delta
        return y
    
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
