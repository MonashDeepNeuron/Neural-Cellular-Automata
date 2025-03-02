import torch
from torch import nn
import torch.nn.functional as F

# Use GPU if available
if torch.cuda.is_available():
    torch.set_default_device("cuda")

def get_living_mask(x):
    # Extract alpha channel (assuming the last channel is the alpha channel)
    alpha = x[:,3,:,:,:]
    # Perform 3D max pooling with a kernel size of 3 and a stride of 1 (same padding)
    pooled = F.max_pool3d(alpha, kernel_size=3, stride=1, padding=1)
    # Return a mask where pooled values greater than 0.1 are considered 'alive'
    return (pooled > 0.1).float()

class NCA_3DV2(nn.Module):
    def __init__(self, channel):
        super(NCA_3DV2, self).__init__()
        
        # Define the layers
        self.conv1 = nn.Conv3d(in_channels=channel, out_channels=channel, kernel_size=3, padding=1) # learnable perception
        self.conv2 = nn.Conv3d(in_channels=channel, out_channels=channel*4, kernel_size=1, padding=0)
        self.conv3 = nn.Conv3d(in_channels=channel*4, out_channels=channel, kernel_size=1, padding=0)
    
    def forward(self, x, fire_rate=0.5):
        # Generate a random update mask 
        update_mask = torch.floor(torch.rand_like(x) + fire_rate)
        
        # Apply the convolutions to the input
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        
        # Apply the update mask and the living mask
        x = x * update_mask
        x = x * get_living_mask(x)  # Apply the living mask
        
        return x