import torch
from torch import nn
import torch.nn.functional as f
from ImgCA import ImgCA

class HNCAImgModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.channels = 12

        self.sensor = nn.Sequential( # Sensor taking output of child and transforming for something the parent can understand by downscaling by average
            nn.AvgPool2d(2, 2),
        )
        
        self.child = ImgCA(input_channels=self.channels, output_channels=self.channels, hidden_channels=64, trainable_perception=False)

        self.parent = ImgCA(input_channels=self.channels, output_channels=self.channels-3, hidden_channels=256, trainable_perception=True)


    def forward(self, x):

        x = x + self.child(x)
                
        y = self.sensor(x)

        y = self.parent(y)

        y = y.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)

        x = torch.cat((x[:, :3, :, :], y + x[:, 3:, :, :]), dim=1)  # add the 9 signal channels from the child to the 3 channels

        return x

    
    def seed(self, n=256, size=128):
        """Creates an initial state for the model."""
        return torch.zeros(n, self.channels, size, size)
    
    def rgb(self, x):
        """Converts the model output to an RGB image."""
        # Return the first 3 channels, clamped between 0 and 1
        return x[:, :3].clamp(0, 1)
