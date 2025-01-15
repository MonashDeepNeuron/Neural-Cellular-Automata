import torch
from torch import nn
import torch.nn.functional as f
from ImgCA import ImgCA

class HNCAImgModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.sensor = nn.Sequential( # Sensor taking output of child and transforming for something the parent can understand by downscaling by average
            nn.AvgPool2d(2, 2),
        )
        
        self.child = ImgCA(input_channels=12, output_channels=12, hidden_channels=64, trainable_perception=False)

        self.parent = ImgCA(input_channels=12, output_channels=9, hidden_channels=256, trainable_perception=True)


    def forward(self, x):

        x = self.child(x)
                
        y = self.sensor(x)

        y = self.parent(y)

        y = y.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)

        x[:, 2:, :, :]  = y + x[:, 2:, :, :]   # add the 9 signal channels from the child to the 3 channels

        return x

