import torch
import torch.nn as nn
import torch.nn.functional as F

class NCA(nn.Module):
    def __init__(self, input_channels=3, state_channels=64, hidden_channels=64, output_channels=3):
        super(NCA, self).__init__()
        
        self.initial_layer = nn.Conv2d(input_channels, state_channels, kernel_size=3, padding=1)
        
        self.hidden_layers = nn.Sequential(
            nn.Conv2d(state_channels, hidden_channels, kernel_size=3, padding=1),  # 3x3 convolution
            nn.InstanceNorm2d(hidden_channels),  
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),  # 1x1 convolution
            nn.InstanceNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),  # 1x1 convolution
            nn.InstanceNorm2d(hidden_channels),
            nn.ReLU(),
        )
        # Prediction layer to output RGB values
        self.prediction_layer = nn.Conv2d(hidden_channels, output_channels, kernel_size=1)

    def forward(self, state, steps=20):
        state = self.initial_layer(state)
        for _ in range(steps):
            hidden_state = self.hidden_layers(state)
            state = state + hidden_state  # Update state with hidden state; I think this is where im getting new colours from. I need to ensure that wheen the original image is added to the passed state; that the bounds are 
                                          # kept between only 255 for one column.
        rgb_output = self.prediction_layer(state)
        rgb_output = torch.sigmoid(rgb_output)  # Constrain output to [0, 1]
        return rgb_output