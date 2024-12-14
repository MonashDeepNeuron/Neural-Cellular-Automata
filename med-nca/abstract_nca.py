import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
 
class BasicNCA(nn.Module):
    r"""Basic implementation of an NCA using a sobel x and y filter for the perception
    """
    def __init__(self, channel_n, fire_rate, device, hidden_size=128, input_channels=1, init_method="standard"):
        r"""Init function
            #Args:
                channel_n: number of channels per cell
                fire_rate: random activation of each cell
                device: device to run model on
                hidden_size: hidden size of model
                input_channels: number of input channels
                init_method: Weight initialisation function
        """
        super(BasicNCA, self).__init__()

        self.device = device
        self.channel_n = channel_n
        self.input_channels = input_channels

        self.fc0 = nn.Conv2d(channel_n*3, hidden_size, kernel_size = 1)
        self.fc1 = nn.Conv2d(hidden_size, channel_n, kernel_size=1, bias=False)

        with torch.no_grad():
            self.fc1.weight.zero_()

        if init_method == "xavier":
            torch.nn.init.xavier_uniform(self.fc0.weight)
            torch.nn.init.xavier_uniform(self.fc1.weight)

        self.fire_rate = fire_rate
        self.to(self.device)

    def perceive(self, x):
        r"""Perceptive function, combines 2 sobel x and y outputs with the identity of the cell
            #Args:
                x: image
        """
        def _perceive_with(x, weight):
            conv_weights = torch.from_numpy(weight.astype(np.float32)).to(self.device)
            conv_weights = conv_weights.view(1,1,3,3).repeat(self.channel_n, 1, 1, 1)
            return F.conv2d(x, conv_weights, padding=1, groups=self.channel_n)

        dx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) # Sobel filter
        dy = dx.T

        y1 = _perceive_with(x, dx)
        y2 = _perceive_with(x, dy)
        y = torch.cat((x,y1,y2),1)
        return y

    def update(self, x_in, fire_rate):
        r"""Update function runs same nca rule on each cell of an image with a random activation
            #Args:
                x_in: image
                fire_rate: random activation of cells
        """
        x = x_in.transpose(1,3)

        ds = self.perceive(x) # ds = difference in state
        ds = self.fc0(ds)
        ds= F.relu(ds)
        ds = self.fc1(ds)

        if fire_rate is None:
            fire_rate=self.fire_rate
        stochastic = torch.rand([ds.size(0),ds.size(1),ds.size(2),1])>fire_rate
        stochastic = stochastic.float().to(self.device)
        ds = ds * stochastic

        x = x+ds

        x = x.transpose(1,3)

        return x

    def forward(self, x, steps=64, fire_rate=0.5):
        r"""Forward function applies update function s times leaving input channels unchanged
            #Args:
                x: image
                steps: number of steps to run update
                fire_rate: random activation rate of each cell
        """
        for step in range(steps):
            x2 = self.update(x, fire_rate).clone() #[...,3:][...,3:]
            x = torch.concat((x[...,:self.input_channels], x2[...,self.input_channels:]), 3)
        return x
