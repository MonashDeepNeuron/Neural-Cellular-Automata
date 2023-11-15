import torch
import torch.nn as nn

class GrowingCA(nn.Module):
    '''
    
    This class represents a growing cellular automata model, following
    the paper presented at https://distill.pub/2020/growing-ca/

    We will implement most functionality except for 'stochastic updating' in
    this initial model.

    Parameters:
    - n_channels : int
        - This represents the number of channels in our input grid/canvas.
    
    - hidden_channels : int
        - How many 1x1 convolutional kernels to we use
    
    - device : torch.device
        - Determines which device we perform computations on.

    '''
    def __init__(self, n_channels = 16, hidden_channels = 128, device = None):
        super().__init__()

        self.n_channels = n_channels
        self.device = device 

        #### Perception Step ###
        sobel_filter = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

        sobel_x = sobel_filter
        sobel_y = sobel_filter.t() 

        # They utilise an identity filter as well
        identity = torch.tensor([[0,0,0], [0,1,0], [0,0,0]])

        # Need to apply all three filters for every channel in our grid
        filters = torch.stack([identity, sobel_x, sobel_y]).repeat((n_channels, 1, 1))

        self.filters = filters[:, None, ...].type(torch.FloatTensor).to(self.device) # send these hardcoded filters to the device

        ### Update Step ###
        self.update_step = nn.Sequential(
            nn.Conv2d(
                3 * n_channels, # We have 3 * n_channels as we are apply 3 kernels to each channel (perception step)
                hidden_channels,
                kernel_size = 1, # applying 1x1 convs
            ),
            nn.ReLU(),
            nn.Conv2d(
                hidden_channels,
                n_channels, # output channels = input channels
                kernel_size = 1,
            )
        )

        #self.to(self.device) # send this model to device

    def perceive(self, x):
        '''
        
        The authors utilise 3x3 sobel filters to estimate partial derivatives
        of cell state channels to get a gradient vector in each direction for
        each cell state, and combine this with the input.

        This is the only place in the model that utilises information from 
        neighbouring cells. Note that there are no learnable parameters.

        Parameters:
        - x : torch.Tensor
            - This is our input grid (batch_size, n_channels, width, height)

        Returns:
        - torch.Tensor
            - After applying the 3 hardcoded kernels (batch_size, 3 * n_channels, width, height)

        '''
        return nn.functional.conv2d(x, self.filters, padding=1, groups=self.n_channels) # we apply padding to keep dimensionality
    
    def update(self, x):
        '''
        
        This represents our update step. It is the only place in this model with
        learnable parameters.

        Parameters:
        - x : torch.Tensor
            - This is our grid after perception (batch_size, 3 * n_channels, width, height)

        Returns:
        - torch.Tensor
            - Updated grid (batch_size, n_channels, width, height)

        '''
        return self.update_step(x)
    
    @staticmethod
    def is_alive_mask(x):
        '''
        
        This method returns alive cells.

        Parameters:
        - x : torch.Tensor
            - This is our current grid (batch_size, n_channels, width, height)

        Returns:
        - torch.Tensor
            - Boolean grid returning alive cells (batch_size, n_channels, width, height)

        '''
        return (nn.functional.max_pool2d(x[:,3:4,:,:], kernel_size=3, stride=1, padding=1)>0.1) # remember the criterion uses alpha (4th channel)
    
    def forward(self, x):
        '''
        
        Perform the forward pass of the model.
        
        Parameters:
        - x : torch.Tensor
            - This is our current grid (batch_size, n_channels, width, height)

        Returns:
        - torch.Tensor
            - Updated grid (batch_size, n_channels, width, height)

        '''
        pre_mask = self.is_alive_mask(x)

        p_out = self.perceive(x)
        dx = self.update(p_out)

        x = x + dx # updated grid is equal to previous grid + change

        post_mask = self.is_alive_mask(x)
        is_alive = (pre_mask & post_mask).to(torch.float32)

        return x * is_alive

