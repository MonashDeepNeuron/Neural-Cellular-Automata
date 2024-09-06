import torch
import torch.nn as nn

class GCA(nn.Module):
    SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    SOBEL_Y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    IDENTITY = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    GRID_SIZE = 32

    def __init__(self, n_channels = 16, hidden_channels = 128): # TODO: UNDERSTAND HIDDEN_CHANNELS AND N_CHANNELS
        super().__init__()

        # Represent the update step as a submodule 
        # TODO: VERIFY THIS THOROUGHLY 
        self.update_network = nn.Sequential(
            nn.Conv2d(3* n_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, n_channels, kernel_size=1, bias=False))
 
        ## TODO: REGISTER EVERYTHING TO TORCH DEVICE TO USE GPU 

    def forward(self, input_grid):
        '''
        1. Construct `perception_grid`
        2. Apply update to each `perception_vector` in `perception_grid` to obtain `ds_grid`
        3. Apply stochastic update mask to `ds_grid`
        4. Obtain next state of grid from `ds_grid` + `state_grid`
        5. Apply alive cell masking to `state_grid`. This is the final updated state.  
        '''
        perception_grid = self.calculate_perception_grid(input_grid)
        ds_grid = self.calculate_ds_grid(perception_grid)
        filtered_ds_grid = self.apply_stochastic_mask(input_grid, ds_grid)

        output_state_grid = input_grid + filtered_ds_grid
        output_state_grid = self.apply_alive_mask(output_state_grid)

        return output_state_grid

    def calculate_perception_grid(self, state_grid):
        '''Calculates 1x48 perception vector for each cell in grid, returns as grid of perception vectors'''
        grad_x = nn.conv2d(self.sobel_x, state_grid)
        grad_y = nn.conv2d(self.sobel_y, state_grid)
        perception_grid = nn.concat(state_grid, grad_x, grad_y, axis = 2)

        return perception_grid 

    def calculate_ds_grid(self, perception_grid):
        '''Updates each perception vector in perception grid, return'''
        return nn.conv2d(self.update_network, perception_grid) # TODO: VERIFY THIS IS HOW TO APPLY UPDATE RULE TO EACH CELL 

    def apply_stochastic_mask(self, ds_grid):
        '''Applies stochastic update mask to ds_grid'''
        # Zero out (with 0.5 chance) a random fraction of the updates using a random mask
        rand_mask = (torch.rand(self.GRID_SIZE, self.GRID_SIZE) < 0.5).float()  
        
        return ds_grid * rand_mask

    def apply_alive_mask(self, state_grid):
        '''Applies alive mask to state_grid'''
        # Construct the boolean mask
        alive_mask = nn.functional.max_pool2d(state_grid[:, 3:4, :, :], kernel_size=3, stride=1, padding=1)> 0.1
        
        return alive_mask*state_grid
