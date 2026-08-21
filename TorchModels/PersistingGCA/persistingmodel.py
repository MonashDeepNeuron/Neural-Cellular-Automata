import torch
import torch.nn as nn
import torch.nn.functional as f


class GCA(nn.Module):
    """
    Persisting variation of Growing Neural Cellular Automata. Minimal changes have been made, except
    Dropout replacing manual stochastic mask.
    """
    SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
    SOBEL_Y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
    IDENTITY = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)

    def __init__(self, n_channels=16, hidden_channels=128):
        ## Hidden channels are the number of channels in the linear layer in network
        ## Hidden channels are the number of channels in the linear layer in network
        super().__init__()

        ## Represent the update step as a submodule
        ## The stochastic update mask is applied in forward() (per-cell, active in
        ## both train and eval) rather than via nn.Dropout, so train and inference
        ## dynamics match -- a requirement for the pattern to persist.
        self.update_network = (
            nn.Sequential(  # pytorch Conv2d layers automatically parallelize
                nn.Conv2d(3 * n_channels, hidden_channels, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(hidden_channels, n_channels, kernel_size=1, bias=False),
            )
        )

        ## Small init on the first layer; zero init on the last layer so the model
        ## starts as a "do-nothing" update (canonical NCA), which stabilises early training.
        torch.nn.init.normal_(self.update_network[0].weight, mean=0.0, std=0.001)
        torch.nn.init.zeros_(self.update_network[0].bias)
        torch.nn.init.zeros_(self.update_network[2].weight)

    def to(self, device):
        """
        Overrides parent nn.Module to method for sending resources to GPU
        Move the model and constant tensors to the device (GPU)"""
        """
        Overrides parent nn.Module to method for sending resources to GPU
        Move the model and constant tensors to the device (GPU)"""
        self.SOBEL_X = self.SOBEL_X.to(device)
        self.SOBEL_Y = self.SOBEL_Y.to(device)
        self.IDENTITY = self.IDENTITY.to(device)
        return super().to(device)

    def forward(self, input_grid):
        """
        Input_grid is tensor with dims: (batch, in_channels, height, width)
        1. Construct `perception_grid` by replacing each cell in input_grid with its feature vector
        2. Apply update to each `perception_vector` in `perception_grid` to obtain `ds_grid`, the grid of changes
        3. Apply stochastic update mask to `ds_grid` to filter out some changes
        Input_grid is tensor with dims: (batch, in_channels, height, width)
        1. Construct `perception_grid` by replacing each cell in input_grid with its feature vector
        2. Apply update to each `perception_vector` in `perception_grid` to obtain `ds_grid`, the grid of changes
        3. Apply stochastic update mask to `ds_grid` to filter out some changes
        4. Obtain next state of grid from `ds_grid` + `state_grid`
        5. Apply alive cell masking to `state_grid` to kill of cells with alpha < 0.1
        This yields output_filtered_grid, a tensor with dims: (batch, in_channels, height, width)
        5. Apply alive cell masking to `state_grid` to kill of cells with alpha < 0.1
        This yields output_filtered_grid, a tensor with dims: (batch, in_channels, height, width)
        """

        ## Add input grid to the device model parameters are on
        input_grid = input_grid.to(next(self.parameters()).device)

        ## Add input grid to the device model parameters are on
        input_grid = input_grid.to(next(self.parameters()).device)

        ## Cells alive BEFORE the update (alpha-based, includes 3x3 neighbourhood)
        pre_life_mask = self.alive_mask(input_grid)

        perception_grid = self.calculate_perception_grid(input_grid)
        ds_grid = self.calculate_ds_grid(perception_grid)

        ## Per-cell stochastic update mask: each cell updates with prob 0.5, the SAME
        ## mask shared across all channels of that cell. Active in train AND eval so
        ## the dynamics are identical at training and inference time.
        update_mask = (torch.rand_like(ds_grid[:, :1, :, :]) <= 0.5).to(ds_grid.dtype)
        output_raw_grid = input_grid + ds_grid * update_mask

        ## A cell is kept only if it was alive both before and after the update.
        post_life_mask = self.alive_mask(output_raw_grid)
        life_mask = (pre_life_mask & post_life_mask).to(output_raw_grid.dtype)
        return output_raw_grid * life_mask

    def calculate_perception_grid(self, state_grid):
        """
        Calculates 1x48 perception vector for each cell in grid, returns as grid of perception vectors.
        Perception vectors are 4 dimensional. Unsqueeze used to add dimension of size 1 at index
        """

        state_grid_padded = f.pad(state_grid, (1, 1, 1, 1), mode="circular")

        grad_x = f.conv2d(
            state_grid_padded,
            self.SOBEL_X.unsqueeze(0).repeat(state_grid.size(1), 1, 1, 1),
            stride=1,
            padding=0,
            groups=state_grid_padded.size(1),
        ) 
        grad_y = f.conv2d(
            state_grid_padded,
            self.SOBEL_Y.unsqueeze(0).repeat(state_grid.size(1), 1, 1, 1),
            stride=1,
            padding=0,
            groups=state_grid_padded.size(1),
        )

        perception_grid = torch.cat([state_grid, grad_x, grad_y], dim=1)

        return perception_grid

    def calculate_ds_grid(self, perception_grid):
        """Updates each perception vector in perception grid, return
        We can apply the submodule network like a function"""
        return self.update_network(perception_grid)


    def alive_mask(self, state_grid):
        """Boolean (batch, 1, H, W) mask of living cells.
        A cell is alive if any cell in its 3x3 neighbourhood is mature (alpha > 0.1).
        """
        return (
            f.max_pool2d(state_grid[:, 3:4, :, :], kernel_size=3, stride=1, padding=1)
            > 0.1
        )
