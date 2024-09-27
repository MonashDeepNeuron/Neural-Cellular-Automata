import torch
import torch.nn as nn
import torch.nn.functional as f


class GCA(nn.Module):
    SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
    SOBEL_Y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
    IDENTITY = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)
    GRID_SIZE = 32

    def __init__(self, n_channels=16, hidden_channels=128):
        # hidden channels are the number of channels in the linear layer in network
        super().__init__()

        # Represent the update step as a submodule
        self.update_network = (
            nn.Sequential(  # pytorch Conv2d layers automatically parallelize
                nn.Conv2d(3 * n_channels, hidden_channels, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(hidden_channels, n_channels, kernel_size=1, bias=False),
            )
        )

        torch.nn.init.normal_(self.update_network[0].weight, mean=0.0, std=0.001, generator=None)
        torch.nn.init.normal_(self.update_network[0].bias, mean=0.0, std=0.001, generator=None)
        torch.nn.init.normal_(self.update_network[2].weight, mean=0.0, std=0.001, generator=None)
        

    def to(self, device):
        """Move the model and constant tensors to the device (GPU)"""
        self.SOBEL_X = self.SOBEL_X.to(device)
        self.SOBEL_Y = self.SOBEL_Y.to(device)
        self.IDENTITY = self.IDENTITY.to(device)
        return super().to(device)

    def forward(self, input_grid):
        """
        1. Construct `perception_grid`
        2. Apply update to each `perception_vector` in `perception_grid` to obtain `ds_grid`
        3. Apply stochastic update mask to `ds_grid`
        4. Obtain next state of grid from `ds_grid` + `state_grid`
        5. Apply alive cell masking to `state_grid`. This is the final updated state.
        """
        perception_grid = self.calculate_perception_grid(input_grid)
        ds_grid = self.calculate_ds_grid(perception_grid)
        filtered_ds_grid = self.apply_stochastic_mask(ds_grid)
        output_raw_grid = input_grid + filtered_ds_grid
        output_filtered_grid = self.apply_alive_mask(output_raw_grid)
        return output_filtered_grid

    def calculate_perception_grid(self, state_grid):
        """
        Calculates 1x48 perception vector for each cell in grid, returns as grid of perception vectors.
        Perception vectors are 4 dimensional. Unsqueeze used to add dimension of size 1 at index
        
        INPUTS: 
            stategrid: Tensor wiith dims (batch, in_channels, height, width)

            self (relevant components): 
                SOBEL_X 
        """
        # print(state_grid.size())
        # print(self.SOBEL_X.unsqueeze(0).repeat(state_grid.size(1), 1, 1, 1).size())
        grad_x = f.conv2d(state_grid, self.SOBEL_X.unsqueeze(0).repeat(state_grid.size(1), 1, 1, 1), stride=1, padding=1, groups = state_grid.size(1)) # TODO Replace padding with logic to make a grid that wraps around on itself.
        grad_y = f.conv2d(state_grid, self.SOBEL_X.unsqueeze(0).repeat(state_grid.size(1), 1, 1, 1), stride=1, padding=1, groups = state_grid.size(1)) # TODO Replace 16 with a dynamic channels variable?

        # print(grad_x.size())

        perception_grid = torch.cat([state_grid, grad_x, grad_y], dim=1)
        # print(perception_grid.size())
        return perception_grid

    def calculate_ds_grid(self, perception_grid):
        """Updates each perception vector in perception grid, return
        We can apply the submodule network like a function"""
        return self.update_network(perception_grid)

    def apply_stochastic_mask(self, ds_grid):
        """Applies stochastic update mask to ds_grid
        Zero out (with 0.5 chance) a random fraction of the updates using a random mask
        Uses the GPU for random mask generation
        """
        rand_mask = (
            torch.rand(self.GRID_SIZE, self.GRID_SIZE, device=ds_grid.device) < 0.5
        ).float()
        return ds_grid * rand_mask

    def apply_alive_mask(self, state_grid):
        """Applies alive mask to state_grid
        Does not use GPU to generate alive mask as doing max pooling"""
        alive_mask = (
            f.max_pool2d(state_grid[:, 3:4, :, :], kernel_size=3, stride=1, padding=1)
            > 0.1
        ) ## [:, 3:4, :, :] originally

        ## a cell is considered empty (set rgba : 0) 
        # if there is no mature (alpha > 0.1) cell in its 3x3 neighbourhood

        # print(f"alive mask: {alive_mask.size()}")

        return alive_mask * state_grid
