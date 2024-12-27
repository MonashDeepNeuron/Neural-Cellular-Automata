import torch
from torch import nn
import torch.nn.functional as f

# Use GPU if available
if torch.cuda.is_available():
    torch.set_default_device('cuda')

# Define static perceptions (not required for parent NCA)
IDENTITY = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float)       # Maintains the current state
SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float)     # Detects horizontal edges
SOBEL_Y = SOBEL_X.T                                                                # Detects vertical edges
LAPLACIAN = torch.tensor([[1, 2, 1], [2, -12, 2], [1, 2, 1]], dtype=torch.float)    # Detects sharp intensity changes

# Stack static perceptions
PERCEPTIONS = torch.stack([IDENTITY, SOBEL_X, SOBEL_Y, LAPLACIAN])
PERCEPTION_COUNT = PERCEPTIONS.shape[0]

class ImgCA(nn.Module):
    def __init__(self, n_channels=12, n_schannels=0, hidden_channels=128, trainable_perception=False):
        # hidden channels is the same as features from the paper
        super().__init__()

        self.n_channels = n_channels
        self.n_schannels = n_schannels
        self.trainable_perception = trainable_perception

        # Define trainable perception if enabled
        if self.trainable_perception:
            self.perceptions = nn.Parameter(PERCEPTIONS[None, None, :, :, :].clone(), requires_grad=True)
        else:
            self.register_buffer('perceptions', PERCEPTIONS[None, None, :, :, :])

        # Batch normalization and layers
        self.bn0 = nn.BatchNorm2d(n_channels + n_schannels)
        self.bn1 = nn.BatchNorm2d((n_channels + n_schannels )* PERCEPTION_COUNT) 
        self.bn2 = nn.BatchNorm2d(hidden_channels)

        self.perception = nn.Conv2d(
            in_channels=n_channels + n_schannels,
            out_channels=(n_channels + n_schannels) * PERCEPTION_COUNT,
            kernel_size=3,
            groups=n_channels + n_schannels,
            padding=0,
            bias=False
        )

        self.features = nn.Conv2d(
            in_channels=(n_channels + n_schannels) * PERCEPTION_COUNT,
            out_channels=hidden_channels,
            kernel_size=1,
            bias=True
        )

        self.new_state = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=n_channels + n_schannels,
            kernel_size=1,
            bias=False
        )

        # Initialize weights of the last layer to zero
        nn.init.zeros_(self.new_state.weight)

    def forward(self, x, s=None, update_rate=0.5):
        if s is not None:
            x = torch.cat([x, s], dim=1)
        
        x = self.circular_pad(x, pad=1)
        x = self.bn0(x)

        y = self.perception(x)
        
        y = self.bn1(y)
        
        y = self.features(y)
        
        y = self.bn2(y)
        
        y = self.new_state(y)

        update_mask = (torch.rand_like(y[:, :1]) + update_rate).floor()
        y = y * update_mask
        return y

    def step(self, x_initial, s=None, n_steps=50, update_rate=0.5):
        """
        Iteratively updates the state `n_steps` times.
        """
        x = x_initial

        # If s is an integer, convert it to a tensor with the correct shape
        if isinstance(s, int):
            s = torch.zeros(
                x.size(0),  # batch size
                s,          # number of signal channels
                x.size(2),  # height
                x.size(3),  # width
                device=x.device
            )

        for _ in range(n_steps):
            x = self.forward(x, s, update_rate=update_rate)
            s = None  # Signal is only used in the first step
        return x


    def circular_pad(self, x, pad):
        """
        Applies circular padding to avoid information loss at edges.
        """
        return f.pad(x, (pad, pad, pad, pad), mode='circular')

    def seed(self, n=256, size=128):
        """
        Creates an initial state for the model.
        """
        return torch.zeros(n, self.n_channels, size, size, device=next(self.parameters()).device)

    def mask_signal_channels(self, x):
        """
        Masks the signal channels with zeros, keeping only feature channels active.
        """
        feature_channels = self.n_channels
        x[:, feature_channels:] = 0
        return x

    def rgb(self, x):
        """
        Converts the model output to an RGB image.
        """
        # Return the first 3 channels, clamped between 0 and 1
        return x[:, :3].clamp(0, 1)
