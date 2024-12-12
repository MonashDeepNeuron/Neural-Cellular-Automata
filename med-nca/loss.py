

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, input, target):
        # Flatten the input and target tensors to compute the loss
        input_flat = input.view(-1)
        target_flat = target.view(-1)

        intersection = (input_flat * target_flat).sum()
        dice_score = (2. * intersection + self.smooth) / (input_flat.sum() + target_flat.sum() + self.smooth)
        
        # Dice Loss is 1 - Dice Score
        return 1 - dice_score