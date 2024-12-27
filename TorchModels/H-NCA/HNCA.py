
import torch
from torch import nn
import torch.nn.functional as f
from ImgCA import ImgCA

class HCAImgModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.leaf_ca_model = ImgCA(n_channels=3, n_schannels=9, hidden_channels=64, trainable_perception=False)
        
        '''state = self.leaf_ca_model.seed(n=1, size=128)
        output = self.leaf_ca_model.step(state, s=self.leaf_ca_model.n_schannels, n_steps=300, update_rate=0.5)
        rgb_image = self.leaf_ca_model.rgb(output)
        print(rgb_image.shape)  # Should be [1, 3, 128, 128]'''


        self.parent_ca_model = ImgCA(n_channels=12, n_schannels=0, hidden_channels=256, trainable_perception=True)
    
        '''state = self.parent_ca_model.seed(n=1, size=256)
        output = self.parent_ca_model.step(state, self.parent_ca_model.n_schannels, n_steps=300, update_rate=0.5)
        rgb_image = self.parent_ca_model.rgb(output)
        print(rgb_image.shape)  # Should be [1, 3, height, width]'''


model = HCAImgModel()
print("model initialized")
