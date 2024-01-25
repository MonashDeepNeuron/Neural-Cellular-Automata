import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import os

def render_video():
    # Turn the images in steps/ into a video with ffmpeg
    os.system("ffmpeg -y -v 0 -framerate 24 -i steps/%05d.jpeg video.mp4")

def make_video(x, model, n_steps=100):
    '''
    
    Helper function to visualise results (aside from tensorboard)

    '''
    os.mkdir("GrowingCA/results/steps", mode=0o777)
    for i in range(n_steps):
        with torch.no_grad():
            x = model(x)
            img = to_rgb(x)
            img = img.detach().cpu().clip(0, 1).squeeze().permute(1, 2, 0)
            img = Image.fromarray(np.array(img*255).astype(np.uint8))
            img.save(f'steps/{i:05}.jpeg')
    render_video()

def load_target(path, im_size=32):
    '''
    
    Load target image.

    Parameters:
    - path : string
        - Path to image (RGBA)

    - im_size : int
        - Size to resize image (we'll use square images for now)

    Returns:
    - torch.tensor
        - Our image in tensor format.
    '''
    img = Image.open(path)
    img = img.resize((im_size, im_size))
    img = np.float32(img) / 255.0
    img[..., :3] *= img[..., 3:]

    return torch.from_numpy(img).permute(2, 0, 1)[None, ...]

def to_rgb(img_rgba):
    '''
    
    Convert RGBA image to RGB.

    '''
    rgb, a = img_rgba[:, :3, ...], torch.clamp(img_rgba[:, 3:, ...], 0, 1)
    return torch.clamp(1.0 - a + rgb, 0, 1)

def starting_seed(size, n_channels):
    '''
    
    Create a starting tensor for training. Note that when starting, the 
    only active pixels are going to be in the middle of the grid.

    Parameters:
    - size: int
        - height/width of grid

    - n_channels: int
        - Number of input channels

    Returns:
    - torch.Tensor
        - Seed (1, n_channels, size, size)

    '''
    x = torch.zeros((1, n_channels, size, size), dtype=torch.float32)
    x[:, 3:, size // 2, size // 2] = 1
    return x
