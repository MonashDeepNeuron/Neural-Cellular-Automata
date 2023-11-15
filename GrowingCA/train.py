import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from PIL import Image

from model import GrowingCA

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
    img = np.float32(img) / 255.0 # normalize
    img[..., :3] *= img[..., 3:] # premultiply RGB by Alpha (this is what the paper dide)

    return torch.from_numpy(img).permute(2,0,1)[None, ...]

def to_rgb(img_rgba):
    '''
    
    Convert RGBA image to RGB.

    '''
    rgb, a = img_rgba[:, :3, ...], torch.clamp(img_rgba[:, 3:, ...], 0, 1)
    return torch.clamp(1.0 - a + rgb, 0 , 1) # this is what paper did

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
    x[:, 3:, size//2, size//2] = 1 # set center pixel to 1 for all channels except RGB
    return x

def train():
    # I'll add a parser later
    BATCH_SIZE = 4

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logdir = "GrowingCA\logs"

    tensorboard_writer = SummaryWriter(logdir)

    # Target 
    img_path = "cat.png"
    img_size = 32
    target_img = load_target(img_path, im_size=img_size)
    target_img = target_img.to(device) # send image to device
    target_img == target_img.repeat(BATCH_SIZE, 1, 1, 1) # we perform batch training with same image

    tensorboard_writer.add_image("Target Image", to_rgb(target_img)[0])

    # Model
    n_channels = 16
    hidden_channels = 128
    device = device
    model = GrowingCA(n_channels=n_channels, hidden_channels=hidden_channels, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Pool to sample from
    pool_size = 1024
    seed = starting_seed(size=img_size, n_channels=n_channels).to(device)
    pool = seed.clone().repeat(pool_size,1,1,1)

    epochs = 5000
    eval_frequency = 500
    eval_iterations = 300

    for epoch in tqdm(range(epochs)):
        batch_idxs = np.random.choice(pool_size, BATCH_SIZE, replace=False).tolist()

        x = pool[batch_idxs]
        for i in range(np.random.randint(64,96)):
            x = model(x)

        loss_batch = ((target_img - x[:, :4, ...] ** 2).mean(dim=[1,2,3]))
        loss = loss_batch.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tensorboard_writer.add_scalar("train/loss", loss, epoch)

        ### REVIEW 
        worst_batch = loss_batch.argmax().item()
        worst_pool = batch_idxs[worst_batch]
        remaining_batch = [i for i in range(BATCH_SIZE) if i != worst_batch]
        remaining_pool = [i for i in batch_idxs if i != worst_pool]

        pool[worst_pool] = seed.clone()
        pool[remaining_pool] = x[remaining_batch].detach()

        # Video for tensorboard

        if epoch % eval_frequency == 0:
            x_eval = seed.clone()

            eval_video = torch.empty(1, eval_iterations, 3, *x_eval.shape[2:])

            for idx_eval in range(eval_iterations):
                x_eval = model(x_eval)
                x_eval_out = to_rgb(x_eval[:, :4].detach().cpu())
                eval_video[0, idx_eval] = x_eval_out

            tensorboard_writer.add_video("Evaluation", eval_video, epoch, fps=60)

if __name__ == "__main__":
    print(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    train()