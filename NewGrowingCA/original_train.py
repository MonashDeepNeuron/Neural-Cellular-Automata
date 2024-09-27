import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from original_model import GrowingCA
import utils
from utils import utils


def train():
    # I'll add a parser later
    BATCH_SIZE = 4
    PADDING = 16
    p = PADDING

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logdir = "logs"

    tensorboard_writer = SummaryWriter(logdir)

    # Target
    img_path = "squirrel.png"
    img_size = 32
    target_img = utils.load_target(img_path, im_size=img_size)
    target_img_ = nn.functional.pad(target_img, (p, p, p, p), "constant", 0)
    target_img = target_img_.to(device)
    target_img = target_img.repeat(BATCH_SIZE, 1, 1, 1)

    tensorboard_writer.add_image("Target Image", utils.to_rgb(target_img)[0])

    # Model
    n_channels = 16
    hidden_channels = 128
    device = device
    model = GrowingCA(
        n_channels=n_channels, hidden_channels=hidden_channels, device=device
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    # Pool to sample from
    pool_size = 1024
    seed = utils.starting_seed(img_size, n_channels).to(device)
    seed = nn.functional.pad(seed, (p, p, p, p), "constant", 0)
    pool = seed.clone().repeat(pool_size, 1, 1, 1)

    epochs = 500  # 5000
    eval_frequency = 100  # 500
    eval_iterations = 500

    for epoch in tqdm(range(epochs)):
        batch_idxs = np.random.choice(pool_size, BATCH_SIZE, replace=False).tolist()

        x = pool[batch_idxs]
        for i in range(
            np.random.randint(64, 96)
        ):  # Number of steps to go from seed --> cat
            x = model(x)

        loss_batch = ((target_img - x[:, :4, ...]) ** 2).mean(dim=[1, 2, 3])
        loss = loss_batch.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tensorboard_writer.add_scalar("train/loss", loss, epoch)

        ### REVIEW
        argmax_batch = loss_batch.argmax().item()
        argmax_pool = batch_idxs[argmax_batch]
        remaining_batch = [i for i in range(BATCH_SIZE) if i != argmax_batch]
        remaining_pool = [i for i in batch_idxs if i != argmax_pool]

        pool[argmax_pool] = seed.clone()
        pool[remaining_pool] = x[remaining_batch].detach()

        # Video for s

        if epoch % eval_frequency == 0:
            x_eval = seed.clone()  # (1, n_channels, size, size)

            eval_video = torch.empty(1, eval_iterations, 3, *x_eval.shape[2:])

            for it_eval in range(eval_iterations):
                x_eval = model(x_eval)
                x_eval_out = utils.to_rgb(x_eval[:, :4].detach().cpu())
                eval_video[0, it_eval] = x_eval_out

            tensorboard_writer.add_video("eval", eval_video, epoch, fps=60)

    return model, seed


if __name__ == "__main__":
    print(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    model, seed = train()
    x = seed.clone()
    utils.make_video(x, model)
