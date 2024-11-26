#### IMPORTS ####

import torch
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
import torchvision
from torch import Tensor
import torch.nn as nn
from model2 import GCA
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
from learning_rate_adjuster import lradj


def visualise(imgTensor, filenameBase="test", anim=False, save=True, show=True):
    """
    Visualise a designated snapshot of the grid specified by idx
    Input in form (channels, height, width)
    """

    if len(imgTensor.shape) < 4:
        imgTensor.unsqueeze(0)

    fig, ax = plt.subplots(1, 2)

    def update(imgIdx):
        # We're only interested in the RGBalpha channels, and need numpy representation for plt
        img = imgTensor[imgIdx].clip(0, 1).squeeze().permute(1, 2, 0)

        if anim:
            plt.suptitle("Update " + str(imgIdx))

        # Plot RGB channels
        plt.subplot(1, 2, 1)
        plt.imshow(img[:, :, 0:3].detach().numpy())
        plt.title("RGB")

        # Plot Alpha channel
        plt.subplot(1, 2, 2)
        plt.imshow(img[:, :, 3].detach().numpy())
        plt.title("Alpha (alive/dead)")

    # End animation update

    if not anim:
        update(0)
        if save:
            plt.savefig(filenameBase + ".png", bbox_inches="tight")

        # Display image
        if show:
            plt.show()
            plt.close("all")
        return

    ani = animation.FuncAnimation(fig, update, frames=len(imgTensor), repeat=False)
    # Display image
    if save:
        # To save the animation using Pillow as a gif
        writer = animation.PillowWriter(
            fps=15, metadata=dict(artist="Me"), bitrate=1800
        )
        ani.save(filenameBase + ".gif", writer=writer)

    if show:
        plt.show()
        plt.close("all")

    return ani


def new_seed(batch_size=1):
    """
    Creates a 4D tensor with dimensions batch_size x GRID_SIZE x GRID_SIZE x CHANNELS
    There is a single 1 in the alpha channel of center cell on each grid in the batch.
    """
    seed = torch.zeros(batch_size, CHANNELS, GRID_SIZE, GRID_SIZE)
    seed[:, 3, GRID_SIZE // 2, GRID_SIZE // 2] = 1  # Alpha channel = 1

    return seed


def load_image(imagePath: str):
    """
    Output image as 3D Tensor, with floating point values between 0 and 1
    Dimensions should be (colour channels, height, width)
    """
    GRID_SIZE = 32
    img = read_image(imagePath, mode=ImageReadMode.RGB_ALPHA)
    ## Pad image with 3 pixels with of black border before resizing

    ## Reduce existing image to 28*28
    img = torchvision.transforms.functional.resize(
        img, ((GRID_SIZE - 4), (GRID_SIZE - 4))
    )

    ## Pad it to original grid size
    padding_transform = torchvision.transforms.Pad(2, 2)
    img = padding_transform(img)

    img = img.to(dtype=torch.float32) / 255

    return img


def forward_pass(model: nn.Module, state, updates, record=False, CHANNELS=16, GRID_SIZE = 32):  # TODO
    """
    Run a forward pass consisting of `updates` number of updates
    If `record` is true, then records the state in a tensor to animate and saves the video
    Returns the final state
    """
    if record:
        frames_array = Tensor(updates, CHANNELS, GRID_SIZE, GRID_SIZE)
        for i in range(updates):
            state = model(state)
            frames_array[i] = state
        return frames_array

    else:
        for i in range(updates):
            state = model(state)

    return state


def update_pass(model, batch, target, optimiser):
    """
    Back calculate gradient and update model paramaters
    """
    device = next(model.parameters()).device
    batch_losses = torch.zeros(BATCH_SIZE, device=device)
    for batch_idx in range(BATCH_SIZE):
        optimiser.zero_grad()
        updates = random.randrange(UPDATES_RANGE[0], UPDATES_RANGE[1])
        output = forward_pass(model, batch[batch_idx].unsqueeze(0), updates)

        ## apply pixel-wise MSE loss between RGBA channels in the grid and the target pattern
        loss = LOSS_FN(output[0, 0:4], target)
        ## .item() removes computational graph for memory efficiency
        batch_losses[batch_idx] = loss.item()
        loss.backward()
        optimiser.step()

    print(f"batch loss = {batch_losses.cpu().numpy()}")  ## print on cpu


def pool_train(model: nn.Module, target: torch.Tensor, optimiser, record=False):
    device = next(model.parameters()).device
    target = target.to(device)

    # sample pool starts off as the default seed for all 1024 random samples that we have 
    sample_pool = [new_seed(1).to(device) for _ in range(POOL_SIZE)]

    try:
        training_losses = []
        updated_learning_rates = []
        loss_window = [None for i in range(ADJUSTMENT_WINDOW)]

        for epoch_idx in range(EPOCHS):
            loss_window_idx = epoch_idx % ADJUSTMENT_WINDOW
            if loss_window_idx == 0 and epoch_idx != 0: # don't start lr adjuster at the start of training
                updated_lr = lradj.get_adjusted_learning_rate(loss_window) 
                loss_window = [None for i in range(ADJUSTMENT_WINDOW)]
                ## SET OPTIMISER
                for param_group in optimiser.param_groups:
                    param_group["lr"] = updated_lr
                updated_learning_rates.append(updated_lr)

            model.train()
            if record:
                outputs = torch.zeros_like(batch)

            # get a random sample of indices from the poolsize to create a batch
            batch_indices = random.sample(range(POOL_SIZE), BATCH_SIZE)
            batch = torch.cat([sample_pool[idx] for idx in batch_indices], dim=0)

            # Replace one sample with the original single-pixel seed state; i actually changed this to ten otherwise it will take way to long to render regular outcome of the model
            #batch[0] = new_seed(1).to(device)
            for i in range(5):
                batch[i] = new_seed(1).to(device)

            ## Optimisation step
            update_pass(model, batch, target, optimiser)

            # Replace samples in the pool with the output states, eg we are updating the pool with the persisting states that we need to train on in future
            for i, idx in enumerate(batch_indices):
                sample_pool[idx] = batch[i].unsqueeze(0)

            test_seed = new_seed(1) # test on the default seed state (could be worth also testing on a persisting state ? )
            MODEL.eval()
            test_run = forward_pass(MODEL, test_seed, 64)
            training_losses.append(
                LOSS_FN(test_run[0, 0:4], target).cpu().detach().numpy()
            )
            print(f"Epoch {epoch_idx} complete, loss = {training_losses[-1]}")

            loss_window[loss_window_idx] = training_losses[-1].item()


    except KeyboardInterrupt:
        pass

    if record:
        return (model, training_losses, outputs)
    else:
        return model, training_losses



def initialiseGPU(model):
    ## Check if GPU available
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} is available.")

    ## Configure device as GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model


if __name__ == "__main__":

    TRAINING = True  # Is our purpose to train or are we just looking rn?
    LOAD_WEIGHTS = False # only load weights if we want to start training from previous

    ## For learning rate adjustmnet
    ADJUSTMENT_WINDOW = 10

    GRID_SIZE = 32
    CHANNELS = 16

    POOL_SIZE=1024
    BATCH_SIZE=32
    UPDATES_RANGE=(64, 192)

    MODEL = GCA()
    MODEL = initialiseGPU(MODEL)
    EPOCHS = 100  # 100 epochs for best results
    ## 30 epochs, once loss dips under 0.8 switch to learning rate 0.0001

    BATCH_SIZE = 32
    UPDATES_RANGE = [64,192] # for longer life

    LR = 1e-4

    optimizer = torch.optim.Adam(MODEL.parameters(), lr=LR)
    LOSS_FN = torch.nn.MSELoss(reduction="mean")

    MODEL_PATH = "model_weights_logo_updated_lr.pth"

    targetImg = load_image("RebuildingGCA/cat.png")

    ## Load model weights if available
    if LOAD_WEIGHTS:
        try:
            MODEL.load_state_dict(
                torch.load(
                    MODEL_PATH,
                    weights_only=True,
                    map_location=torch.device(
                        "cuda" if torch.cuda.is_available() else "cpu"
                    ),
                )
            )
            print("Loaded model weights successfully!")
        except FileNotFoundError:
            print("No previous model weights found, training from scratch.")
            if not TRAINING:
                exit()

    if TRAINING:
        MODEL, losses = pool_train(MODEL, targetImg, optimizer)

        ## Plot loss
        plt.plot(range(len(losses)), losses)

        ## Save the model's weights after training
        torch.save(MODEL.state_dict(), MODEL_PATH)

    ## Switch state to evaluation to disable dropout e.g.
    MODEL.eval()

    ## Plot final state of evaluation OR evaluation animation
    img = new_seed(1)
    video = forward_pass(MODEL, img, 2000, record=True)
    anim = visualise(video, anim=True)