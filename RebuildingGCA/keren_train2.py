#### ABOUT : ####


#### IMPORTS ####

import torch
from torchvision.io import read_image
import torchvision
from torch import Tensor
import torch.nn as nn
from model2 import GCA

import random


# MPL Imports
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation


def loss_fn(pred, target):
    """
    pred: (Batch, channels, height, width)
    target: (he)
    """

    # print(f"pred: {pred.sum()}, target: {target.sum()}")

    pixel_per_channel_loss = torch.square(pred - target)
    # print(f"pixel_per_channel_loss {pixel_per_channel_loss}")
    loss = torch.sum(pixel_per_channel_loss, dim=(0, 1, 2))
    # print(f"loss_fn loss")
    return loss


#### VISUALISATION ####

# def snapshot(currentGrid: torch.Tensor, save = []): # TODO
#     """
#         Add snapshot of the grid to the visuals list.
#     """
#     save.append(currentGrid.detach())
#     return save

# End snapshot


def visualise(imgTensor, anim=False):  # TODO
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

        # Plot RGB channels
        plt.subplot(1, 2, 1)
        plt.imshow(img[:, :, 0:3].detach().numpy())

        # Plot Alpha channel
        plt.subplot(1, 2, 2)
        plt.imshow(img[:, :, 3].detach().numpy())

    if not anim:
        update(0)

        # Display image
        plt.show()
        return

    ani = animation.FuncAnimation(fig, update, frames=len(imgTensor))
    # Display image
    plt.show()
    return ani

    # TODO enable an animation option
    # frames = [] # for storing the generated images
    # fig = plt.figure()
    # for i in range(4):
    #     frames.append([plt.imshow(img[i], cmap=cm.Greys_r,animated=True)])

    # ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,
    #                                 repeat_delay=1000)
    # # ani.save('movie.mp4')
    # plt.show()


# End visualise


#### INITIALISATION ####


def new_seed(batch_size=1):
    """
    Creates a 4D tensor with dimensions batch_size x GRID_SIZE x GRID_SIZE x CHANNELS
    There is a single 1 in the alpha channel of center cell on each grid in the batch.
    """
    # Tensor with Single seed cell
    seed = torch.zeros(batch_size, CHANNELS, GRID_SIZE, GRID_SIZE)
    seed[:, 3, GRID_SIZE // 2, GRID_SIZE // 2] = 1  # Alpha channel = 1

    return seed


# End new_seed


def load_image(imagePath: str):
    """
    Output image as 3D Tensor, with floating point values between 0 and 1
    Dimensions should be (colour channels, height, width)
    """
    img = read_image(imagePath)
    img = torchvision.transforms.functional.resize(img, (GRID_SIZE, GRID_SIZE))
    img = img.to(dtype=torch.float32) / 255

    return img


#### FORWARD PASSES ####
def forward_pass(model, input, updates, record=False):  # TODO
    """
    Run a forward pass
    """
    if record:

        saveState = Tensor(updates, CHANNELS, GRID_SIZE, GRID_SIZE)

        for i in range(updates):
            input = model(input)
            saveState[i] = input
        # End for

        return saveState

    for i in range(updates):
        input = model(input)
    # End for

    return input


# End forwards


#### BACKWARD UPDATE PASS ####
def update_pass(model, batch, target, optimiser):  # TODO
    """
    Back calculate gradient and update model paramaters
    """

    batch_losses = Tensor(BATCH_SIZE)
    for batch_idx in range(BATCH_SIZE):

        # Forward pass
        updates = random.randrange(UPDATES_RANGE[0], UPDATES_RANGE[1])
        output = forward_pass(model, batch[batch_idx].unsqueeze(0), updates)

        # Calculate loss
        """
        At the last step we apply pixel-wise L2 loss between RGBA channels in the grid and the target pattern. 
        This loss can be differentiably optimized 4 with respect to update rule parameters by 
        backpropagation-through-time
        """
        # batch_losses[batch_idx] = LOSS_FN(output[0, 0:4], target)
        # loss = LOSS_FN(output[0, 0:4], target)
        loss = LOSS_FN(output[0, 0:4], target)
        batch_losses[batch_idx] = loss

        # Back propogation
        optimiser.zero_grad()
        loss.backward()

        # print(f"GRADIENT: {MODEL.update_network[0].weight.grad}")

        optimiser.step()

    # End for
    # loss = torch.mean(batch_losses)
    # loss.backward()
    # optimiser.step()
    print([param for param in MODEL.parameters()])

    print(f"batch loss = {batch_losses}")


#### TRAINING ROUND ####


def train(model: nn.Module, target: torch.Tensor):  # TODO
    """
    TRAINING PROCESS:
        - Define training data storage variables
        - For each epoch:
            - Initialise batch
            - Forward pass (runs the model on the batch)
            - Backward pass (calculates loss and updates params)
            - SANITY CHECK: check current loss and record loss
            - Save model if this is the best model TODO
        - Return the trained model
    """

    optimiser = torch.optim.Adam(params=MODEL.parameters(), maximize=False, lr=LR)
    print(optimiser.param_groups)
    try:
        # Record training data
        training_losses = []

        for epoch in range(EPOCHS):

            model.train()

            batch = new_seed(BATCH_SIZE)  ## TODO duplicate seed

            # Optimisation step
            update_pass(model, batch, target, optimiser)

            # Assess accuracy TODO
            MODEL.eval()
            test_seed = new_seed(1)
            test_run = forward_pass(MODEL, test_seed, 64)
            training_losses.append(LOSS_FN(test_run[0, 0:4], target).detach().numpy())
            print(f"Epoch {epoch} complete, loss = {training_losses[-1]}")

            max_val = torch.max(test_run)
            min_val = torch.min(test_run)

            print(f"maximum on output grid = {max_val}")
            print(f"minimum on output grid = {min_val}")

            # Save best model weights by deep copy of weights TODO
            # if (training_losses[-1] < bestLoss):
            #     BEST_MODEL = model
    except KeyboardInterrupt:
        pass

    return model, training_losses


#### SAVE WEIGHTS ####


if __name__ == "__main__":

    #### DEFINE TRAINING PARAMETERS ####

    GRID_SIZE = 32
    CHANNELS = 16

    MODEL = GCA()

    EPOCHS = 5
    BATCH_SIZE = 32
    UPDATES_RANGE = [
        64,
        96,
    ]  # [1, 10]#[64, 96]# [1, 10]# The model is compared to the target image after this many updates, and loss is calculated

    LR = 1e-5
    LOSS_FN = loss_fn  # torch.nn.MSELoss(size_average=None , reduce=None , reduction="mean" ) # TODO change to L2
    print([param for param in MODEL.parameters()])

    # random.seed(0)
    # # TODO torch device gpu

    # LOAD TARGET IMAGE

    targetImg = load_image("cat.png")
    # visualise(targetImg)
    # print(seed.shape)
    # visualise(seed[0])

    MODEL, loss = train(MODEL, targetImg)
    print(loss)
    MODEL.eval()

    plt.plot(range(len(loss)), loss)

    seed = new_seed(1)
    img = forward_pass(MODEL, seed, 64)
    video = forward_pass(MODEL, seed, 64, record=True)

    visualise(img)
    anim = visualise(video, anim=True)

    # MODEL, record_stages = train(MODEL, targetImg, record=True)

    # print(MODEL)
    # visualise(targetImg)
