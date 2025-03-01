"""
PURPOSE: To train the persisting NCA model based off the persisting 
    stage model in https://distill.pub/2020/growing-ca/.
FEATURES:
    - Loads target image from specified file path as well as model if existing model is present
    - Training the model on specified number of epochs and with 
        or without CUDA
    - Uses an adaptive learning rate throughout training (see learning_rate_adjuster.py)
    - Incorperates pool training (see https://distill.pub/2020/growing-ca/)
    - Output of loss as plot
    - Saves final model weights to a second specified model path.
    - Visualisation of the model and generation of a GIF file 
        demonstrating model output
    - Visualisation of the progressive state of the model pool
"""


#### IMPORTS ####

import torch
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
import torchvision
from torch import Tensor
import torch.nn as nn
from persistingmodel import GCA
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
from learning_rate_adjuster import lradj
import numpy as np

TRAINING = False  # Is our purpose to train or are we just looking rn?
LOAD_WEIGHTS = True # only load weights if we want to start training from previous

## For learning rate adjustmnet
ADJUSTMENT_WINDOW = 7

GRID_SIZE = 40
CHANNELS = 16

POOL_SIZE = 1024

EPOCHS = 5000  # 5000 recommended epochs 
## 30 epochs, once loss dips under 0.8 switch to learning rate 0.0001

MODEL_PATH = "abc_4.pth"
SAVE_PATH = "abc_4.pth"

LR = 1e-4
BATCH_SIZE = 12
LR_FACTOR = 1/BATCH_SIZE


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
        plt.clf()
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


def new_seed(batch=1):
    """
    Creates a 4D tensor with dimensions batch_size x GRID_SIZE x GRID_SIZE x CHANNELS
    There is a single 1 in the alpha channel of center cell on each grid in the batch.
    """
    if not isinstance(batch, Tensor):
        seed = torch.zeros(batch, CHANNELS, GRID_SIZE, GRID_SIZE)
        seed[:, 3, GRID_SIZE // 2, GRID_SIZE // 2] = 1  # Alpha channel = 1
    
        return seed
    
    batch[:, :, :, :] = 0
    batch[:, 3, GRID_SIZE // 2, GRID_SIZE // 2] = 1  # Alpha channel = 1
    return batch


def load_image(imagePath: str):
    """
    Output image as 3D Tensor, with floating point values between 0 and 1
    Dimensions should be (colour channels, height, width)
    """
    img = read_image(imagePath, mode=ImageReadMode.RGB_ALPHA)
    ## Pad image with 3 pixels with of black border before resizing

    image_size = 28

    ## Reduce existing image to 28*28
    img = torchvision.transforms.functional.resize(
        img, (( image_size ), ( image_size ))
    )
    ## Pad it to original grid size
    padding_transform = torchvision.transforms.Pad((GRID_SIZE- image_size)//2, (GRID_SIZE- image_size)//2)
    img = padding_transform(img)
    img = img.to(dtype=torch.float32) / 255

    return img


def forward_pass(model: nn.Module, state, updates, record=False): 
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


def update_pass(model, batch, target, optimiser, updates_range):
    """
    Back calculate gradient and update model paramaters
    """
    device = next(model.parameters()).device
    batch_losses = torch.zeros(BATCH_SIZE, device=device)
    optimiser.zero_grad()
    updates = random.randint(updates_range[0],updates_range[1])
    batch = forward_pass(model, batch, updates)
    ## apply pixel-wise MSE loss between RGBA channels in the grid and the target pattern
    batch_losses = LOSS_FN(batch[0, 0:4], target)
    ## .item() removes computational graph for memory efficiency
    batch_losses.backward()
    optimiser.step()
    optimiser.zero_grad()

    return batch.detach()

def standard_train(model: nn.Module, target: torch.Tensor, optimiser, record=False):
    device = next(model.parameters()).device
    print(f"Loading to {device}")

    target = target.to(device)
    
    print(f"Loaded to {device}")
    ## Optimisation step

    model.eval()
    snapshots = update_pass(model, new_seed(2), target, optimiser, [64, 96]).detach()

    batch = new_seed(BATCH_SIZE)

    best_loss = LOSS_FN(snapshots[0, 0:4], target).cpu().detach().numpy()
    best_model = model.state_dict()

    try:
        training_losses = []
        updated_learning_rates = []
        loss_window = [None for i in range(ADJUSTMENT_WINDOW)]

        for epoch_idx in range(EPOCHS):
            loss_window_idx = epoch_idx % ADJUSTMENT_WINDOW
            if loss_window_idx == 0 and epoch_idx != 0: # don't start lr adjuster at the start of training
                
                updated_lr = lradj.get_adjusted_learning_rate(loss_window)*LR_FACTOR
                loss_window = [None for i in range(ADJUSTMENT_WINDOW)]
                ## SET OPTIMISER
                for param_group in optimiser.param_groups:
                    param_group["lr"] = updated_lr
                    print(f"New lr is {updated_lr}")
                updated_learning_rates.append(updated_lr)

            model.train()

            # create a batch
            batch = new_seed(batch).to(device)

            ## Optimisation step
            batch = update_pass(model, batch, target, optimiser, UPDATES_RANGE)

            test_seed = new_seed(1) # test on the default seed state (could be worth also testing on a persisting state ? )
            MODEL.eval()
            test_run = forward_pass(MODEL, test_seed, 96)
            training_losses.append(
                LOSS_FN(test_run[0, 0:4], target).cpu().detach().numpy()
            )
            loss_window[loss_window_idx] = training_losses[-1].item()

            # Save best model weights every 4 epochs
            save_interval = 16
            if (epoch_idx % save_interval == 0):

                print(f"Epoch {epoch_idx} complete, loss = {training_losses[-1]}")

                if training_losses[-1] < best_loss:
                    best_loss = training_losses[-1]
                    best_model = model.state_dict()

                if (record):
                    selected = random.sample(range(BATCH_SIZE), 2)
                    snapshots = torch.cat((snapshots, batch[selected]), dim=0)


    except KeyboardInterrupt:
        pass

    model.load_state_dict(best_model)

    if record:
        return (model, training_losses, snapshots)
    else:
        return model, training_losses


def pool_train(model: nn.Module, target: torch.Tensor, optimiser, seedrate, record=False):

    device = next(model.parameters()).device
    print(f"Loading to {device}")

    target = target.to(device)

    # sample pool starts off as the default seed for all 1024 random samples that we have 
    sample_pool = new_seed(POOL_SIZE).to(device)
    
    print(f"Loaded to {device}")
    ## Optimisation step
    model.eval()

    snapshots = sample_pool[[1,2], :, :, :]
    
    best_loss = LOSS_FN(snapshots[0, 0:4], target).cpu().detach().numpy()
    best_model = model.state_dict()

    try:
        training_losses = []
        updated_learning_rates = []
        loss_window = [None for i in range(ADJUSTMENT_WINDOW)]

        for epoch_idx in range(EPOCHS):
            loss_window_idx = epoch_idx % ADJUSTMENT_WINDOW
            if loss_window_idx == 0 and epoch_idx != 0: # don't start lr adjuster at the start of training
                updated_lr = lradj.get_adjusted_learning_rate(loss_window)*LR_FACTOR
                loss_window = [None for i in range(ADJUSTMENT_WINDOW)]
                ## SET OPTIMISER
                for param_group in optimiser.param_groups:
                    param_group["lr"] = updated_lr
                    print(f"New lr is {updated_lr}")
                updated_learning_rates.append(updated_lr)

            model.train()
            # get a random sample of indices from the poolsize to create a batch
            batch_indices = random.sample(range(POOL_SIZE), BATCH_SIZE)
            seeds = random.sample(range(BATCH_SIZE), seedrate)
            batch = sample_pool[batch_indices, :, :, :]
            batch[seeds] = new_seed(seedrate).to(device)

            ## Optimisation step
            output = update_pass(model, batch, target, optimiser, UPDATES_RANGE)

            # Replace samples in the pool with the output states, eg we are updating the pool with the persisting states that we need to train on in future
            sample_pool[batch_indices] = output.detach()

            test_seed = new_seed(1) # test on the default seed state (could be worth also testing on a persisting state ? )
            MODEL.eval()
            test_run = forward_pass(MODEL, test_seed, 96)
            training_losses.append(
                LOSS_FN(test_run[0, 0:4], target).cpu().detach().numpy()
            )

            loss_window[loss_window_idx] = training_losses[-1].item()
            
            # Save best model weights every 4 epochs
            save_interval = 20
            if (epoch_idx % save_interval == 0):
                print(f"Epoch {epoch_idx} complete, loss = {training_losses[-1]}")
                if training_losses[-1] < best_loss:
                    best_loss = training_losses[-1]
                    best_model = model.state_dict()
                if (record):
                    selected = random.sample(range(BATCH_SIZE), 2)
                    snapshots = torch.cat((snapshots, batch[selected]), dim=0)

    except KeyboardInterrupt:
        pass
        
    model.load_state_dict(best_model)

    if record:
        return (model, training_losses, snapshots)
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

    print("Initialising model...")

    MODEL = GCA()
    MODEL = initialiseGPU(MODEL)

    targetImg = load_image("./cat.png")

    ## Load model weights if available
    if LOAD_WEIGHTS:
        print("Loading model weights...")
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
            LOAD_WEIGHTS = False
            print("No previous model weights found, training from scratch.")
            if not TRAINING:
                exit()

    if TRAINING:
        print("Training...")
        losses1, recording1 = None, None

        if (not LOAD_WEIGHTS):
            LR = 1e-3
            BATCH_SIZE = 2

            optimizer = torch.optim.Adam(MODEL.parameters(), lr=LR, weight_decay= 1e-8)
            LOSS_FN = torch.nn.MSELoss(reduction="mean")

            UPDATES_RANGE=(64, 96)
            MODEL, losses1, recording1 = standard_train(MODEL, targetImg, optimizer, record=True)
    
            ## Save the model's weights after training
            torch.save(MODEL.state_dict(), SAVE_PATH)

        optimizer = torch.optim.Adam(MODEL.parameters(), lr=LR)
        LOSS_FN = torch.nn.MSELoss(reduction="mean")

        UPDATES_RANGE=(10, 50)
        MODEL, losses2, recording2 = pool_train(MODEL, targetImg, optimizer, record=True, seedrate = 1)

        ## Save the model's weights after training
        torch.save(MODEL.state_dict(), SAVE_PATH)
    
        # Prepare weights list
        weights_list = []

        # Log weight and bias shapes
        for i, (name, param) in enumerate(MODEL.named_parameters()):
            shape = tuple(param.shape)
            print(f"Layer {i}: {name} - Shape: {shape} - Size {param.numel()}")

            # Append flattened weight/bias data
            weights_list.append(param.detach().cpu().numpy().flatten())

        # Concatenate all weights and biases into a single NumPy array
        weights = np.concatenate(weights_list, dtype=np.float32)

        # Save to a raw binary file (compact format)
        weights.tofile("./weights.bin")


        ## Plot loss
        losses = losses2 #np.concatenate((losses1, losses2), axis=0)

        plt.plot(range(len(losses)), losses)

        plt.savefig("loss.png")


        ## Visialise the training snapshots
        if (LOAD_WEIGHTS):
            anim = visualise(recording2, anim=True, filenameBase="pool", show=False)
        else :
            anim = visualise(torch.cat((recording1, recording2), dim=0), anim=True, filenameBase="pool", show=False)

    ## Switch state to evaluation to disable dropout e.g.
    MODEL.eval()

    GRID_SIZE = 60

    ## Plot final state of evaluation OR evaluation animation
    img = new_seed(1)
    video = forward_pass(MODEL, img, 600, record=True)
    anim = visualise(video, filenameBase = "train", anim=True)
    anim = visualise(video[-1].unsqueeze(0), filenameBase = "train", anim=False)
