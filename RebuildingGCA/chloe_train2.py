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


def visualise(imgTensor, filenameBase = "test", anim = False, save = True, show = True):
    """
    Visualise a designated snapshot of the grid specified by idx
    Input in form (channels, height, width)
    """

    if (len(imgTensor.shape) < 4):
        imgTensor.unsqueeze(0)

    fig, ax = plt.subplots(1, 2)

    def update(imgIdx):
        # We're only interested in the RGBalpha channels, and need numpy representation for plt
        img =  imgTensor[imgIdx].clip(0, 1).squeeze().permute(1, 2, 0)

        if (anim):
            plt.suptitle("Update " + str(imgIdx))


        # Plot RGB channels
        plt.subplot(1,2, 1)
        plt.imshow(img[:, :, 0:3].detach().numpy())
        plt.title("RGB")

        # Plot Alpha channel
        plt.subplot(1, 2, 2)
        plt.imshow(img[:, :, 3].detach().numpy())
        plt.title("Alpha (alive/dead)")

    # End animation update

    if not anim:
        update(0)
        if (save):
            plt.savefig(filenameBase+'.png', bbox_inches='tight')

        # Display image
        if (show):
            plt.show()
            plt.close('all')
        return
    
    ani = animation.FuncAnimation(fig, update, frames = len(imgTensor), repeat = False)
    # Display image
    if (save):
        # To save the animation using Pillow as a gif
        writer = animation.PillowWriter(fps=15,
                                        metadata=dict(artist='Me'),
                                        bitrate=1800)
        ani.save(filenameBase+'.gif', writer=writer)

    if (show):
        plt.show()
        plt.close('all')

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
    img = read_image(imagePath, mode = ImageReadMode.RGB_ALPHA)
    img = torchvision.transforms.functional.resize(img, (GRID_SIZE, GRID_SIZE))
    img = img.to(dtype=torch.float32) / 255

    return img


def forward_pass(model: nn.Module, state, updates, record=False):  # TODO
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


def train(model: nn.Module, target: torch.Tensor, optimiser, record=False):  # TODO
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

    ## Obtain device of model, and send all related data to device
    device = next(model.parameters()).device
    target = target.to(device)

    try:
        training_losses = []
        for epoch in range(EPOCHS):
            model.train()
            if record:
                outputs = torch.zeros_like(batch)

            batch = new_seed(BATCH_SIZE)  ## TODO duplicate seed
            batch = batch.to(device)

            ## Optimisation step
            update_pass(model, batch, target, optimiser)

            ## TODO: Assess accuracy. Can we remove this for faster training?
            test_seed = new_seed(1)
            MODEL.eval()
            test_run = forward_pass(MODEL, test_seed, 64)
            training_losses.append(
                LOSS_FN(test_run[0, 0:4], target).cpu().detach().numpy()
            )
            print(f"Epoch {epoch} complete, loss = {training_losses[-1]}")

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
    
    TRAINING = True # Is our purpose to train or are we just looking rn?

    GRID_SIZE = 32
    CHANNELS = 16

    MODEL = GCA()
    MODEL = initialiseGPU(MODEL)
    EPOCHS = 100 # 100 epochs for best results
    ## 30 epochs, once loss dips under 0.8 switch to learning rate 0.0001

    BATCH_SIZE = 32
    UPDATES_RANGE = [64, 96]

    LR = 1e-3
    optimizer = torch.optim.Adam(MODEL.parameters(), lr=LR)
    LOSS_FN = torch.nn.MSELoss(reduction="mean")

    MODEL_PATH = "model_weights_logo.pth"

    targetImg = load_image("logo.png")

    ## Load model weights if available
    try:
        MODEL.load_state_dict(torch.load(MODEL_PATH, weights_only=True, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
        print("Loaded model weights successfully!")
    except FileNotFoundError:
        print("No previous model weights found, training from scratch.")
        if (not TRAINING):
            exit()


    if (TRAINING):
        MODEL, loss = train(MODEL, targetImg, optimizer)
        print(loss)
        
        ## Plot loss
        plt.plot(range(len(loss)), loss)
        
        ## Save the model's weights after training
        torch.save(MODEL.state_dict(), MODEL_PATH)


    ## Switch state to evaluation to disable dropout e.g.
    MODEL.eval()       

    
    ## Plot final state of evaluation OR evaluation animation
    img = new_seed(1)
    video = forward_pass(MODEL, img, 96, record=True)
    anim = visualise(video, anim=True)
