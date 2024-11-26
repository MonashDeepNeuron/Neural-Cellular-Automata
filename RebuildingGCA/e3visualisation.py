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
import retaining_image as ri

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

def grid_interaction(model: GCA, img: Tensor):
    """
    Creates a matplotlib interactive grid to interact with the model step by step.
    """
    # Initialize the figure and axes
    fig, ax = plt.subplots()

    # Ensure the input image is on the appropriate device
    device = next(model.parameters()).device
    img = img.to(device)

    # Create an image plot with the initial image state
    state = img.clone().detach()
    imshow = ax.imshow(state[0, :3].permute(1, 2, 0).cpu().numpy().clip(0, 1))
    plt.title("Press 'n' for next step, 'q' to quit")

    # Define the event handler for keypress events
    def on_key(event):
        nonlocal state
        if event.key == "n":  # Move to the next step
            state = model(state)  # Apply the model transformation
            imshow.set_data(state[0, :3].permute(1, 2, 0).cpu().numpy().clip(0, 1))
            fig.canvas.draw()
        elif event.key == "q":  # Quit interaction
            plt.close(fig)

    # Connect the keypress event to the handler
    fig.canvas.mpl_connect("key_press_event", on_key)

    # Show the interactive plot
    plt.show()


if __name__ == "__main__":
    
    TRAINING = False  # Is our purpose to train or are we just looking rn?
    LOAD_WEIGHTS = True # only load weights if we want to start training from previous

    ## For learning rate adjustmnet
    ADJUSTMENT_WINDOW = 10

    GRID_SIZE = 32
    CHANNELS = 16

    POOL_SIZE= 1024
    BATCH_SIZE= 32
    UPDATES_RANGE = (64, 192)


    MODEL = GCA()
    MODEL = ri.initialiseGPU(MODEL)
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

    ## Switch state to evaluation to disable dropout e.g.
    MODEL.eval()

    ## Plot final state of evaluation OR evaluation animation
    img = new_seed(1)
    video = grid_interaction(MODEL, img)
