#### ABOUT :)) ####




#### IMPORTS ####

import torch
import torch.nn as nn
from model import GrowingGCA

import random


# MPL Imports
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation


#### DEFINE TRAINING PARAMETERS ####

GRID_SIZE = 32
CHANNELS = 16

MODEL = GrowingGCA()

EPOCHS = 5000
BATCH_SIZE = 32
UPDATES_RANGE = [64, 96] # The model is compared to the target image after this many updates, and loss is calculated

LR = 2e-3
LOSS_FN = torch.nn.MSELoss() # TODO change to L2
OPTIM = torch.optim.Adam(MODEL.parameters(), lr=LR)

VISUALS = []


#### VISUALISATION ####

def snapshot(currentGrid: torch.Tensor, save = VISUALS): # TODO
    """
        Add snapshot of the grid to the visuals list.
    """
    save.append(currentGrid.detach())

# End snapshot


def visualise(imgTensors): # TODO
    """
        Visualise a designated snapshot of the grid specified by idx

    """

    # We're only interested in the RGBalpha channels, and need numpy representation for plt
    img = imgTensors[:, :, :, 0:4].numpy()

    frames = [] # for storing the generated images
    fig = plt.figure()
    for i in range(6):
        frames.append([plt.imshow(img[i], cmap=cm.Greys_r,animated=True)])

    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,
                                    repeat_delay=1000)
    # ani.save('movie.mp4')
    plt.show()

    pass

# End visualise





#### INITIALISATION ####

def new_seed(batch_size = 1):
    """
        Creates a 4D tensor with dimensions batch_size x GRID_SIZE x GRID_SIZE x CHANNELS
        There is a single 1 in the alpha channel of center cell on each grid in the batch. 
    """
    # Tensor with Single seed cell 
    seed = torch.zeroes(batch_size, GRID_SIZE, GRID_SIZE, CHANNELS)
    seed[:, GRID_SIZE//2, GRID_SIZE//2, 4] = 1 # Alpha channel = 1
    return seed
# End new_seed


def load_image(imagePath: str):
    # Output image as 4D Tensor
    pass




#### FORWARD PASSES ####
def forward_pass(model: nn.Module, input, updates, record=False): # TODO
    if (record):
        recording = []
        for i in range(updates):
            input = model(input)
            snapshot(input, recording)
        # End for
        return (input, recording)
    # End if recording


    for i in range(updates):
        input = model(input)
    # End for

    return input

# End forwards



#### BACKWARD UPDATE PASS ####
def backward_pass(losses): # TODO
    loss = torch.mean(losses)
    OPTIM.zero_grad()
    loss.backward()
    OPTIM.step()
    
    pass




#### TRAINING ROUND ####

def train(model: nn.Module, target: torch.Tensor, record=False): # TODO
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

    # Record training data
    training_losses = []
    if (record):
        outputs = torch.zeros_like(batch)


    batch = new_seed(BATCH_SIZE)## TODO duplicate seed
    batch_losses = []

    # For epochs TODO
    # Forwards
    for input in batch:
        updates = random.randrange(UPDATES_RANGE[0], UPDATES_RANGE[1])
        output = forward_pass(model, input, updates)

        if (record):
            snapshot(outputs, input)

        # TODO Cut down to 4 channels
        batch_losses.append(LOSS_FN(output, target))
    # End for

    # Backwards
    backward_pass(batch_losses)

    # Assess accuracy TODO
    training_losses.append(LOSS_FN(forward_pass(new_seed[0])))

    # Save best model weights by deep copy of weights TODO
    # if (training_losses[-1] < bestLoss):
    #     BEST_MODEL = model


    if (record):
        return (model, training_losses, outputs)
    else:    
        return model, training_losses



#### SAVE WEIGHTS ####






if __name__ == "__main__":
    # random.seed(0)
    # # TODO torch device gpu
    # targetImg = load_image("#TODO image path")
    # MODEL, record_stages = train(MODEL, targetImg, record=True)

    # print(MODEL)

    


    visualise()




