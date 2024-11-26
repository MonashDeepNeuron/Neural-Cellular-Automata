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
from retaining_image import initialiseGPU

# Change these constants to desired output behaviour
GRID_SIZE = 32
CHANNELS = 16
TICKRATE = 30  # Milliseconds per tick fyi
TRAINING = False  # Is our purpose to train or are we just looking rn?
LOAD_WEIGHTS = True  # only load weights if we want to start training from previous

MODEL = GCA()
MODEL = initialiseGPU(MODEL)
MODEL_PATH = "model_weights_logo_updated_lr.pth"

class Grid:
    def __init__(self, model: GCA, img: torch.Tensor, channels: int, grid_size: int):
        self.model = model
        self.img = img
        self.channels = channels
        self.grid_size = grid_size
        self.count = [0]  # Initialize count as a list with a single element
        self.device = next(model.parameters()).device
        self.state = self.new_seed(1).to(self.device)
        self.fig, self.ax = plt.subplots()
        self.imshow = self.ax.imshow(self.state[0, :3].permute(1, 2, 0).cpu().detach().numpy().clip(0, 1))
        plt.title("Press 'q' to quit")
        self.mouse_pressed = False
        self.prev_x = None
        self.prev_y = None

    def new_seed(self, batch_size):
        """
        Creates a 4D tensor with dimensions batch_size x GRID_SIZE x GRID_SIZE x CHANNELS
        There is a single 1 in the alpha channel of center cell on each grid in the batch.
        """
        seed = torch.zeros(batch_size, CHANNELS, GRID_SIZE, GRID_SIZE)
        seed[:, 3, GRID_SIZE // 2, GRID_SIZE // 2] = 1  # Alpha channel = 1
        return seed

    def tick(self):
        '''
        Tick method updates a single foward step for the ML model and updates the plot
        '''
        self.count[0] += 1
        plt.suptitle(f"Step: {self.count[0]}")
        self.state = self.model(self.state)
        self.imshow.set_data(self.state[0, :3].permute(1, 2, 0).cpu().detach().numpy().clip(0, 1))
        self.fig.canvas.draw()

    def remove_pixels(self, x, y, radius):
        '''
        Remove pixels from the grid (used in event handlers)
        '''
        _, _, height, width = self.state.shape
        for i in range(height):
            for j in range(width):
                if (i - y) ** 2 + (j - x) ** 2 <= radius ** 2:
                    self.state[0, :, i, j] = 0  # Set the pixel values to 0 (black)

    def interpolate_and_remove(self, x0, y0, x1, y1, radius):
        dist = max(abs(x1 - x0), abs(y1 - y0))
        for i in range(dist + 1):
            #Idk how this works without safe division but it does
            x = int(x0 + i * (x1 - x0) / dist)
            y = int(y0 + i * (y1 - y0) / dist)
            self.remove_pixels(x, y, radius)

    def on_click(self, event):
        #this should be a double click explosion? could be fun 
        if event.inaxes is not None:
            self.mouse_pressed = True
            self.prev_x, self.prev_y = int(event.xdata), int(event.ydata)
            radius = 5  # Define the radius around the cursor to remove pixels
            self.remove_pixels(self.prev_x, self.prev_y, radius)
            self.imshow.set_data(self.state[0, :3].permute(1, 2, 0).cpu().detach().numpy().clip(0, 1))
            self.fig.canvas.draw()

    def on_release(self, event):
        self.mouse_pressed = False
        self.prev_x = None
        self.prev_y = None

    def on_motion(self, event):
        if self.mouse_pressed and event.inaxes is not None:
            x, y = int(event.xdata), int(event.ydata)
            radius = 5  # Define the radius around the cursor to remove pixels
            if self.prev_x is not None and self.prev_y is not None:
                self.interpolate_and_remove(self.prev_x, self.prev_y, x, y, radius)
            self.prev_x, self.prev_y = x, y
            self.imshow.set_data(self.state[0, :3].permute(1, 2, 0).cpu().detach().numpy().clip(0, 1))
            self.fig.canvas.draw()

    def update(self, frame):
        self.tick()

    def grid_interaction(self):
        # Connect all event handlers into canvas. I Suppose this is sort of like 
        # the state transducer, and could be made in a more functionally reactive way if we wanted.
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)

        # Create the animation
        ani = animation.FuncAnimation(self.fig, self.update, interval=TICKRATE)  
        plt.show()

def load_image(imagePath: str):
    """
    Output image as 3D Tensor, with floating point values between 0 and 1
    Dimensions should be (colour channels, height, width)
    """
    img = read_image(imagePath, mode=ImageReadMode.RGB_ALPHA)
    img = torchvision.transforms.functional.resize(img, (GRID_SIZE - 4, GRID_SIZE - 4))
    padding_transform = torchvision.transforms.Pad(2, 2)
    img = padding_transform(img)
    img = img.to(dtype=torch.float32) / 255
    return img

if __name__ == "__main__":
    img = load_image("RebuildingGCA/cat.png")

    if LOAD_WEIGHTS:
        try:
            MODEL.load_state_dict(
                torch.load(
                    MODEL_PATH,
                    map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                )
            )
            print("Loaded model weights successfully!")
        except FileNotFoundError:
            print("No previous model weights found, training from scratch.")
            if not TRAINING:
                exit()

    grid = Grid(MODEL, img, CHANNELS, GRID_SIZE)
    grid.grid_interaction()