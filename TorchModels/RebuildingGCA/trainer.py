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

from utils import initialiseGPU, new_seed, load_image, visualise

"""
This class creates an instance of a model and provides methods to train it
1. Without pooling
2. With pooling
"""

## constants to make class attributse
TRAINING = True  # Is our purpose to train or are we just looking rn?

GRID_SIZE = 32
CHANNELS = 16

MODEL = GCA()
MODEL = initialiseGPU(MODEL)
EPOCHS = 30

BATCH_SIZE = 32
UPDATES_RANGE = [64, 96]

LR = 1e-3

LOSS_FN = torch.nn.MSELoss(reduction="mean")
MODEL_PATH = "model_weights_logo.pthj"


class Trainer:
    optimizer = torch.optim.Adam(MODEL.parameters(), lr=LR)

    def __init__(self, with_sample_pooling=True):
        self.with_sample_pooling = with_sample_pooling

    def forward_pass(self, model: nn.Module, state, updates, record=False):  # TODO
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

    def update_pass(self, model, batch, target, optimiser):
        """
        Back calculate gradient and update model paramaters
        """
        device = next(model.parameters()).device
        batch_losses = torch.zeros(BATCH_SIZE, device=device)
        for batch_idx in range(BATCH_SIZE):
            optimiser.zero_grad()
            updates = random.randrange(UPDATES_RANGE[0], UPDATES_RANGE[1])
            output = self.forward_pass(model, batch[batch_idx].unsqueeze(0), updates)

            ## apply pixel-wise MSE loss between RGBA channels in the grid and the target pattern
            loss = LOSS_FN(output[0, 0:4], target)
            ## .item() removes computational graph for memory efficiency
            batch_losses[batch_idx] = loss.item()
            loss.backward()
            optimiser.step()

        print(f"batch loss = {batch_losses.cpu().numpy()}")  ## print on cpu

    def train_with_pooling(
        self, model: nn.Module, target: torch.Tensor, optimiser, record=False
    ):
        """
        TRAINING PROCESS:
            - Define training data storage variables
            - For each epoch:
                - Initialise pooling
                    - Sample N seeds from pool to use as input to forward pass (our batch)
                    - Forward pass (runs the model on the batch)
                    - Backward pass (calculates loss and updates params)
                    - Update pool with final states from forward pass
                    - Calculate loss for all samples from pool, replace worst sample with starting seed
                    - Repeat steps 2 onwards, pool has been updated
                - SANITY CHECK: check current loss and record loss
                - Save model if this is the best model TODO
            - Return the trained model
        """

    def train(
        self, model: nn.Module, target: torch.Tensor, optimiser, record=False
    ):  # TODO
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
            # minibatch epoch =N
            training_losses = []
            for epoch in range(EPOCHS):
                model.train()
                if record:
                    outputs = torch.zeros_like(batch)

                batch = new_seed(BATCH_SIZE)  ## TODO duplicate seed
                batch = batch.to(device)

                ## Optimisation step
                self.update_pass(model, batch, target, optimiser)

                ## TODO: Assess accuracy. Can we remove this for faster training?
                test_seed = new_seed(1)
                MODEL.eval()
                test_run = self.forward_pass(MODEL, test_seed, 64)
                training_losses.append(
                    LOSS_FN(test_run[0, 0:4], target).cpu().detach().numpy()
                )
                print(f"Epoch {epoch} complete, loss = {training_losses[-1]}")

                # check modulo minibatch epoch
                # for input
                # run lradj.adjust_learning_rate
                #

        except KeyboardInterrupt:
            pass

        if record:
            return (model, training_losses, outputs)
        else:
            return model, training_losses

    def __call__(self):
        targetImg = load_image("logo.png")

        ## Load model weights if available
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
            MODEL, losses = self.train(MODEL, targetImg, self.optimizer)

            print(losses)
            ## Plot loss
            plt.plot(range(len(losses)), losses)

            ## Save the model's weights after training
            torch.save(MODEL.state_dict(), MODEL_PATH)

        ## Switch state to evaluation to disable dropout e.g.
        MODEL.eval()

        ## Plot final state of evaluation OR evaluation animation
        img = new_seed(1)
        video = self.trainforward_pass(MODEL, img, 200, record=True)
        anim = visualise(video, anim=True)
