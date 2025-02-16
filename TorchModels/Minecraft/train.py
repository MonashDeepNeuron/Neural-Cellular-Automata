#### IMPORTS ####

import torch
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
import torchvision
from torch import Tensor
import torch.nn as nn

from model import NCA_3D
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
import numpy as np
import os
import trimesh

# Use GPU if available
if torch.cuda.is_available():
    torch.set_default_device("cuda")


def visualise(imgTensor, filenameBase="test", anim=False, save=True, show=True):
    """
    Visualise a designated snapshot of the grid specified by idx and axis.

    Input in form (channels, x,y,z)
    """

    if len(imgTensor.shape) < 5:
        imgTensor.unsqueeze(0)

    imgTensor = imgTensor.squeeze(0).permute(1, 2, 3, 4, 0).cpu().detach().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # def update(imgIdx):
    # We're only interested in the RGBalpha channels, and need numpy representation for plt
    # img = imgTensor[imgIdx].clip(0, 1).squeeze().permute(1, 2, 3, 0)

    # if anim:
    #     plt.suptitle("Update " + str(imgIdx))

    ## define references to vectors of each colour
    colours = np.zeros_like(imgTensor)
    colours[..., 0] = imgTensor[..., 0]  # red
    colours[..., 1] = imgTensor[..., 1]
    colours[..., 2] = imgTensor[..., 2]

    imgTensor = imgTensor[:, :, :, :, 0]
    print(imgTensor.shape)
    ## x, y, z
    ax.voxels(imgTensor[:, :, :, 3], edgecolor="k")

    # End animation update
    plt.show()
    # if not anim:
    #     update(0)
    #     if save:
    #         plt.savefig(filenameBase + ".png", bbox_inches="tight")

    #     # Display image
    #     if show:
    #         plt.show()
    #         plt.close("all")
    #     return

    # ani = animation.FuncAnimation(fig, update, frames=len(imgTensor), repeat=False)
    # # Display image
    # if save:
    #     # To save the animation using Pillow as a gif
    #     writer = animation.PillowWriter(
    #         fps=15, metadata=dict(artist="Me"), bitrate=1800
    #     )
    #     ani.save(filenameBase + ".gif", writer=writer)

    # if show:
    #     plt.show()
    #     plt.close("all")

    # return ani


def new_seed(batch_size=1):
    """
    seed is like a cube map that sets a singular pixel activated
    """
    seed = torch.zeros(
        batch_size, CHANNELS, 6, 8, 6
    )  # create a cube map 15 = depth, 11 is width, 13 height
    seed[:, 3, 4, 3, 0] = 1  # Alpha channel = 1
    return seed


def load_image(imagePath: str):
    """
    Get the output image, which is a OBJ, and transform it into a tensor such that we can use it as the target image for the output.
    1. Obtain mesh
    2. Obtain the vertices from the mesh
    3. Convert colours to mesh

    """
    RESOLUTION = 16
    HIDDEN_CHANNELS = 9

    mesh_path = imagePath
    mesh = trimesh.load(mesh_path, force = "mesh")

    if mesh.is_empty:
        raise ValueError("The mesh is empty and cannot be voxelized.")

    print("Mesh loaded successfully with vertices:", mesh.vertices.shape)
    print("Mesh loaded successfully with faces:", mesh.faces.shape)

## voxelize mesh with resolution 1 (dist between two most adjacent vertices is one)
    # pitch=mesh.extents.min()
    # pitch = 2/5
    
    # Calculate the pitch to match the bounding box dimensions
    # In this case, we'll use the mesh dimension as the number of voxels.
    #pitch = np.float64(0.14284999999999992) # This ensures 1 voxel per unit dimension (simplified example)

    bounding_box = mesh.bounds
    # Calculate the bounding box size (length along each axis)
    bbox_size = bounding_box[1] - bounding_box[0]

    # Desired number of voxels along each axis (you can adjust this to match the mesh dimensions)
    desired_voxels = (5, 7, 5)

    # Calculate the pitch for each axis to match the desired number of voxels
    pitch = tuple(bbox_size[i] / desired_voxels[i] for i in range(3))

    voxel = mesh.voxelized(pitch=min(pitch))
    # voxel = mesh.voxelized(mesh.min)

    ## convert texture information into colours
    colours = mesh.visual.to_color().vertex_colors
    colours = np.asarray(colours)

    ## get vertices from mesh
    mesh_vertices = mesh.vertices
    _, vertex_idx = trimesh.proximity.ProximityQuery(mesh).vertex(voxel.points)

    # we initialize a array of zeros of size X,Y,Z,4 to contain the colors for each voxel of the voxelized mesh in the grid
    cube_colour = np.zeros([voxel.shape[0], voxel.shape[1], voxel.shape[2], 4])

    ## map vertices to grid coordinates
    # We loop through all the calculated closest voxel points
    for idx, vert in enumerate(vertex_idx):
        # Get the voxel grid index of each closest voxel center point
        vox_verts = voxel.points_to_indices(mesh_vertices[vert])
        # Get the color vertex color
        curr_colour = colours[vert]
        # Set the alpha channel of the color
        curr_colour[3] = 255
        # add the color to the specific voxel grid index
        cube_colour[vox_verts[0], vox_verts[1], vox_verts[2], :] = curr_colour

    # Convert the voxel grid and its colors to a PyTorch tensor
    # The tensor will have shape (X, Y, Z, 4), where 4 corresponds to RGBA channels
    voxel_tensor = torch.tensor(cube_colour, dtype=torch.float32)

    # ## this breaks if tensor not of shaep 15, 13, 11
    # padding = (0, 0, 1, 0, 1, 0, 1, 0)

    # voxel_tensor = torch.nn.functional.pad(
    #     voxel_tensor, padding, mode="constant", value=0
    # )
    # print("Shape of the resultant tensor:", voxel_tensor.shape)

    return voxel_tensor


def forward_pass(model: nn.Module, state, updates, target, record=False):  # TODO
    """
    Run a forward pass consisting of `updates` number of updates
    If `record` is true, then records the state in a tensor to animate and saves the video
    Returns the final state
    """
    if record:
        frames_array = Tensor(updates, CHANNELS, targetVoxel.shape[0], targetVoxel.shape[1], targetVoxel.shape[2])
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
        output = forward_pass(model, batch[batch_idx].unsqueeze(0), updates, target=target)

        ## apply pixel-wise MSE loss between RGBA channels in the grid and the target pattern
        output = output.squeeze(0).permute(1, 2, 3, 0)
        print(output.shape)
        print(target.shape)
        loss = LOSS_FN(
            output[:, :, :, 0:4], target
        )  # FLAG, indexation may need to be changed here
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

            batch = new_seed(BATCH_SIZE)
            batch = batch.to(device)

            update_pass(model, batch, target, optimiser)

            test_seed = new_seed(1)
            MODEL.eval()
            test_run = forward_pass(MODEL, test_seed, 32, target)
            # training_losses.append(
            #     LOSS_FN(test_run[0, 0:4], target).cpu().detach().numpy()
            # )
            # print(f"Epoch {epoch} complete, loss = {training_losses[-1]}")

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

    GRID_SIZE = 32
    CHANNELS = 16

    MODEL = NCA_3D()
    # MODEL = initialiseGPU(MODEL)
    EPOCHS = 15  # 50  # 100 epochs for best results
    ## 30 epochs, once loss dips under 0.8 switch to learning rate 0.0001

    BATCH_SIZE = 32
    UPDATES_RANGE = [30, 48]

    LR = 1e-3

    optimizer = torch.optim.Adam(MODEL.parameters(), lr=LR)
    LOSS_FN = torch.nn.MSELoss(reduction="mean")

    targetVoxel = load_image("./TorchModels/Minecraft/source/tree.obj")

    if TRAINING:
        MODEL, losses = train(MODEL, targetVoxel, optimizer)

        print(losses)
        ## Plot loss
        plt.plot(range(len(losses)), losses)

        ## Save the model's weights after training
        torch.save(MODEL.state_dict(), "Minecraft.pth")

    ## Switch state to evaluation to disable dropout e.g.
    MODEL.eval()

    ## Plot final state of evaluation OR evaluation animation
    img = new_seed(1)
    video = forward_pass(MODEL, img, 200, record=True, target=targetVoxel)
    anim = visualise(video, anim=True)
