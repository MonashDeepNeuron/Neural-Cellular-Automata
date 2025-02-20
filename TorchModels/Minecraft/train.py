import torch
from torch import Tensor
import torch.nn as nn
from model import NCA_3D
import random
import matplotlib.pyplot as plt
import numpy as np
import trimesh

if torch.cuda.is_available():
    torch.set_default_device("cuda")


def visualise(imgTensor, isNCAVoxel = False, filenameBase="test", anim=False, save=True, show=True):
    """
    Visualise a designated snapshot of the grid specified by idx and axis.

    - isNCAVoxel: If True, the input tensor is in the format of NCA (batch, channel, x, y, z)
    If False, the input tensor is in the format of target_voxel (x, y, z, channel)
    """

    if isNCAVoxel:
        imgTensor = imgTensor[0, :, :, :, :]
        imgTensor = imgTensor.permute(1, 2, 3, 0)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1, 1, 1])
    ax.voxels(imgTensor[:, :, :, 0], edgecolor="k")
    plt.show()

def new_seed(target_voxel, batch_size=1):
    """
    seed is like a cube map that sets a singular pixel activated
    """
    SHAPE = [target_voxel.shape[i] for i in range(len(target_voxel.shape))]
    seed = torch.zeros(
        batch_size, CHANNELS, SHAPE[0], SHAPE[1], SHAPE[2]
        )
    
    ## Batch, channels, x, y, z
    seed[:, 0, SHAPE[0]//2, SHAPE[1]//2, 0] = 1  # Alpha channel = 1
    return seed


def load_image(imagePath: str):
    """
    Get the output image, which is a OBJ, and transform it into a tensor such that we can use it as the target image for the output.
    1. Obtain mesh
    2. Obtain the vertices from the mesh
    3. Convert colours to mesh
    """

    mesh_path = imagePath
    mesh = trimesh.load(mesh_path, force = "mesh")    #
    if mesh.is_empty:
        raise ValueError("The mesh is empty and cannot be voxelized.")

    print("Mesh loaded successfully with vertices:", mesh.vertices.shape)
    print("Mesh loaded successfully with faces:", mesh.faces.shape)

    # Calculate the pitch for each axis to match the desired number of voxels
    voxel = mesh.voxelized(pitch = 0.15) 
    ## TODO: convert texture information into colours
    # colours = mesh.visual.to_color().vertex_colors
    # colours = np.asarray(colours)

    ## get vertices from mesh
    mesh_vertices = mesh.vertices
    _, vertex_idx = trimesh.proximity.ProximityQuery(mesh).vertex(voxel.points)

    # we initialize a array of zeros of size X,Y,Z,1 to contain the colors for each voxel of the voxelized mesh in the grid

    alpha_values = np.zeros([voxel.shape[0], voxel.shape[1], voxel.shape[2], 1])
    ## map vertices to grid coordinates
    for idx, vert in enumerate(vertex_idx):
        vox_verts = voxel.points_to_indices(mesh_vertices[vert])
        alpha_values[vox_verts[0], vox_verts[1], vox_verts[2]] = 1
    
    voxel_tensor = torch.tensor(alpha_values, dtype=torch.float32)

    return voxel_tensor


def forward_pass(model: nn.Module, state, updates, target, record=False):  # TODO
    """
    Run a forward pass consisting of `updates` number of updates
    If `record` is true, then records the state in a tensor to animate and saves the video
    Returns the final state
    """
    if record:
        frames_array = Tensor(updates, CHANNELS, target_voxel.shape[0], target_voxel.shape[1], target_voxel.shape[2])
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
 
        loss = LOSS_FN(
            output[:, :, :, 0:1], target
        )  
        batch_losses[batch_idx] = loss.item()
        loss.backward()
        optimiser.step()

    print(f"batch loss = {batch_losses.cpu().numpy()}")  ## print on cpu


def train(model: nn.Module, target: torch.Tensor, optimiser, record=False):  # TODO
    device = next(model.parameters()).device
    
    target = target.to(device)

    try:
        training_losses = []
        for epoch in range(EPOCHS):
            model.train()
            if record:
                outputs = torch.zeros_like(batch)

            batch = new_seed(target_voxel=target_voxel, batch_size=BATCH_SIZE)
            batch = batch.to(device)

            update_pass(model, batch, target, optimiser)

            # test_seed = new_seed(target_voxel=target_voxel, batch_size=BATCH_SIZE)
            MODEL.eval()
            # test_run = forward_pass(MODEL, test_seed, 32, target)
           

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
    TRAINING = True 
    GRID_SIZE = 32
    CHANNELS = 16

    MODEL = NCA_3D()
    EPOCHS = 20
    BATCH_SIZE = 32
    UPDATES_RANGE = [64, 96]

    LR = 1e-3

    optimizer = torch.optim.Adam(MODEL.parameters(), lr=LR)
    LOSS_FN = torch.nn.MSELoss(reduction="mean")

    target_voxel = load_image("./tree.obj")

    if TRAINING:
        MODEL, losses = train(MODEL, target_voxel, optimizer)
        torch.save(MODEL.state_dict(), "Minecraft.pth")

    ## Switch state to evaluation to disable dropout e.g.
    MODEL.eval()

    ## Plot final state of evaluation OR evaluation animation
    img = new_seed(target_voxel=target_voxel, batch_size=1)
    model_generated_voxel = forward_pass(MODEL, img, 200, record=True, target=target_voxel)
    anim = visualise(model_generated_voxel, isNCAVoxel=True, anim=True)
