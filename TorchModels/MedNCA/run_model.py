"""
Example of intended run of the med-nca model
"""
#### IMPORTS ####

import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import random_split, DataLoader, Dataset, Subset
import random
import os

import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Compose, Resize
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
import torchvision
import torchvision.transforms.functional as F

from PIL import Image

from model2 import GCA
import visualiser

from loss import CustomLoss

from train2 import CustomDataset

def load_dataset(images_folder, trimaps_folder, transform, channels, input_channels):
    # Instantiate the dataset
    dataset = CustomDataset(images_folder, trimaps_folder, transform=transform, 
                            channels= channels, img_channels = input_channels, apply_custom_transform=True)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    return dataloader

def run_frames_1(model: nn.Module, state, updates, record=False):
    if record:
        frames_array = Tensor(updates, CHANNELS, GRID_SIZE, GRID_SIZE)
        for i in range(updates):
            state = model(state)
            frames_array[i] = F.resize(state, (GRID_SIZE, GRID_SIZE)).to(DEVICE)
        
        return state, frames_array

    else:
        for i in range(updates):
            state = model(state)

    return state

def run_frames_2(model: nn.Module, state, updates, record=False):
    if record:
        frames_array = Tensor(updates, CHANNELS, GRID_SIZE, GRID_SIZE)
        state = torch.nn.functional.pad(state, (1, 1, 1, 1), mode="circular")
        for update_no in range(updates):
            for patch_i in range(4):
                for patch_j in range(4):
                    i, j, h, w = (patch_i* GRID_SIZE//4), (patch_j* GRID_SIZE//4), GRID_SIZE//4 + 2, GRID_SIZE//4 + 2
                    # print(i,j,h,w)
                    # print(state[:, :, i:i+h, j:j+w].shape)
                    state[:, :, i+1:i+h-1, j+1:j+w-1] = model(state[:, :, i:i+h, j:j+w], pad = False)
            frames_array[update_no] = state[:, :, 1:-1, 1:-1]
        
        state = state[:, :, 1:-1, 1:-1]
        return state, frames_array

    else:
        for i in range(updates):
            for i in range(16):
                i, j, h, w = transforms.RandomCrop.get_params(state, output_size=(GRID_SIZE//4, GRID_SIZE//4))
                state[:, :, i:i+h, j:j+w] = model(state[:, :, i:i+h, j:j+w])


    return state


def forward_pass(model1: nn.Module, model2: nn.Module, batch, 
                 targets, updates, record=False, lower_model_only = False):
    """
    Run a forward pass consisting of `updates` number of updates
    If `record` is true, then records the state in a tensor to animate and saves the video
    Returns the final state
    """
    #Downsample images and targets by 4  
    batch_downsampled = F.resize(batch, (GRID_SIZE//4, GRID_SIZE//4)).to(DEVICE)
    
    # Apply model for 'updates' steps
    if not record:
        output = run_frames_1(model1, batch_downsampled, updates)
    else: 
        output, recording1 = run_frames_1(model2, batch_downsampled, updates, record = True)


    # upscale this back to the originalgrid size
    output_sized = F.resize(output, (GRID_SIZE, GRID_SIZE)).to(DEVICE)

    #replace with high res image 
    output_sized[:, :3, :, :] = batch[:, :3, :, :]

    if not lower_model_only:
        # Only recommendable with small batch and model in eval mode.
        if not record: 
            output_sized = run_frames_2(model2, output_sized, updates)
        else: 
            output_sized, recording2 = run_frames_2(model2, output_sized, updates, record = True)

    if (not record):
        return output_sized
    
    recording = torch.cat((recording1, recording2), dim=0)
    return output_sized, recording

def get_visuals(model1, model2, data, i):
    output, recording = forward_pass(model1, model2, data, target, updates=UPDATES_RANGE[0], record=True)
    visualiser.plotRGB(output[:, 3:], filenameBase=f"output{i}")
    visualiser.plotRGB(data, filenameBase=f"data{i}")
    visualiser.plotRGB(target, filenameBase=f"target{i}")
    visualiser.animateRGB(recording[:, 3:], filenameBase=f"recording{i}", alpha=False)
    visualiser.visualiseHidden(recording, channels_idxs=[3,4,6,7,8,9,10,11], filenameBase=f"hidden{i}")



def initialiseGPU(model1, model2):
    ## Check if GPU available
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} is available.")

    ## Configure device as GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1 = model1.to(device)
    model2 = model2.to(device)
    return model1, model2, device

if __name__ == "__main__":

    GRID_SIZE = 128
    CHANNELS = 64
    INPUT_CHANNELS = 3

    MODEL1 = GCA(n_channels=CHANNELS, input_channels=INPUT_CHANNELS)
    MODEL2 = GCA(n_channels=CHANNELS, input_channels=INPUT_CHANNELS)
    MODEL1, MODEL2, DEVICE = initialiseGPU(MODEL1, MODEL2)
    EPOCHS = 2000  # 100 epochs for best results
    ## 30 epochs, once loss dips under 0.8 switch to learning rate 0.0001

    BATCH_SIZE = 1
    UPDATES_RANGE = [64, 96]

    # These path weights are the load and save location
    # Old model will be updated with the new training result
    MODEL_PATH1 = "imgseg_big_mse_50ep.pth"
    MODEL_PATH2 = "imgseg_small_mse_50ep.pth"

    images_folder = './TorchModels/ImageSegmentation/images'
    trimaps_folder = './TorchModels/ImageSegmentation/trimaps_colored'
    output_folder = './TorchModels/ImageSegmentation/outputimages'


    transform = transforms.Compose([
        Resize((GRID_SIZE, GRID_SIZE)),  # Resize images and trimaps to 256x256, downsample by a scale of 4 to 64x64 then b2 it up 
        ToTensor()
    ])

    dataloader = load_dataset(images_folder, trimaps_folder, transform, CHANNELS, INPUT_CHANNELS)
    ## Load model weights if available
    try:
        MODEL1.load_state_dict(
            torch.load(
                MODEL_PATH1,
                weights_only=True,
                map_location=torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                ),
            )
        )
        print("Loaded model1 weights successfully!")
        MODEL2.load_state_dict(
            torch.load(
                MODEL_PATH1,
                weights_only=True,
                map_location=torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                ),
            )
        )
        print("Loaded model2 weights successfully!")
    except FileNotFoundError:
        print("No previous model weights found")
        exit()

    ## Switch state to evaluation to disable dropout e.g.
    MODEL1.eval()
    MODEL2.eval()

    for i, (data, target) in enumerate(dataloader):
        data.to(DEVICE)
        target.to(DEVICE)
        
        get_visuals(MODEL1, MODEL2, data, i)
        