#    images_folder = '/home/labadmin/dev/imagesegment/images'
#    trimaps_folder = '/home/labadmin/dev/imagesegment/trimaps_colored'
#    output_folder = '/home/labadmin/dev/imagesegment/outputimages'

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

# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, images_folder, trimaps_folder, channels = 64, 
                 img_channels = 3, transform=None, apply_custom_transform = False):
        self.images_folder = images_folder
        self.trimaps_folder = trimaps_folder
        self.transform = transform
        self.images = sorted(os.listdir(images_folder))
        self.trimaps = sorted(os.listdir(trimaps_folder))

        self.channels = channels
        self.img_channels = img_channels
        self.custom_transform = apply_custom_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_folder, self.images[idx])
        trimap_path = os.path.join(self.trimaps_folder, self.trimaps[idx])
        image = Image.open(img_path).convert("RGB")
        trimap = Image.open(trimap_path).convert("RGB")  # Keep trimap in RGB

        if self.transform:
            image = self.transform(image)
            trimap = self.transform(trimap)

        if (self.custom_transform):
            """Applies the same data augmentation transformations to the input 
            image and its corresponding label."""
            # Random Resized Crop
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                image, scale=(0.5, 1.2), ratio=(1.0, 1.0)  # Example scale/ratio
            )
            
            # Random Horizontal Flip (manually)
            flip = random.random() < 0.1  # Flip with probability 0.2
            
            # # Random Color Jitter (manually)
            # brightness_factor = random.uniform(0.95, 1.05)
            # contrast_factor = random.uniform(0.95, 1.07)
            # saturation_factor = random.uniform(0.95, 1.05)
            # hue_factor = random.uniform(-0.03, 0.03)

            # Apply transformations to both images
            image = F.resized_crop(image, i, j, h, w, size=image.shape[1:2])
            image = F.hflip(image) if flip else image
            #image = F.adjust_brightness(image, brightness_factor)
            #image = F.adjust_contrast(image, contrast_factor)
            #image = F.adjust_saturation(image, saturation_factor)
            #image = F.adjust_hue(image, hue_factor)

            trimap = F.resized_crop(trimap, i, j, h, w, size=image.shape[1:2])
            trimap = F.hflip(trimap) if flip else trimap


        image = torch.concat((image, torch.zeros(self.channels-self.img_channels, image.size(1), image.size(2))))

        return image, trimap



def load_datasets(images_folder, trimaps_folder, transform, channels, input_channels):
    # Instantiate the dataset
    dataset = CustomDataset(images_folder, trimaps_folder, transform=transform, 
                            channels= channels, img_channels = input_channels, apply_custom_transform=True)

    # Determine indices for training and validation sets
    dataset_size = len(dataset)
    # last_six_indices = list(range(dataset_size - 6, dataset_size))
    # remaining_indices = list(range(dataset_size - 6))
    remaining_indices = list(range(dataset_size))
    # Split remaining dataset into training and validation sets
    remaining_train_size = int(0.8 * len(remaining_indices))
    remaining_val_size = len(remaining_indices) - remaining_train_size
    train_indices, val_indices = random_split(remaining_indices, [remaining_train_size, remaining_val_size])

    # Combine the last 6 images with the training set
    train_indices = list(train_indices) # + last_six_indices

    # Create Subset datasets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    return train_loader, val_loader



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

def run_frames(model: nn.Module, state, updates, record=False):
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


def forward_pass(model1: nn.Module, model2: nn.Module, batch, 
                 targets, updates, record=False, lower_model_only = False, full_image = False):
    """
    Run a forward pass consisting of `updates` number of updates
    If `record` is true, then records the state in a tensor to animate and saves the video
    Returns the final state
    """
    #Downsample images and targets by 4  
    batch_downsampled = F.resize(batch, (GRID_SIZE//4, GRID_SIZE//4)).to(DEVICE)
    
    # Apply model for 'updates' steps
    output = run_frames(model1, batch_downsampled, updates)

    # upscale this back to the originalgrid size
    output_sized = F.resize(output, (GRID_SIZE, GRID_SIZE)).to(DEVICE)

    #replace with high res image 
    output_sized[:, :3, :, :] = batch[:, :3, :, :]

    if not lower_model_only:
        #get a random patch
        if (not full_image):
            i, j, h, w = transforms.RandomCrop.get_params(output_sized, output_size=(GRID_SIZE//4, GRID_SIZE//4))
            patch_input = F.crop(output_sized, i, j, h, w).to(DEVICE)
            patch_target = F.crop(targets, i, j, h, w).to(DEVICE)

            # get output from b2 
            output_patch = run_frames(model2, patch_input, updates)
        else:
            # Only recommendable with small batch and model in eval mode.
            output_patch = run_frames(model2, output_sized, updates)
            patch_target = targets
    else:
        output_patch = output_sized
        patch_target = targets

    return output_patch, patch_target
    

def update_pass(model1, model2, batch, targets, optimiser1, optimiser2, lower_model_only = False):
    """
    Back calculate gradient and update model paramaters
    """
    device = next(model1.parameters()).device
    device = next(model2.parameters()).device
    batch_losses = torch.zeros(BATCH_SIZE, device=device)
    
    updates = random.randrange(UPDATES_RANGE[0], UPDATES_RANGE[1])

    output_patch, patch_target = forward_pass(model1, model2, batch, targets, updates, lower_model_only = lower_model_only)
    
    ## apply pixel-wise MSE loss between RGBA channels in the grid and the target pattern
    loss = LOSS_FN(output_patch[:, 3:6], patch_target)
    
    optimiser1.zero_grad()
    optimiser2.zero_grad()
    loss.backward()
    optimiser1.step()
    optimiser2.step()

    ## .item() removes computational graph for memory efficiency
    batch_losses = loss.item()

    print(f"batch loss = {batch_losses}")  ## print on cpu


def train(model1: nn.Module, model2:nn.Module, optimiser1, optimiser2, 
train_loader, val_loader, record=False, model_save1 = "MODEL_PATH1.pth", model_save2 = "MODEL_PATH2.pth"):  # TODO
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
    device = next(model1.parameters()).device
    best_loss = None
    best_weights = (model1.state_dict(), model2.state_dict())

    if record:
        recording = None

    try:
        # minibatch epoch =N
        training_losses = []
        for epoch in range(EPOCHS):
            model1.train()
            model2.train()
            i=0
            for batch, target in train_loader:
                i+=1
                print(f"Iteration {i} of {len(train_loader)}")
                batch = batch.to(device)
                targets = target.to(device)
                update_pass(model1, model2, batch, targets, optimiser1, optimiser2, 
                            lower_model_only=(epoch%2 == 0))
                
            test, test_target = next(iter(val_loader))

            model1.eval()
            model2.eval()
            test_run, test_target_scaled = forward_pass(model1, model2, test, test_target, UPDATES_RANGE[1])
            training_losses.append(
                LOSS_FN(test_run[:, 3:6, : , :], test_target_scaled).cpu().detach().numpy()
            )

            if ((not best_loss) or best_loss > training_losses[-1]) :
                print("--Saving weights", end = " ")
            
                best_loss = training_losses[-1]
                torch.save(MODEL1.state_dict(), model_save1)
                torch.save(MODEL2.state_dict(), model_save2)
                best_weights = (model1.state_dict(), model2.state_dict())

            if record:
                if recording == None:
                    recording = torch.cat((test_run[0, :3].unsqueeze(0).detach(),test_run[0, 3:6].unsqueeze(0).detach(), test_target_scaled[0].unsqueeze(0).detach()), dim=0)
                else:
                    recording = torch.cat((recording, test_run[0, :3].unsqueeze(0).detach(),test_run[0, 3:6].unsqueeze(0).detach(), test_target_scaled[0].unsqueeze(0).detach()), dim=0) 
            
            print("best_loss =", best_loss)
            print(f"Epoch {epoch} complete, loss = {training_losses[-1]}")


    except KeyboardInterrupt:
        pass

    model1.load_state_dict(best_weights[0])
    model2.load_state_dict(best_weights[1])
    if record:
        return (model1, model2, training_losses, recording)
    else:
        return (model1, model2, training_losses)


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
    torch.autograd.set_detect_anomaly(True)

    TRAINING = True  # Is our purpose to train or are we just looking rn?

    GRID_SIZE = 128
    CHANNELS = 64
    INPUT_CHANNELS = 3

    MODEL1 = GCA(n_channels=CHANNELS, input_channels=INPUT_CHANNELS)
    MODEL2 = GCA(n_channels=CHANNELS, input_channels=INPUT_CHANNELS)
    MODEL1, MODEL2, DEVICE = initialiseGPU(MODEL1, MODEL2)
    EPOCHS = 30  # 100 epochs for best results
    ## 30 epochs, once loss dips under 0.8 switch to learning rate 0.0001

    BATCH_SIZE = 6
    UPDATES_RANGE = [64, 96]

    LR1 = 1e-3
    LR2 = 1e-3

    optimiser1 = torch.optim.Adam(MODEL1.parameters(), lr=LR1)
    optimiser2 = torch.optim.Adam(MODEL2.parameters(), lr=LR2)
    LOSS_FN = CustomLoss()

    # These path weights are the load and save location
    # Old model will be updated with the new training result
    MODEL_PATH1 = "imgseg_big128.pth"
    MODEL_PATH2 = "imgseg_small128.pth"

    images_folder = './TorchModels/ImageSegmentation/images'
    trimaps_folder = './TorchModels/ImageSegmentation/trimaps_colored'
    output_folder = './TorchModels/ImageSegmentation/outputimages'


    transform = transforms.Compose([
        Resize((GRID_SIZE, GRID_SIZE)),  # Resize images and trimaps to 256x256, downsample by a scale of 4 to 64x64 then b2 it up 
        ToTensor()
    ])

    train_loader, val_loader = load_datasets(images_folder, trimaps_folder, transform, CHANNELS, INPUT_CHANNELS)
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
        print("No previous model weights found, training from scratch.")
        if not TRAINING:
            exit()



    if TRAINING:
        MODEL1, MODEL2, losses, output = train(MODEL1, MODEL2, optimiser1, optimiser2, train_loader, 
                                               val_loader, record=True, model_save1=MODEL_PATH1, model_save2=MODEL_PATH2)
        # losses_file = open("losses.txt", "a")
        # losses_file.write(losses)
        # losses_file.close()

        print(losses)
        ## Plot loss
        plt.plot(range(len(losses)), losses)
        plt.savefig("Loss plot.png")

        ## Save the model's weights after training
        torch.save(MODEL1.state_dict(), MODEL_PATH1)
        torch.save(MODEL2.state_dict(), MODEL_PATH2)

        visualiser.animateRGB(output, filenameBase="recording", alpha = False, fps = 2)

    ## Switch state to evaluation to disable dropout e.g.
    MODEL1.eval()
    MODEL2.eval()


    data, target = next(iter(val_loader))
    data.to(DEVICE)
    target.to(DEVICE)

    patch_output, patch_target = forward_pass(MODEL1, MODEL2, data, target, updates=UPDATES_RANGE[0], full_image = True)
    visualiser.plotRGB(patch_output[:, 3:], filenameBase="output")
    visualiser.plotRGB(data, filenameBase="data")
    visualiser.plotRGB(target, filenameBase="target")
    visualiser.plotRGB(patch_target, filenameBase="patch_target")
    visualiser.plotRGB(patch_output, filenameBase="patch")

    # Save model weights
    torch.save(MODEL1.state_dict(), MODEL_PATH1)
    torch.save(MODEL2.state_dict(), MODEL_PATH2)

    print("Model weights saved.")

