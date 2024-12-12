from med_nca import BackboneNCA
from loss import DiceLoss
import os
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Compose, Resize
import torchvision.transforms.functional as F
from torch.utils.data import random_split, DataLoader, Dataset, Subset
import torch.nn as nn
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random

torch.cuda.empty_cache()  # Clear unused memory

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Transformations
transform = Compose([
    Resize((256, 256)),  # Resize images and trimaps to 256x256, downsample by a scale of 4 to 64x64 then b2 it up 
    ToTensor()
])

# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, images_folder, trimaps_folder, transform=None):
        self.images_folder = images_folder
        self.trimaps_folder = trimaps_folder
        self.transform = transform
        self.images = sorted(os.listdir(images_folder))
        self.trimaps = sorted(os.listdir(trimaps_folder))

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

        return image, trimap


b1 = BackboneNCA(
    channel_n=64,          
    fire_rate=0.5,         
    device=device,         
    hidden_size=128,       
    input_channels=3   
).to(device)

b2 = BackboneNCA(
    channel_n=64,         
    fire_rate=0.5,      
    device=device,
    hidden_size=128,
    input_channels=3
).to(device)

optimizer_b1 = torch.optim.Adam(b1.parameters(), lr=3e-4)
optimizer_b2 = torch.optim.Adam(b2.parameters(), lr=3e-4)  

diceloss = DiceLoss()
criterion = nn.BCELoss()

images_folder = '/home/labadmin/dev/imagesegment/images'
trimaps_folder = '/home/labadmin/dev/imagesegment/trimaps_colored'
output_folder = '/home/labadmin/dev/imagesegment/outputimages'

# Instantiate the dataset
dataset = CustomDataset(images_folder, trimaps_folder, transform=transform)

# Determine indices for training and validation sets
dataset_size = len(dataset)
last_six_indices = list(range(dataset_size - 6, dataset_size))
remaining_indices = list(range(dataset_size - 6))

# Split remaining dataset into training and validation sets
remaining_train_size = int(0.8 * len(remaining_indices))
remaining_val_size = len(remaining_indices) - remaining_train_size
train_indices, val_indices = random_split(remaining_indices, [remaining_train_size, remaining_val_size])

# Combine the last 6 images with the training set
train_indices = list(train_indices) + last_six_indices

# Create Subset datasets
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)
EPOCH = 30
STEPS = 30

for epoch in range(EPOCH):  # Use 'epoch' instead of 'i' for clarity
    b1.train()
    b2.train()

    for iteration, (data, target) in enumerate(train_loader):  # Use 'iteration' for loop variable
        data, target = data.to(device), target.to(device)
        
        # 1: Downsample the image
        downscaled_data = F.resize(data, (64, 64)).to(device)

        # 2: Apply b1 for s steps 
        for _ in range(STEPS):
            downscaled_data = b1(downscaled_data).to(device)
        
        torch.cuda.empty_cache()  # Move this outside of the iteration loop for better memory management

        # 3: Upscale b1 output 
        upscaled_data = F.resize(downscaled_data, (256, 256)).to(device)

        # 4: Replace the first channel with the high-resolution image; WTF IS THIS EVEN DOING?!
        upscaled_data[:, 0, :, :] = data[:, 0, :, :]

        # 5: Get a random patch and apply b2
        patch_size = (64, 64)
        i, j, h, w = transforms.RandomCrop.get_params(upscaled_data, output_size=patch_size)
        patch_input = F.crop(upscaled_data, i, j, h, w).to(device)
        patch_target = F.crop(target, i, j, h, w).to(device)

        # Apply b2 for s steps
        for _ in range(STEPS):
            patch_input = b2(patch_input)

        patch_input = torch.sigmoid(patch_input)

        # Calculate the loss
        bceloss = criterion(patch_input, patch_target)
        loss = bceloss

        # Backpropagate and optimization
        optimizer_b1.zero_grad()
        optimizer_b2.zero_grad()
        loss.backward()
        optimizer_b1.step()
        optimizer_b2.step()

        # Print out loss every few iterations if necessary
        #if iteration % 10 == 0:  # Print every 10 iterations for cleaner output
        print(f"Epoch [{epoch+1}/{EPOCH}], Iteration [{iteration+1}/{len(train_loader)}], Loss: {loss.item()}")

    torch.cuda.empty_cache()  # Clear cache at the end of each epoch

# Save model weights
torch.save(b1.state_dict(), 'b1_weights.pth')
torch.save(b2.state_dict(), 'b2_weights.pth')
print("Model weights saved.")
