import os
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Compose, Resize
import torchvision.transforms.functional as F
from torch.utils.data import random_split, DataLoader, Dataset, Subset
import torch.nn as nn
from model import NCA
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random

torch.cuda.empty_cache()  # Clear unused memory

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

# Transformations
transform = Compose([
    Resize((400, 400)),  # Resize images and trimaps to 128x128
    ToTensor()
])

def robusttraining(inputimg, targetimg):
    """Applies the same data augmentation transformations to the input image and its corresponding label."""
    # Random Resized Crop
    i, j, h, w = transforms.RandomResizedCrop.get_params(
        inputimg, scale=(0.9, 1.1), ratio=(1.0, 1.0)  # Example scale/ratio
    )
    
    # Random Horizontal Flip (manually)
    flip = random.random() < 0.1  # Flip with probability 0.2
    
    # Random Color Jitter (manually)
    brightness_factor = random.uniform(0.95, 1.05)
    contrast_factor = random.uniform(0.95, 1.07)
    saturation_factor = random.uniform(0.95, 1.05)
    hue_factor = random.uniform(-0.03, 0.03)

    # Apply transformations to both images
    augmented_input = F.resized_crop(inputimg, i, j, h, w, size=(720, 720))
    augmented_input = F.hflip(augmented_input) if flip else augmented_input
    #augmented_input = F.adjust_brightness(augmented_input, brightness_factor)
    #augmented_input = F.adjust_contrast(augmented_input, contrast_factor)
    #augmented_input = F.adjust_saturation(augmented_input, saturation_factor)
    #augmented_input = F.adjust_hue(augmented_input, hue_factor)

    augmented_target = F.resized_crop(targetimg, i, j, h, w, size=(720, 720))
    augmented_target = F.hflip(augmented_target) if flip else augmented_target
    #augmented_target = F.adjust_brightness(augmented_target, brightness_factor)
    #augmented_target = F.adjust_contrast(augmented_target, contrast_factor)
    #augmented_target = F.adjust_saturation(augmented_target, saturation_factor)
    #augmented_target = F.adjust_hue(augmented_target, hue_factor)

    return augmented_input, augmented_target

# Absolute paths to the folders
images_folder = './TorchModels/ImageSegmentation/images'
trimaps_folder = './TorchModels/ImageSegmentation/trimaps_colored'
output_folder = './TorchModels/ImageSegmentation/outputimages'

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
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Instantiate model, optimizer, and loss
model = NCA(input_channels=3, state_channels=64, hidden_channels=64, output_channels=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion = nn.MSELoss()

# Train the Model
for epoch in range(20):
    model.train()
    for i, (data, target) in enumerate(train_loader):
        if i % 5 == 0:
            # Apply robust training augmentations to both image and label
            augmented_data, augmented_target = zip(*[robusttraining(img, lbl) for img, lbl in zip(data, target)])
            # `zip()` outputs tuples, so they need to be stacked
            augmented_data = torch.stack(augmented_data)
            augmented_target = torch.stack(augmented_target)
        else:
            # Treat data and target as normal
            augmented_data, augmented_target = data, target
        
        # Move data and target to device
        augmented_data, augmented_target = augmented_data.to(device), augmented_target.to(device)

        # Forward pass, backward pass, optimizer step
        optimizer.zero_grad()

        output = model.forward(augmented_data)

        loss = criterion(output, augmented_target)
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()  # Clear unused memory this is necessary when doing 720x720

        # Print loss for debugging
        print(f"Epoch {epoch}, Iteration {i}/{len(train_loader)}, Loss: {loss.item()}")


# Save the model weights after training
torch.save(model.state_dict(), 'higher_res_robust_using350350_20epochs.pth')
print("Model weights saved.")



# Validate the Model
model.eval()
with torch.no_grad():
    val_loss = 0
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        val_loss += loss.item()
    val_loss /= len(val_loader)
    print(f"Validation Loss: {val_loss}")

