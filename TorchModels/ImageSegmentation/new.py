import os
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Compose, Resize
from torch.utils.data import random_split, DataLoader, Dataset
import torch.nn as nn
from model import NCA
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random

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
    Resize((128, 128)),  # Resize images and trimaps to 128x128
    ToTensor()
])

def robusttraining(inputimg, targetimg):
    """Applies the same data augmentation transformations to the input image and its corresponding label."""
    
    augmentations = transforms.Compose([
        transforms.RandomResizedCrop(size=(128, 128), scale=(0.8, 1.0)), 
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
        transforms.RandomRotation(degrees=15), 
    ])
    
    # Apply augmentations to both the input image and the target
    augmented_input = augmentations(inputimg)
    augmented_target = augmentations(targetimg)
    
    return augmented_input, augmented_target

# Absolute paths to the folders
images_folder = '/home/labadmin/dev/imagesegment/images'
trimaps_folder = '/home/labadmin/dev/imagesegment/trimaps_colored'
output_folder = '/home/labadmin/dev/imagesegment/outputimages'

# Instantiate the dataset
dataset = CustomDataset(images_folder, trimaps_folder, transform=transform)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Instantiate model, optimizer, and loss
model = NCA(input_channels=3, state_channels=64, hidden_channels=64, output_channels=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion = nn.MSELoss()

# Train the Model
for epoch in range(30):
    model.train()
    for i, (data, target) in enumerate(train_loader):
        # Apply robust training augmentations to both image and label
        augmented_data, augmented_target = zip(*[robusttraining(img, lbl) for img, lbl in zip(data, target)])
        
        # Stack the augmented images and labels to create a batch
        augmented_data = torch.stack(augmented_data)
        augmented_target = torch.stack(augmented_target)
        
        # Move data and target to device
        augmented_data, augmented_target = augmented_data.to(device), augmented_target.to(device)
        
        optimizer.zero_grad()
        output = model.forward(augmented_data)
        loss = criterion(output, augmented_target)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Iteration {i}, Loss: {loss.item()}")


# Save the model weights after training
torch.save(model.state_dict(), 'model_weights_segmenting_robusttrainingsegmenting.pth')
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

