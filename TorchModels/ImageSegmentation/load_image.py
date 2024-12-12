import os
import torch
from torchvision.transforms import ToTensor, Compose, Resize
from torch.utils.data import Dataset
from model import NCA
from PIL import Image
import numpy as np
import random

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, images_folder, transform=None):
        self.images_folder = images_folder
        self.transform = transform
        self.images = sorted(os.listdir(images_folder))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_folder, self.images[idx])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image

# Transformations
transform = Compose([
    Resize((128, 128)),  # Resize images to 128x128
    ToTensor()
])

# Absolute paths to the folders
images_folder = '/home/labadmin/dev/imagesegment/images'
output_folder = '/home/labadmin/dev/imagesegment/outputimages'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Instantiate the dataset
dataset = CustomDataset(images_folder, transform=transform)

# Load the model
model = NCA(input_channels=3, state_channels=64, hidden_channels=64, output_channels=3).to(device)
model.load_state_dict(torch.load('/home/labadmin/dev/imagesegment/higher_res_robust_using350350_20epochs.pth'))

model.eval()

# Select 10 random images
random_indices = random.sample(range(len(dataset)), 10)
random_images = [dataset[i] for i in random_indices]

# Apply the model to the images and display them
with torch.no_grad():
    for i, image in enumerate(random_images):
        image = image.unsqueeze(0).to(device)  # Add batch dimension and move to device
        output = model(image)
        output_classes = torch.argmax(output, dim=1).squeeze(0)  # Get the predicted classes

        # Convert tensors to numpy arrays
        input_image_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output_classes_np = output_classes.cpu().numpy()

        # Create an empty mask
        mask = np.zeros((output_classes_np.shape[0], output_classes_np.shape[1], 3), dtype=np.uint8)
        
        # Define the color for the dog class (assuming class 1 is the dog)
        dog_class = 1
        mask[output_classes_np == dog_class] = [0, 255, 0]  # Green color for the dog class

        # Overlay the mask on the original image
        overlayed_image_np = input_image_np.copy()
        overlayed_image_np[mask[:, :, 1] == 255] = [0, 255, 0]  # Apply green mask

        # Convert numpy arrays to PIL images
        overlayed_image_pil = Image.fromarray((overlayed_image_np * 255).astype(np.uint8))

        # Save the images
        overlayed_image_pil.save(os.path.join(output_folder, f'overlayed_image_{i}.png'))

print("Output images saved.")
