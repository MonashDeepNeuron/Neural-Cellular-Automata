import numpy as np
import torch
import torch.nn as nn

from gca_model import GCA


# Function to save the model weights as binary
def save_weights(model):
    model_state = model.state_dict()  # Get model parameters

    with open("model_weights.bin", "wb") as f:
        for name, param in model_state.items():
            weight = param.cpu().numpy()  # Convert to NumPy array
            weight = weight.astype(np.float32)  # Ensure it's in float32 format
            weight.tofile(f)  # Save the weights as binary
            print(weight)  # Print the NumPy array values

    # Optionally print out the layers for verification
    for name, param in model.state_dict().items():
        print(f"Layer name: {name}, Shape: {param.shape}")


if __name__ == "__main__":
    # Initialize the model
    model = GCA()

    # Load weights from .pth file
    try:
        model.load_state_dict(torch.load("model_weights.pth"))
        print("Loaded model weights successfully!")
    except FileNotFoundError:
        print("No previous model weights found.")

    # Save the model weights as binary
    save_weights(model)
