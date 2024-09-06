# from neuralNode import NeuralNet
from model import GrowingCA
from matplotlib import pyplot as plt
import torch
import numpy as np
import torch


GRID_SIZE = 32
VECTOR_LENGTH = 16

def show_image(input, channels = (0,1,2,3)):

    image = [[[0 for i in range(len(channels))] for j in range(GRID_SIZE)] for k in range(GRID_SIZE)]
    for i in range(GRID_SIZE): 
        for j in range(GRID_SIZE): 
            for k in range(len(channels)):
                image[i][j][k]= input[i*GRID_SIZE*VECTOR_LENGTH + j* VECTOR_LENGTH + channels[k]]
                if (image[i][j][k] > 1):
                    image[i][j][k]=1
                elif (image[i][j][k] < 0):
                    image[i][j][k] = 0

    # End organising channels
    
    plt.imshow(image)
    plt.show()

def load_model(model_class, filepath="model_weights.pth", device="cpu"):
    model = model_class().to(device)
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.eval()  # Set the model to evaluation mode
    print(f"Model weights loaded from {filepath}")
    return model
   
def update_by_nn_model(model, grid):
    grid = model(grid)
    return grid


if (__name__ == "__main__"):

    # Extract weights and biases from binary file

    model_path = "GrowingCADebug/20_8_model_weights.pt"
    model = load_model(GrowingCA, model_path)
    model.eval()
    print(model)

    modelParams = model.state_dict()
    print(len(modelParams))


    weights1 = modelParams["update_step.0.weight"]
    bias = modelParams["update_step.0.bias"]
    weights2 = modelParams["update_step.2.weight"]

    # net = NeuralNet(weights1, bias, weights2)

    grid = [0.0]* (GRID_SIZE* GRID_SIZE* VECTOR_LENGTH)
    
    # grid[int(GRID_SIZE/2)*GRID_SIZE*VECTOR_LENGTH + int(GRID_SIZE/2)*VECTOR_LENGTH + 3] = 1.0

    grid = torch.Tensor(np.array(grid)).reshape(1, -1, GRID_SIZE, GRID_SIZE)
    grid[:, 3:, GRID_SIZE//2, GRID_SIZE//2] = 1

    for i in range(1):
        grid = update_by_nn_model(model, grid)
    show_image(grid.detach().flatten())



    
