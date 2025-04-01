import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
# Randomly generate list of n (x,y) coordinates (cells)
lst = list(np.random.rand(15, 3))
RADIUS = 0.1 # within 2 units of distance  


def cell_neighbourhood(cell_list: list[list[int]], cell_idx: int):
    cell = cell_list[cell_idx]
    ## Naive solution: iterate through and find all cells within the radius
    combine = lambda other : [ (other[i] - cell[i]) ** 2 for i in range(len(cell)) ]    
    distance = lambda other: np.sqrt(sum(combine(other)))
    am_close = lambda other : distance(other) < RADIUS
    return list(filter(am_close, cell_list))
    # return cell_list

# def aggregate

def plot_grid(cell_list):
    
    

    plt.scatter(cell_list[:, 0], cell_list[:, 1], cell_list[:, 2])
    plt.show()

    pass

    # #   Find the min and max of the x and y locations to create the adequate plotting space
    # x_min = cell_list[:, 0].min(axis=0)
    # x_max = cell_list[:, 0].max(axis=0)
    # y_min = cell_list[:, 1].min(axis=0)
    # y_max = cell_list[:, 1].max(axis=0)
    # z_min = cell_list[:, 2].min(axis=0)
    # z_max = cell_list[:, 2].max(axis=0)

    # buffer = 5 # Give some space around the furthest cell

    # x = np.arange(x_min -buffer, x_max + buffer)
    # y = np.arange(y_min -buffer, y_max + buffer)
    # z = np.arange(z_min -buffer, z_max + buffer)
    
    # X, Y, Z = np.meshgrid(x, y, z)
    # # Make the cells at each location 1, all the others remain 0
   
    # cell_space = np.zeros(len(x), len(y), len(z))
    
    # for cell in cell_list:
    #     cell_space()

    # plt.pcolormesh()

    # return None



if __name__ == "__main__":
    print(f"lst: {lst}")

    print(f"len: {len(lst)}")
    print(f"cell_neighbourhood(lst, 0): {cell_neighbourhood(lst, 0)}")
    print(f"len(lst, 0): {len(cell_neighbourhood(lst, 0))}")