import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from collections import namedtuple
# Randomly generate list of n (x,y) coordinates (cells)

lst = list(np.random.rand(200, 3))

RADIUS = 0.2 # within 2 units of distance  

def cell_neighbourhood(cell_list, cell_idx):
    """
    Takes a list of cell location and the index of the cell of interest and returns a list of all the cells 
    that are within the specified radius. This deterimines cell-cell contacts.
    """
    cell = cell_list.pop(cell_idx)
    ## Naive solution: iterate through and find all cells within the radius
    combine = lambda other : [ (other[i] - cell[i]) ** 2 for i in range(len(cell)) ]    # [(x1-x2)^2, (y1-y2)^2, (z1-z2)^2]
    distance = lambda other: np.sqrt(sum(combine(other))) # sqrt([(x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2]) i.e. distance formula
    am_close = lambda other : distance(other) < RADIUS
    return list(filter(am_close, cell_list))
    # return cell_list

# def aggregate

def plot_grid(cell_list, connections):
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(cell_list[:, 0], cell_list[:, 1], cell_list[:, 2])
    for i in range(len(cell_list)):
        for j in range(len(connections[i])):
            ax.plot((cell_list[i][0], connections[i][j][0]), (cell_list[i][1], connections[i][j][1]), (cell_list[i][2], connections[i][j][2]), 'r')
        # For cell in list, plot a line to each connection
    plt.show()

if __name__ == "__main__":
    # print(f"len: {len(lst)}")
    # print(f"cell_neighbourhood(lst, 0): {cell_neighbourhood(lst, 0)}")
    # print(f"len(lst, 0): {len(cell_neighbourhood(lst, 0))}")

    cells = np.array(lst.copy())
    connections = []

    for i in range(len(lst)): # TODO:Move this out to a new function
        # Note: lst gets 1 shorter with each run of this loop
        connections.append(cell_neighbourhood(lst, 0))


    plot_grid(cells, connections)