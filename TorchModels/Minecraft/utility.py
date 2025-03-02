import torch
from torch import Tensor
import torch.nn as nn
from model import NCA_3D
import random
import matplotlib.pyplot as plt
import numpy as np
import trimesh
import os
import matplotlib.animation as animation

def load_object_as_voxelgrid(objPath: str):
    '''

        This utility function is used to return a VoxelGrid object representing
        an input object/mesh.
    
    '''

    mesh = trimesh.load(objPath, force = "mesh")    
    if mesh.is_empty:
        raise ValueError("The mesh is empty and cannot be voxelized.")
    else:
        print("Mesh loaded successfully with vertices:", mesh.vertices.shape)
        print("Mesh loaded successfully with faces:", mesh.faces.shape)

    # generate VoxelGrid object 
        # Parameters
        # -----------
        # mesh : trimesh.Trimesh
        # Source geometry
        # point : (3, ) float
        # Point in space to voxelize around
        # pitch :  float
        # Side length of a single voxel cube
        # radius : int
        # Number of voxel cubes to return in each direction.
        # kwargs : parameters to pass to voxelize_subdivide

        # Returns
        # -----------
        # voxels : VoxelGrid instance with resolution (m, m, m) where m=2*radius+1
    voxel = trimesh.voxel.creation.local_voxelize(mesh,mesh.centroid, mesh.extents.max() / 16, 8, fill=True)

    # Get occupancy matrix of dimennsions X, Y, Z where True represents an occupied Voxel
    voxel_matrix = voxel.matrix

    # Use occupancy matrix to create X, Y, Z, 4 matrix holding colour information
    rgba_matrix = np.zeros(voxel_matrix.shape + (4,), dtype=int)
    # TODO: Encode colour using voxel information, this just sets all occupied voxels to a RGB = 0.5 and Alpha = 1
    rgba_matrix[voxel_matrix] = [0.5, 0.5, 0.5, 1]

    # # Use occupancy matrix to create X, Y, Z, 1 matrix holding colour information
    # a_matrix = np.zeros(voxel_matrix.shape + (1,), dtype=int)
    # # TODO: Encode colour using voxel information, this just sets all occupied voxels to a RGB = 0.5 and Alpha = 1
    # a_matrix[voxel_matrix] = 1

    voxel_tensor = torch.tensor(rgba_matrix, dtype=torch.float32)

    return voxel, voxel_tensor

def new_seed(target_vtensor, BATCH_SIZE=1, CHANNELS=4):
    """
        seed is like a cube map that sets a singular pixel activated
        CHANNELS should always be RGBA (+ more), Alpha should always be the 4th 
        
        Returns: (BATCH_SIZE, CHANNELS, X, Y, Z)
    """
    SHAPE = [target_vtensor.shape[i] for i in range(len(target_vtensor.shape))]
    seed = torch.zeros(BATCH_SIZE, CHANNELS, SHAPE[0], SHAPE[1], SHAPE[2])
    
    # (BATCH_SIZE, CHANNELS, X, Y, Z)
    seed[:, :4, SHAPE[0]//2, SHAPE[1]//2, SHAPE[2]//2] = 1  
    return seed
