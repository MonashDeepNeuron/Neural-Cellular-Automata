import numpy as np
class NeuralNet:
    def __init__(self, weights1, bias, weights2, grid_x, grid_y):
        self.w1 = weights1
        self.b1 = bias
        self.w2 = weights2

        # Grid
        self.grid_x = grid_x
        self.grid_y = grid_y
        

        self.sobel_x = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        self.sobel_y = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        

    '''
    Functions to map x y coordinates from a 2D grid_to the index and value
    in the grid_x * grid_y * 16 vector 
    '''
    def cellIndex(self, x, y):
        return (y % (self.grid_y)) * (self.grid_x) + (x % (self.grid.x))*16
    
    def cellValue(self, x, y, cellStateIn):
        return cellStateIn[self.cellIndex(x, y)]
    
    def calculateSobelX(self, x, y):
        sum = 0
        for i in range([-1, 0, 1]):
            for j in range([-1, 0, 1]):
                sum += self.cellValue(x + i, y + j)*self.sobel_x[(j+1)*3+(i+1)]
        return sum


    def calculateSobelY(self, x, y):
        sum = 0
        for i in range([-1, 0, 1]):
            for j in range([-1, 0, 1]):
                sum += self.cellValue(x + i, y + j)*self.sobel_y[(j+1)*3+(i+1)]
        return sum

    def relu(x):
        return max(0,x)

    def computePerceptionVector(self, x, y):
        perceptionVector = [0*48]
        for j in range(0, 16):
            sobelX = self.calculateSobelX(x, y)
            sobelY = self.calculateSobelY(x, y)
            identity = self.cellValue(x, y)

            perceptionVector[j] = sobelX
            perceptionVector[j + 16] = sobelY
            perceptionVector[j + 32] = identity
        return perceptionVector

    def computeLinearLayers(self, perceptionVector):
        '''
            w1 = second linear layer weight vector 128 * 48 
            b1 = second linear layer bias vector 128 
            w2 = second linear layer weight vector 128 * 16
        '''
        h1 = [0*128] # output of first linear layer
        h2 = [0*12] # output of second linear
        for j in range(0, 128):
            for i in range(0,48):
                h1[j] += self.w1[j*48 + i] * perceptionVector[i]
            h1[j] = h1[j] + self.b1[j]
            h1[j] = self.relu(h1[j])


        for j in range(0, 128):
            for i in range(0,48):
                h1[j] += self.w1[j*48 + i] * perceptionVector[i]

        h1[j] = h1[j] + self.b1[j]
        h1[j] = self.relu(h1[j])



    // Compute second linear layer

    for (var j: u32 = 0u; j < 16u; j = j + 1u) { // for each weight in the second linear layer
        for (var i: u32 = 0u; i < 128u; i = i + 1u) { // for each perceptron in h1
            h2[j] += w2[j * 128u + i] * h1[i];
        }
    }

    /* For using the cell storage array buffer to write out the contents of the perception vector*/

    // var subsetPerceptionVector = array<f32, 16>();

    // for (var i: u32 = 0u; i < 16u; i = i + 1u) {
    //     subsetPerceptionVector[i] = perceptionVector[i];
    // }

    // return subsetPerceptionVector;
    return h2;
}
    def update_value(self, grid, x, y):
        # Copy array values

        # Input array should be gridsize * gridsize * 16, flattened

        # Apply sobel x

        # Apply sobel y

        # Concat original, sobel x, sobel y

        # Pass through neural net 

        return # Return result




