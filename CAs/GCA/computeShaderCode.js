export const computeShaderCode =
/*wgsl*/`
@group(0) @binding(0) var<uniform> grid: vec2<f32>; 
@group(0) @binding(1) var<storage> cellStateIn: array<f32>;
@group(0) @binding(2) var<storage, read_write> cellStateOut: array<f32>; 
@group(0) @binding(3) var<storage> w1 : array<f32>;
@group(0) @binding(4) var<storage> w2 : array<f32>;
@group(0) @binding(5) var<storage> b1 : array<f32>;

const WORKGROUP_SIZE: u32 = 16;

/* Converts x y to scalar value. Represents first index of sixteen channels*/
fn cellIndex(cell: vec2<u32>) -> u32 {
    return ((cell.y % u32(grid.y)) * u32(grid.x) + (cell.x % u32(grid.x)))*16;
}

/* Returns index of channel of given cell*/
fn cellValue(x: u32, y: u32, channel: u32) -> f32 {
    return cellStateIn[cellIndex(vec2(x, y))+ channel];
}


/* Converts x y to scalar value. Represents first index of sixteen channels
Takes x, y, channel --> k (indexes cellstatein column vector) */
fn getK(x: u32, y: u32, channel: u32) -> u32 {
    return (y * u32(grid.x)* 16 + x * 16 + channel);
}


/* Returns value of cellStateIn at k*/
fn getValueOfCell(x: u32, y: u32, channel: u32) -> f32{
    return cellStateIn[getK(x,y,channel)];
}





fn calculateSobelX(x: i32, y: i32, channel: i32) -> f32 {
    // NOTE: 0 <= channel < 16

    // Find the gradient by applying an X-oriented sobel filter 
    // -1 | 0 |  1
    // -2 | 0 |  2
    // -1 | 0 |  1



    let sobelFilterX: array<f32, 9> = array<f32, 9>(-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0);
    var sum: f32 = 0.0;
    for (var j: i32 = -1; j <= 1; j = j + 1) {
        for (var i: i32 = -1; i <= 1; i = i + 1) {
            let cell_value = getValueOfCell(u32(y+j),u32(x+i), u32(channel));

            sum += cell_value * sobelFilterX[(j + 1) * 3 + (i + 1)];
            
            // sum += cellValue(u32(x + i), u32(y + j), u32(channel)) * sobelFilterX[(j + 1) * 3 + (i + 1)];
        }
    }
    return sum;
}

fn calculateSobelY(x: i32, y: i32, channel: i32) -> f32 {
    // NOTE: 0 <= channel < 16

    // Find the gradient by applying an Y-oriented sobel filter 
    // -1 | -2 | -1
    //  0 |  0 |  0
    //  1 |  2 |  1

    // This function only works on one of the 16 channels at a 
    // time. 

    let sobelFilterY: array<f32, 9> = array<f32, 9>(-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0);
    var sum: f32 = 0.0;
    for (var j: i32 = -1; j <= 1; j = j + 1) {
        for (var i: i32 = -1; i <= 1; i = i + 1) {
            sum += cellValue(u32(x + i), u32(y + j), u32(channel)) * sobelFilterY[(j + 1) * 3 + (i + 1)];
        }
    }
    return sum;
}

fn relu(x: f32) -> f32 {
    if (x > 0) {
        return x;
    }
    return 0.0;
}

/* perception vector with contiguous identity + sobelx + sobel y */
fn computePerceptionVector(x: i32, y: i32) -> array<f32, 48> {
    var perceptionVector: array<f32, 48> = array<f32, 48>(); // x y identity (16 times)

    // For each channel, first store the identity (original state)
    for (var j: i32 = 0; j <= 15; j = j + 1) {
        perceptionVector[j] = cellValue(u32(x), u32(y), u32(j)); // Identity (original state)
    }

    // Next, store the Sobel X values for each channel
    for (var j: i32 = 0; j <= 15; j = j + 1) {
        perceptionVector[16 + j] = calculateSobelX(x, y, j); // Sobel X gradient
    }

    // Finally, store the Sobel Y values for each channel
    for (var j: i32 = 0; j <= 15; j = j + 1) {
        perceptionVector[32 + j] = calculateSobelY(x, y, j); // Sobel Y gradient
    }

    return perceptionVector;
}


/* perception vector with interleaved identity + sobelx + sobel y */
// fn computePerceptionVector(x: i32, y: i32) -> array<f32, 48> {

//     // our py code: filters = torch.stack([identity, sobel_x, sobel_y]).repeat((n_channels, 1, 1))
//     // paper py code: perception_grid = concat(state_grid, grad_x, grad_y, axis=2)

//     var perceptionVector: array<f32, 48> = array<f32, 48>(); // x y identity (16 times)

//     // For each channel calculate sobelX, sobelY and identity, and add to perception vector
//     for (var j: i32= 0; j <= 15; j = j + 1) {
//         let sobelX = calculateSobelX(x, y, j);
//         let sobelY = calculateSobelY(x, y, j);
    

//         // As implemented in our version
//         perceptionVector[3*j] = cellValue(u32(x), u32(y), u32(j));
//         perceptionVector[3*j+1] = sobelX;
//         perceptionVector[3*j+2] = sobelY;
//     }
//     return perceptionVector;
// }

// fn applyStochasticMask(ds_grid: array<f32, 16>) -> array<f32, 16> {
//     let random_val = random();  // Generate random number
//     for (var i: u32 = 0u; i < 16u; i = i + 1u) {
//         if (random_val < 0.5) {
//             ds_grid[i] = 0.0;  // Zero out with 50% probability
//         }
//     }
//     return ds_grid;
// }

fn applyAliveMask(inputVector: array<f32, 16>, x: i32, y: i32) -> array<f32, 16> {
    /*
        A cell is inherently dead if there is no cells in its 3x3 
        neighbourhood that are alive (channel 4 > 0.1) (alpha = index 3), return 
        empty state vector as new state value
    */

    for (var i: i32 = -1; i <= 1; i++) {
        for (var j: i32 = -1; j <= 1; j++) {
            if (cellValue(u32(x + i), u32(y + j), 3u) > 0.1) {
                return inputVector;
            }
        }
    }

    return array<f32, 16>(); // TODO: CHECK IF THIS RETURNS EMPTY 16 VECTOR
}




fn computeLinearLayers(perceptionVector: array<f32, 48>, x: i32, y: i32) -> array<f32, 16> {
        /*
        
        w1 = second linear layer weight vector 128 
        b1 = second linear layer bias vector 128 
        w2 = second linear layer weight vector 16

        48 vec -> fully interconnected linear (w1) -> 128 vec -> add bias (b1)
        -> 128 vec -> fully interconnected linear (w2) -> 16 vec (output)

    */
    var h1 = array<f32, 128>(); // First linear layer output
    var h2 = array<f32, 16>();  // Second linear layer output
    
    // First linear layer (48 -> 128)
    for (var i: u32 = 0u; i < 128u; i = i + 1u) {
        for (var j: u32 = 0u; j < 48u; j = j + 1u) {
            h1[i] += w1[i * 48u + j] * perceptionVector[j];
        }
        h1[i] = relu(h1[i] + b1[i]);  // Apply ReLU and bias
    }
    
    // Second linear layer (128 -> 16)
    for (var i: u32 = 0u; i < 16u; i = i + 1u) {
        for (var j: u32 = 0u; j < 128u; j = j + 1u) {
            h2[i] += w2[i * 128u + j] * h1[j];
        }
    }
    return h2;
    // Apply stochastic mask to the output
    // return applyStochasticMask(h2);
}

/* Main function 
We apply alive masking to the cell itself, rather than just the ds_cell
*/
@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE)
fn computeMain(@builtin(global_invocation_id) cell: vec3<u32>) {
    let offset = cellIndex(cell.xy);

    // Compute the perception vector
    let perceptionVector = computePerceptionVector(i32(cell.x), i32(cell.y));

    // Apply the update using linear layers
    let output = computeLinearLayers(perceptionVector, i32(cell.x), i32(cell.y));

    // Calculate the final output (add the update to the current state)
    var finalState: array<f32, 16>;
    for (var i: u32 = 0u; i < 16u; i = i + 1u) {
        finalState[i] = cellStateIn[i + offset] + output[i]; // input_grid + filtered_ds_grid
    }

    // Apply the alive mask to the entire final state
    let finalOutput = applyAliveMask(finalState, i32(cell.x), i32(cell.y)); // output_filtered_grid

    // Write the final output to cellStateOut
    for (var i: u32 = 0u; i < 16u; i = i + 1u) {
        cellStateOut[i + offset] = finalOutput[i];
    }
}

/*
Single Cell Forward Pass
Operates on a cell-basis rather than grid-basis
We apply alive masking to the change not the cell 
*/
// @compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE)
// fn computeMain(@builtin(global_invocation_id) cell: vec3<u32>) {
//     let offset = cellIndex(cell.xy);
//     let perceptionVector = computePerceptionVector(i32(cell.x), i32(cell.y));
    
//     // Compute updated state and apply stochastic updates
//     let output = computeLinearLayers(perceptionVector, i32(cell.x), i32(cell.y));
    
//     /* Apply alive mask before writing the final output*/
//     let finalOutput = applyAliveMask(output, i32(cell.x), i32(cell.y));
    
//     /* Write final output to cellStateOut*/
//     for (var i: u32 = 0u; i < 16u; i = i + 1u) {
//         cellStateOut[i + offset] = cellStateIn[i+offset] + output[i];
//     }
// }
`