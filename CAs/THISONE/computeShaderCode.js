

export const computeShaderCode =
/*wgsl*/`
@group(0) @binding(0) var<uniform> grid: vec2<f32>; // uniform variable - data does not change across individual shader invocations
// - passing dimensions of canvas size
@group(0) @binding(1) var<storage> cellStateIn: array<f32>;
@group(0) @binding(2) var<storage, read_write> cellStateOut: array<f32>; 
@group(0) @binding(3) var<storage> w1 : array<f32>;
@group(0) @binding(4) var<storage> w2 : array<f32>;
@group(0) @binding(5) var<storage> b1 : array<f32>;
const WORKGROUP_SIZE: u32 = 16;
// cellIndex[i] gives us the i+1th channel
// Hashes the x and y value for a cell to its position in 1D cell array (1st value in 16 cell block) 
fn cellIndex(cell: vec2<u32>) -> u32 {
    return ((cell.y % u32(grid.y)) * u32(grid.x) + (cell.x % u32(grid.x)))*16;
}

fn cellValue(x: u32, y: u32, channel: u32) -> f32 {
    // NOTE: 0 <= channel < 16
    return cellStateIn[cellIndex(vec2(x, y))+ channel];
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
            sum += cellValue(u32(x + i), u32(y + j), u32(channel)) * sobelFilterX[(j + 1) * 3 + (i + 1)];
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

fn computePerceptionVector(x: i32, y: i32) -> array<f32, 48> {

    // our py code: filters = torch.stack([identity, sobel_x, sobel_y]).repeat((n_channels, 1, 1))
    // paper py code: perception_grid = concat(state_grid, grad_x, grad_y, axis=2)

    var perceptionVector: array<f32, 48> = array<f32, 48>(); // x y identity (16 times)

    // For each channel calculate sobelX, sobelY and identity, and add to perception vector
    for (var j: i32= 0; j <= 15; j = j + 1) {
        let sobelX = calculateSobelX(x, y, j);
        let sobelY = calculateSobelY(x, y, j);
        
        // As implemented in the paper
        // perceptionVector[j] = cellValue(u32(x), u32(y), u32(j));
        // perceptionVector[16+j] = sobelX;
        // perceptionVector[32+j] = sobelY;

        // As implemented in our version
        perceptionVector[3*j] = cellValue(u32(x), u32(y), u32(j));
        perceptionVector[3*j+1] = sobelX;
        perceptionVector[3*j+2] = sobelY;
    }
    return perceptionVector;
}


fn surroundingLifeFilter(inputVector: array<f32, 16>, x: i32, y: i32) -> array<f32, 16> {
    /*
        A cell is inherently dead if there is no cells in its 3x3 
        neighbourhood that are alive (channel 4 > 0.1) (aplha = index 3), return 
        empty state vector as new state value
    */

    for (var i: i32 = -1; i <= 1; i++) {
        for (var j: i32 = -1; j <= 1; j++) {
            if (cellValue(u32(x + i), u32(y + j), 3u) > 0.1) {
                return inputVector;
            }
        }
    }

    return array<f32, 16>(); // Return empty
}



fn computeLinearLayers(perceptionVector: array<f32, 48>, x: i32, y: i32) -> array<f32, 16> {
    /*
        
        w1 = second linear layer weight vector 128 
        b1 = second linear layer bias vector 128 
        w2 = second linear layer weight vector 16

        48 vec -> fully interconnected linear (w1) -> 128 vec -> add bias (b1)
        -> 128 vec -> fully interconnected linear (w2) -> 16 vec (output)

    */
    
    var h1 = array<f32, 128>(); // output first layer
    var h2 = array<f32, 16>(); // output second layer
    
    // var x = array<f32, 16>();

    // output of first linear layer
    for (var i: u32 = 0u; i < 128u; i = i + 1u) { // iterate through each layer in cell 
        for (var j: u32 = 0u; j < 48u; j = j + 1u) { // for each perceptron in 1st linear layer
            h1[i] += w1[i * 48u + j] * perceptionVector[j];
        }
        h1[i] = h1[i] + b1[i];
        h1[i] = relu(h1[i]);
    }
    
    // Compute second linear layer   
    for (var i: u32 = 0u; i < 16u; i = i + 1u) { // for each weight in the second linear layer
        for (var j: u32 = 0u; j < 128u; j = j + 1u) { // for each perceptron in h1 // TODO: make j < 128u
            h2[i] += w2[i* 128u  + j] * h1[j];
        }
    }

    let output = surroundingLifeFilter(h2, x, y);

    return output;

}

// Main function invoked by main thread (executed by each workgroup)
@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE)
fn computeMain(@builtin(global_invocation_id) cell: vec3<u32>) {
    let offset = cellIndex(cell.xy);
    let perceptionVector = computePerceptionVector(i32(cell.x), i32(cell.y));
    let output = computeLinearLayers(perceptionVector, i32(cell.x), i32(cell.y));
    for (var i: u32 = 0u; i < 16u; i = i + 1u){
        cellStateOut[i+offset] = output[i];
    }

}
`