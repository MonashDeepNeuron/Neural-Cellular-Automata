@group(0) @binding(0) var<uniform> grid: vec2<f32>; // uniform variable - data does not change across individual shader invocations
// - passing dimensions of canvas size
@group(0) @binding(1) var<storage> cellStateIn: array<f32>;
@group(0) @binding(2) var<storage, read_write> cellStateOut: array<f32>; 
@group(0) @binding(3) var<storage> w1 : array<f32>;
@group(0) @binding(3) var<storage> w2 : array<f32>;
@group(0) @binding(3) var<storage> b1 : array<f32>;
override WORKGROUP_SIZE: u32 = 16;
// cellIndex[i] gives us the i+1th channel
// Hashes the x and y value for a cell to its position in 1D cell array (1st value in 16 cell block) 
fn cellIndex(cell: vec2<u32>) -> u32 {
    return ((cell.y % u32(grid.y)) * u32(grid.x) + (cell.x % u32(grid.x)))*16;
}

fn cellValue(x: u32, y: u32) -> f32 {
    return cellStateIn[cellIndex(vec2(x, y))];
}

fn calculateSobelX(x: u32, y: u32) -> f32 {
    let sobelFilterX: array<f32, 9> = array<f32, 9>(-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0);
    var sum: f32 = 0.0;
    for (let j: i32 = -1; j <= 1; j++) {
        for (let i: i32 = -1; i <= 1; i++) {
            sum += cellValue(x + u32(i), y + u32(j)) * sobelFilterX[(j + 1) * 3 + (i + 1)];
        }
    }
    return sum;
}

fn calculateSobelY(x: u32, y: u32) -> f32 {
    let sobelFilterY: array<f32, 9> = array<f32, 9>(-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0);
    var sum: f32 = 0.0;
    for (let j: i32 = -1; j <= 1; j++) {
        for (let i: i32 = -1; i <= 1; i++) {
            sum += cellValue(x + u32(i), y + u32(j)) * sobelFilterY[(j + 1) * 3 + (i + 1)];
        }
    }
    return sum;
}

float relu(float x) {
    return max(0.0, x);
}

fn computePerceptionVector(x: u32, y: u32) -> array<f32, 48> {
    var perceptionVector: array<f32, 48>; // x y identity (16 times)

    // For each channel calculate sobelX, sobelY and identity, and add to perception vector
    for (var j: i32 = 0; j <= 15; j++) {
        let sobelX = calculateSobelX(x, y);
        let sobelY = calculateSobelY(x, y);
        let identity = cellValue(x, y);
        
        perceptionVector[3*j] = sobelX;
        perceptionVector[3*j+1] = sobelY;
        perceptionVector[3*j+2] = identity;
    }
    return perceptionVector;
}

fn computeLinearLayers(perceptionVector array<f32, 48>) -> array<f32, 16> {
    /*
        w1 = second linear layer weight vector 128 
        b1 = second linear layer bias vector 128 
        w2 = second linear layer weight vector 16
    */
    
    // Compute first linear layer   
    let float h1[128]; // output of first linear layer
    for (int i = 0; i < 128; ++i) { // iterate through each layer in cell 
        for (int j = 0; j < 48; ++j) { // for each perceptron in 1st linear layer
            h1[i] += w1[i * 48 + j] * perceptionVector[j];
        }
        h1[i] = h1[i] + b1[i];
        h1[i] = relu(h1[i]);
    }
    
    // Compute second linear layer   
    float h2[16]; // output of second linear layer
    for (int i = 0; i < 16; ++i) { // for each weight in the second linear layer
        for (int j = 0; j < 128; ++j) { // for each perceptron in h1
            h2[i] += w2[i * 128 + j] * h1[j];
        }
    }
    return h2;

}

// Main function invoked by main thread (executed by each workgroup)
@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE)
fn computeMain(@builtin(global_invocation_id) cell: vec3<u32>) {
    let offset = cellIndex(cell.xy);
    let perceptionVector = computePerceptionVector(cell.x, cell.y);
    let output = computeLinearLayers(perceptionVector);
    for (int i = 0; i <16; ++i){
        cellStateOut[i+offset] = output[i];
    }
}