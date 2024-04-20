@group(0) @binding(0) var<uniform> grid: vec2<f32>;
@group(0) @binding(1) var<storage> cellStateIn: array<f32>; // 
@group(0) @binding(2) var<storage, read_write> cellStateOut: array<f32>; //
@group(0) @binding(3) var<storage> weights : array<i32>;
override WORKGROUP_SIZE: u32 = 16;
// cellIndex[i] gives us the i+1th channel
// Hashes the x and y value for a cell to its position in 1D cell array (1st value in 16 cell block) 
// Wraps
fn cellIndex(cell: vec2<u32>) -> u32 {
    return (cell.y % u32(grid.y)) * u32(grid.x) + (cell.x % u32(grid.x));
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

fn computePerceptionVector(x: u32, y: u32) -> array<f32, 48> {
    var perceptionVector: array<f32, 48>;

    // For each channel calculate sobelX, sobelY and identity, and add to perception vector
    for (var j: i32 = 0; j <= 15; j++) {
    let sobelX = calculateSobelX(x, y);
    let sobelY = calculateSobelY(x, y);
    let identity = cellValue(x, y); // For simplicity, using the cell value as identity
    perceptionVector[3*j] = sobelX;
    perceptionVector[3*j+1] = sobelY;
    perceptionVector[3*j+2] = identity;
}
}

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE)
fn computeMain(@builtin(global_invocation_id) cell: vec3<u32>) {
    let i = cellIndex(cell.xy);
    let perceptionVector = computePerceptionVector(cell.x, cell.y);
    let weightedSum = perceptionVector[0] * rule[0] + perceptionVector[1] * rule[1] + perceptionVector[2] * rule[2];
    cellStateOut[i] = applyActivation(weightedSum);
}

// weights = should be 272 numbers
// perception vec
/* CONSTRUCTING THE FCN COMPONENT

    for j = 0; j <= 48; j++;
        val = perception_vec[j]
        for i=0; i<= 128; i++;
            val2 = val*weights[i]+

            layer 2




*/