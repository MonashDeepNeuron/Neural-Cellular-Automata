const simulation = /* wgsl */ `
struct ParameterShape {
  channels: u32,
  convolutions: u32,
  hidden_channels: u32,
  size: u32
};

@group(0) @binding(0) var<uniform> shape: ParameterShape;
@group(0) @binding(1) var<storage> state: array<f32>;
@group(0) @binding(2) var<storage, read_write> next_state: array<f32>;
@group(0) @binding(3) var<storage> l1_w: array<f32>;
@group(0) @binding(4) var<storage> l1_b: array<f32>;
@group(0) @binding(5) var<storage> l2_w: array<f32>;
@group(0) @binding(6) var<uniform> seed: u32;

// Constants
const CHANNELS = 12;
const CONVOLUTIONS = 4;
const HIDDEN_CHANNELS = 96;
const PERCEPTION_VECTOR = CHANNELS * CONVOLUTIONS;

// Perception kernels
const SOBEL_X: mat3x3f = mat3x3f(
    -1.0, 0.0, 1.0,
    -2.0, 0.0, 2.0,
    -1.0, 0.0, 1.0
);

const SOBEL_Y: mat3x3f = mat3x3f(
    -1.0, -2.0, -1.0,
     0.0,  0.0,  0.0,
     1.0,  2.0,  1.0
);

const LAPLACIAN: mat3x3f = mat3x3f(
     1.0,  2.0,  1.0,
     2.0, -12.0, 2.0,
     1.0,  2.0,  1.0
);

@workgroup_size(8, 8)
@compute
fn compute_main(@builtin(global_invocation_id) pos: vec3<u32>) {
    let size = shape.size;

    // Compute grid position
    let x = pos.x;
    let y = pos.y;

    // Ensure valid grid bounds
    if (x >= size || y >= size) {
        return;
    }

    // Perception convolution
    var perception_out: array<f32, PERCEPTION_VECTOR>;

    // Copy identity convolution directly from state
    for (var c = 0u; c < CHANNELS; c++) {
        perception_out[c * 4 + 0] = state[index(vec3u(c, x, y), size, CHANNELS)];
    }

    // Compute convolutions (SOBEL_X, SOBEL_Y, LAPLACIAN)
    for (var c = 0u; c < CHANNELS; c++) {
        perception_out[c * 4 + 1] = convolve(vec2u(x, y), SOBEL_X, c);
        perception_out[c * 4 + 2] = convolve(vec2u(x, y), SOBEL_Y, c);
        perception_out[c * 4 + 3] = convolve(vec2u(x, y), LAPLACIAN, c);
    }

    // Fully connected layers
    var hidden_out: array<f32, HIDDEN_CHANNELS>;  // Hidden layer output
    for (var h = 0u; h < HIDDEN_CHANNELS; h++) {
        var sum: f32 = 0.0;
        for (var p = 0u; p < PERCEPTION_VECTOR; p++) {
            sum += l1_w[h * PERCEPTION_VECTOR + p] * perception_out[p];
        }
        hidden_out[h] = ReLU(sum + l1_b[h]);
    }

    // Output layer (next state computation)
    for (var c = 0u; c < CHANNELS; c++) {
        var sum: f32 = 0.0;
        for (var h = 0u; h < HIDDEN_CHANNELS; h++) {
            sum += l2_w[c * HIDDEN_CHANNELS + h] * hidden_out[h];
        }

        let mask = rand(vec3u(c, x, y));
        next_state[index(vec3u(c, x, y), size, CHANNELS)] = state[index(vec3u(c, x, y), size, CHANNELS)] + sum * mask;
    }
}

// Helper function to perform convolution with a 3x3 kernel for a specific channel
fn convolve(coord: vec2u, kernel: mat3x3f, channel: u32) -> f32 {
    var sum: f32 = 0.0;
    let size = shape.size;

    for (var ky = 0u; ky < 3u; ky++) {
        for (var kx = 0u; kx < 3u; kx++) {
            let x = (coord.x + kx - 1 + size) % size; // Circular padding
            let y = (coord.y + ky - 1 + size) % size;
            let state_idx = index(vec3u(channel, x, y), size, CHANNELS);
            sum += state[state_idx] * kernel[ky][kx];
        }
    }

    return sum;
}

// Helper function to compute 1D index for a 3D grid (channels, size, size)
fn index(coord: vec3u, size: u32, channels: u32) -> u32 {
    return (coord.x * size * size) + (coord.y * size) + coord.z;
}

// Random value
fn rand(coord: vec3u) -> f32 {
    var hash: u32 = (coord.x * 73856093u) ^ (coord.y * 19349663u) ^ (coord.z * 83492791u) ^ seed;
    hash = (hash >> 13) ^ hash;
    return f32((hash * (hash * hash * 15731u + 789221u) + 1376312589u) & 0x7fffffff) / f32(0x7fffffff);
}

// ReLU activation function
fn ReLU(x: f32) -> f32 {
    return max(0.0, x);
}
`;

export default simulation;
