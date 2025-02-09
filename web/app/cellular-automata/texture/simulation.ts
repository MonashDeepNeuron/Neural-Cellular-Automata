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
fn compute_main(@builtin(global_invocation_id) pos: vec3u) {
  let size = shape.size;

  // Compute grid position
  let x = pos.x;
  let y = pos.y;

  // Ensure valid grid bounds
  if (x >= size || y >= size) {
    return;
  }

  // Perception convolution
  var perceptions: array<f32, PERCEPTION_VECTOR>;

  // Copy identity convolution directly from state
  for (var c = 0u; c < CHANNELS; c++) {
    perceptions[c * 4 + 0] = state[index(c, x, y)];
  }

  // Compute convolutions (SOBEL_X, SOBEL_Y, LAPLACIAN)
  for (var c = 0u; c < CHANNELS; c++) {
    perceptions[c * 4 + 1] = convolve(vec2u(x, y), SOBEL_X, c);
    perceptions[c * 4 + 2] = convolve(vec2u(x, y), SOBEL_Y, c);
    perceptions[c * 4 + 3] = convolve(vec2u(x, y), LAPLACIAN, c);
  }

  // Fully connected layers
  var hidden: array<f32, HIDDEN_CHANNELS>;
  for (var h = 0u; h < HIDDEN_CHANNELS; h++) {
    var sum: f32 = 0.0;
    for (var p = 0u; p < PERCEPTION_VECTOR; p++) {
      sum += l1_w[h * PERCEPTION_VECTOR + p] * perceptions[p];
    }
    // ReLU(x) = max(0, x)
    hidden[h] = max(0.0, sum + l1_b[h]);
  }

  // Output layer (next state computation)
  for (var c = 0u; c < CHANNELS; c++) {
    var sum: f32 = 0.0;
    for (var h = 0u; h < HIDDEN_CHANNELS; h++) {
      sum += l2_w[c * HIDDEN_CHANNELS + h] * hidden[h];
    }

    let i = index(c, x, y);
    next_state[i] = state[i] + sum * mask(i);
  }
}

// Helper function to perform convolution with a 3x3 kernel for a specific channel
fn convolve(coord: vec2u, kernel: mat3x3f, channel: u32) -> f32 {
  var sum: f32 = 0.0;
  let size = shape.size;

  for (var ky = 0u; ky < 3u; ky++) {
    for (var kx = 0u; kx < 3u; kx++) {
      // Find neighbours with circular padding
      let x = (coord.x + kx - 1 + size) % size;
      let y = (coord.y + ky - 1 + size) % size;
      let state_idx = index(channel, x, y);
      sum += state[state_idx] * kernel[ky][kx];
    }
  }

  return sum;
}

// Helper function to compute 1D index for a 3D grid (channels, size, size)
fn index(c: u32, x: u32, y: u32) -> u32 {
  return (c * shape.size * shape.size) + (x * shape.size) + y;
}

// Random 0 or 1
fn mask(index: u32) -> f32 {
  var hash: u32 = index ^ seed;
  hash = hash * 0x45d9f3bu;      // Prime multiplier
  hash = hash ^ (hash >> 13);    // Additional bit mixing
  hash = hash * 0x27d4eb2du;     // Another prime multiplier for variability
  return f32((hash >> 16) & 1u); // Extract 0 or 1 as f32
}
`;

export default simulation;
