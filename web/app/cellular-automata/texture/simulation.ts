const simulation = /* wgsl */ `
struct ParameterShape {
  channels: u32,
  convolutions: u32,
  hidden_channels: u32,
  size: u32
};

@group(0) @binding(0) var<uniform> shape: ParameterShape;
@group(0) @binding(1) var<storage> state: array<f32>;
@group(0) @binding(2) var<storage> next_state: array<f32>; 
@group(0) @binding(3) var<storage> l1_w: array<f32>;
@group(0) @binding(4) var<storage> l1_b: array<f32>;
@group(0) @binding(5) var<storage> l2_w: array<f32>;
@workgroup_size(8)

@compute
fn compute_main() {
  
}

fn ReLU(x: f32) -> f32 {
  return max(0, x);
}

fn convolve(i: u32, kernel: mat3x3f) -> f32 {

}

fn index(i: vec2u) -> u32 {
  // TODO: Complete with convention in terms or r,c
  return (i.x % shape.size) + (i.y % shape.size)
}
`;

export default simulation;
