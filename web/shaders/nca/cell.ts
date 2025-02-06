const cell = /* wgsl */ `
struct ParameterShape {
  channels: u32,
  convolutions: u32,
  hidden_channels: u32,
  size: u32
}

@group(0) @binding(0) var<uniform> shape: ParameterShape;
@group(0) @binding(1) var<storage> state: array<f32>;

struct VertexInput {
  @location(0) pos: vec2f,
  @builtin(instance_index) index: u32,
};

struct VertexOutput {
  @builtin(position) pos: vec4f,
  @interpolate(flat) @location(0) index: u32
};

@vertex
fn vertex_main(input: VertexInput) -> VertexOutput {
  let cell = vec2f(f32(input.index % shape.size), floor(f32(input.index / shape.size)));
  let cellOffset = cell / f32(shape.size) * 2;
  let gridPos = (input.pos + 1) / f32(shape.size) - 1 + cellOffset;
  
  var output: VertexOutput;
  output.pos = vec4f(gridPos, 0, 1);
  output.index = input.index;
  return output; 
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4f {
  let square = shape.size * shape.size;
  var colour: vec4f;
  colour[0] = clamp(state[input.index + square], 0f, 1f);
  colour[1] = clamp(state[input.index + square], 0f, 1f);
  colour[2] = clamp(state[input.index + square * 2], 0f, 1f);
  colour[3] = 1f;
  return colour;
}
`;

export default cell;
