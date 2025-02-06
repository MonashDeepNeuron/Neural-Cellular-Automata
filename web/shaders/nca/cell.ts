const cell = /* wgsl */ `
struct ParameterShape {
  channels: u32,
  convolutions: u32,
  hidden_channels: u32,
  size: u32
};

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
  let invSize = 1.0 / f32(shape.size);
  let cell = vec2f(f32(input.index % shape.size), f32(input.index / shape.size));
  let gridPos = (input.pos + 1.0) * invSize - 1.0 + cell * 2.0 * invSize;

  return VertexOutput(vec4f(gridPos, 0.0, 1.0), input.index);
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4f {
  let square = shape.size * shape.size;
  let colour = clamp(
    vec4f(state[input.index], state[input.index + square], state[input.index + square * 2], 1.0),
    vec4f(0.0),
    vec4f(1.0)
);

  return colour;
}
`;

export default cell;
