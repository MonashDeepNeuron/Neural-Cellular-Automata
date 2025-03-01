const cell = /* wgsl */ `
@group(0) @binding(0) var<storage> size: u32;
@group(0) @binding(1) var<storage> state: array<f32>;

struct VertexInput {
  @location(0) pos: vec2f,
  @builtin(instance_index) index: u32,
};

struct VertexOutput {
  @builtin(position) pos: vec4f,
  @location(1) @interpolate(flat) index: u32,
};

@vertex
fn vertex_main(input: VertexInput) -> VertexOutput {
	let invSize = 1.0 / f32(size);
	let cell = vec2f(f32(input.index % size), f32(input.index / size));
	let gridPos = (input.pos + 1.0) * invSize - 1.0 + cell * 2.0 * invSize;

	return VertexOutput(vec4f(gridPos, 0.0, 1.0), input.index);
}

@fragment
fn fragment_main(input : VertexOutput) -> @location(0) vec4f {
  let value = state[input.index];
  if (value < 0.0) {
    // Negative: blue/purple scale
    return vec4f(0, pow(0.9, 1/-value), pow(0.98, 1/(pow(-value,3))), 1);
  } else {
    // Positive: red/yellow scale
    return vec4f(pow(0.9, 1/value), pow(0.98, 1/(pow(value,3))), 0,  1);// BE Red/yellow scale
  }
}
`;

export default cell;
