const cell = /* wgsl */ `
struct GridSize {
  channels: f32;
  rows: f32;
  cols: f32;
}
@group(0) @binding(0) var<uniform> size: GridSize; // channels, rows, cols
@group(0) @binding(1) var<uniform> state: array<f32>;

struct VertexInput {
  @location(0) pos: vec2f,
  @builtin(instance_index) index: u32,
};

struct VertexOutput {
  @builtin(position) pos: vec4f,
  @location(0) i: f32
};

@vertex
fn vertex_main(input: @VertexInput) -> @builtin(position) vec4f {
  let i = f32(input.index);
  let cell = vec2f(i % grid.x, floor(i / grid.x));
  let cellOffset = cell / grid * 2;
  let gridPos = (input.pos + 1) / grid - 1 + cellOffset;
  
  var output: VertexOutput;
  output.pos = vec4f(gridPos, 0, 1);
  return output; 
}

@fragment
fn fragment_main(@location(0) pos: vec2f) -> @location(0) vec4f {

}

`;

export default cell;
