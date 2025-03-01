const cell = /* wgsl */ `
@group(0) @binding(0) var<uniform> grid: vec2f;
@group(0) @binding(1) var<storage> cellState: array<f32>;

struct VertexInput {
    @location(0) pos:vec2f,
    @builtin(instance_index) instance: u32,
};

struct VertexOutput {
    @builtin(position) pos: vec4f,
    @location(0) cell: vec2f,
    @location(1) @interpolate(flat) cellInstance: u32,
};

@vertex
fn vertexMain(input: VertexInput) -> 
    VertexOutput {

    let i = f32(input.instance);
    let state = cellState[input.instance];

    let cell = vec2f(i % grid.x, floor(i/grid.x));
    let cellOffset = cell/grid *2;
    var gridPos : vec2f;

    // In earlier versions, dead squares were defined as squares with size 0
    // This might have enhanced efficiency in terms of what had to be updated 
    // on the screen in each cycle (a guess, not verified)
    
    // Note this verison defines multiple levels of colouration and
    // thus colour is now defined through the fragment shader
    gridPos = (input.pos +1)/grid -1 + cellOffset;

    var output: VertexOutput;
    output.pos = vec4f(gridPos, 0, 1); // ( (X,Y), Z, W)
    output.cellInstance = input.instance;
    output.cell = cell;

    return output;
}

@fragment
fn fragmentMain(vertexOut : VertexOutput) -> @location(0) vec4f {
    let state = f32(cellState[vertexOut.cellInstance]);
    if (state < 0.0) {
        return vec4f(0, pow(0.9, 1/-state), pow(0.98, 1/(pow(-state,3))), 1); // BE blue/purple scale
    }
    // return vec4f(pow(0.98, 1/(pow(state,3))), pow(state, 0.8),pow(0.9, 1/state),  1);//vec4f(1, 1, 1, 1);
    return vec4f(pow(0.9, 1/state), pow(0.98, 1/(pow(state,3))), 0,  1);// BE Red/yellow scale
}
`;

export default cell;
