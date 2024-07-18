export const guiShader =
    /*wgsl*/`
    @group(0) @binding(0) var<uniform> grid: vec2f;
    @group(0) @binding(1) var<storage> cellState: array<f32>;
    // THIS NEEDS TO BE MADE EXTERNAL
    let NUM_CHANNELS = 16;

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

        let cell = vec2f(i % grid.x, floor(i/grid.x));
        let cellOffset = cell/grid *2;
        var gridPos : vec2f;

        // In earlier versions, black squares were defined as squares with size 0
        // This probably enhanced efficiency.
        
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
        let index = vertexOut.cellInstance * NUM_CHANNELS;
        let colour: array<f32, 4>;
        for (int i = 0u; i < 4; i++){
            colour[i] = cellState[index];
        }

        return colour;//vec4f(1, 1, 1, 1);
    }`