export const guiShader =
    /*wgsl*/`
    @group(0) @binding(0) var<uniform> grid: vec2f;
    @group(0) @binding(1) var<storage> cellState: array<u32>;
    @group(0) @binding(3) var<storage> rule: array<u32>;

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
        VertexOutput{

        let i = f32(input.instance);
        let state = f32(cellState[input.instance]);

        let cell = vec2f(i % grid.x, floor(i/grid.x));
        let cellOffset = cell/grid *2;
        var gridPos : vec2f;

        // In earlier versions, black squares were defined as squares with size 0
        // This probably enhanced efficiency.
        
        // Note this verison defines multiple levels of colouration and
        // thus colour is now defined through the fragment shader
        if (state <= 0){
            gridPos = vec2f(0,0);
        } else {
            gridPos = (input.pos +1)/grid -1 + cellOffset;
        }

        var output: VertexOutput;
        output.pos = vec4f(gridPos, 0, 1); // ( (X,Y), Z, W)
        output.cellInstance = input.instance;
        output.cell = cell;

        return output;
    }

    @fragment
    fn fragmentMain(@location(0) cell: vec2f, @location(1) @interpolate(flat) cellInstance: u32) -> @location(0) vec4f {
        var state = f32(cellState[cellInstance]);
        var intensity = f32(0);
        let num_states = f32(rule[1]);
        if (num_states > 2){
            intensity = (state)/(num_states-1); // 0 counts as a state
            // Eg. Aim to create 0, 0.5, 1 for 3 states 
        } else {
            intensity = state;
        }

        return vec4f(pow(0.9, 1/intensity), pow(0.98, 1/(pow(intensity,3))), pow(intensity, 0.8),  1);//vec4f(1, 1, 1, 1);
    }`