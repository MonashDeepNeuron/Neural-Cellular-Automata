export const guiShader = {
    label: "Cell shader",
    code:
        /*wgsl*/`
        @group(0) @binding(0) var<uniform> grid: vec2f;
        @group(0) @binding(1) var<storage> cellState: array<u32>;

        struct VertexInput {
            @location(0) pos:vec2f,
            @builtin(instance_index) instance: u32,
        };

        struct VertexOutput {
            @builtin(position) pos: vec4f,
            @location(0) cell: vec2f,
        };

        @vertex
        fn vertexMain(input: VertexInput) -> 
            VertexOutput{

            let i = f32(input.instance);
            let state = f32(cellState[input.instance]);

            let cell = vec2f(i % grid.x, floor(i/grid.x));
            let cellOffset = cell/grid *2;
            let gridPos = (input.pos*state+1)/grid -1 + cellOffset;

            var output: VertexOutput;
            output.pos = vec4f(gridPos, 0, 1); // ( (X,Y), Z, W)
            output.cell = cell;

            return output;
        }

        @fragment
        fn fragmentMain(@location(0) cell: vec2f) -> @location(0) vec4f {
            return vec4f(1, 1, 1, 1);
        }`
};