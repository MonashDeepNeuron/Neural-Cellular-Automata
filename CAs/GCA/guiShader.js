export const guiShader =
    /*wgsl*/`
    @group(0) @binding(0) var<uniform> grid: vec2f;
    @group(0) @binding(1) var<storage> cellState: array<f32>;
    // THIS NEEDS TO BE MADE EXTERNAL
    const NUM_CHANNELS = 16;

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
        var colour: vec4f = vec4f(0.0, 0.0, 0.0, 0.0);

        /* If cell's alpha channel is 0 */
        /* TODO: CURRENTLY PRINTING OUT THE CHANNELS VALUES EVEN IF ALPHA = 0*/
        if (cellState[index + 3] == 0) {
            // for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                
            //     colour[i] = cellState[index + i];
            //     // colour[i] =0; //cellState[index + i];
            // }
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                colour[i] = 0.0;  
                if (cellState[index + i]> 1){
                    colour[i] = 0.5;
                }
                else if (cellState[index + i]< 0){
                    colour[i] = 0;
                }
                else{
                    colour[i] = cellState[index + i];
                }

            }
        
        /* If cell's alpha channel > 0 */
        /* Clipping the colour values between 0 and 1*/
        } else { 
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                if (cellState[index + i]> 1){
                    colour[i] = 1;
                }
                else if (cellState[index + i]< 0){
                    colour[i] = 0;
                }
                else{
                colour[i] = cellState[index + i];
                }
            }
        }

        return colour;
    }`