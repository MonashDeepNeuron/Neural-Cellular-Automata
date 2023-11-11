export const WORKGROUP_SIZE = 8;

export const computeShaderWGSL = {
    label: "Game of Life simulation shader",
    code:
        /*wgsl*/`
        @group(0) @binding(0) var<uniform> grid: vec2f;
        @group(0) @binding(1) var<storage> cellStateIn: array<u32>;
        @group(0) @binding(2) var<storage, read_write> cellStateOut: array<u32>;
        
        fn cellIndex(cell: vec2u) -> u32 {
            // Supports grid wrap-around
            return (cell.y%u32(grid.y))*u32(grid.x)+(cell.x % u32(grid.x));
        }

        fn cellActive(x : u32, y: u32) -> u32 {
            return cellStateIn[cellIndex(vec2(x,y))];
        }

        @compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
        fn computeMain(@builtin(global_invocation_id) cell: vec3u) {

            // Determine number of neighbours
            let activeNeighbors = cellActive(cell.x+1, cell.y+1) +
            cellActive(cell.x+1, cell.y) +
            cellActive(cell.x+1, cell.y-1) +
            cellActive(cell.x, cell.y-1) +
            cellActive(cell.x, cell.y+1) +
            cellActive(cell.x-1, cell.y-1) +
            cellActive(cell.x-1, cell.y) +
            cellActive(cell.x-1, cell.y+1);

            let i = cellIndex(cell.xy);

            switch activeNeighbors {
                case 2: {
                    cellStateOut[i] = cellStateIn[i];
                }
                case 3: {
                    cellStateOut[i] = 1;
                }
                default: {
                    cellStateOut[i] = 0;
                }
            }
        }
    `
};


export const guiShaderWGSL = {
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
        }
        `
};

/* 
here's what doesnt work
...
code : await fetch()
...
*/

/*
now trying 
fetch('http://localhost/foo.txt')
  .then(response => response.text())
  .then((data) => {
    console.log(data)
  })
*/
