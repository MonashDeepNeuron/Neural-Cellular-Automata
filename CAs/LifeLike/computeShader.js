export const computeShader =
    /*wgsl*/`
    @group(0) @binding(0) var<uniform> grid: vec2f;
    @group(0) @binding(1) var<storage> cellStateIn: array<u32>;
    @group(0) @binding(2) var<storage, read_write> cellStateOut: array<u32>;
    @group(0) @binding(3) var<storage> rule: array<u32>;
    override WORKGROUP_SIZE: u32 = 16;
    // override POSSIBLE_NEIGHBOURS: u32 = 9;
    
    fn cellIndex(cell: vec2u) -> u32 {
        // Supports grid wrap-around
        // grid.x and grid.y 
        return (cell.y % u32(grid.y))*u32(grid.x)+(cell.x % u32(grid.x));
    }

    fn cellActive(x : u32, y: u32) -> u32 {
        return cellStateIn[cellIndex(vec2(x,y))];
    }

    @compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE)
    fn computeMain(@builtin(global_invocation_id) cell: vec3u) {

        // Determine number of neighbours
        let activeNeighbours = 
            cellActive(cell.x+1, cell.y+1) +
            cellActive(cell.x+1, cell.y) +
            cellActive(cell.x+1, cell.y-1) +
            cellActive(cell.x, cell.y-1) +
            cellActive(cell.x, cell.y+1) +
            cellActive(cell.x-1, cell.y-1) +
            cellActive(cell.x-1, cell.y) +
            cellActive(cell.x-1, cell.y+1);

        let i = cellIndex(cell.xy);

        // update using bit operations 
        let shiftNumber = cellStateIn[i] * 9 + activeNeighbours;
        let newState = (rule[0] >> shiftNumber) & 1 ;
        cellStateOut[i] = newState ; 
    }`;