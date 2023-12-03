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

        let gridSize = rule[0]*2 + 1; // only square grid currently supported

        // Determine number of neighbours
        var activeNeighbours = 0u;
        
        for (var i = 0u; i < gridSize; i++){ // Row iterator
            for (var j = 0u; j < gridSize; j++){ // Column iterator
                activeNeighbours = activeNeighbours + cellActive(cell.x - (gridSize-1)/2 + j, cell.y-(gridSize-1)/2 + i);
            }
        }

        let thisCell = cellIndex(cell.xy);
        activeNeighbours = activeNeighbours - cellStateIn[thisCell];

        //  0                   1    ... (no.srange items) ... no.srange+2 
        // [r, (c excluded), no. srange, su, sl, su, sl, ... , no. brange, bu, bl, bu, bl, ... , n]
        //                      ^if alive                         ^if dead
        // See parse for origin

        let thisState = cellStateIn[thisCell]; // State of this cell
        let ruleOffset =  1-(thisState-1)*(rule[1]+1); // Index of either no.sranges or no.branges
            // More complex than strictly necessary bc I got birth and survival the inconventient way around
        var ruleIter = ruleOffset+1; // Index of current range we are checking
        
        let valsToCheck = rule[ruleOffset]; // the value of either no.sranges of no.branges

        while (ruleIter <= ruleOffset + valsToCheck){
            if (activeNeighbours >= rule[ruleIter] && activeNeighbours <= rule[ruleIter+1]){
                cellStateOut[thisCell] = 1;
                return;
            }
            ruleIter = ruleIter + 2;
        }
        cellStateOut[thisCell] = 0;
    }`;