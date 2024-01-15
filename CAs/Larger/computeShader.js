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
        if (cellStateIn[cellIndex(vec2(x,y))] > 0){
            return 1;
        } else {
            return 0;
        }
    }

    fn countMooreNeighbours(cell:vec3u, thisCell:u32, radius:u32) -> u32 {
        let gridSize = radius*2 + 1;
        var activeNeighbours = 0u;
        for (var i = 0u; i < gridSize; i++){ // Row iterator
            for (var j = 0u; j < gridSize; j++){ // Column iterator
                activeNeighbours = activeNeighbours + cellActive(cell.x - radius + j, cell.y - radius + i);
            }
        }
        activeNeighbours = activeNeighbours - cellStateIn[thisCell];
        return activeNeighbours;
    }

    fn countVonNeumannNeighbours(cell:vec3u, thisCell:u32, radius:u32) -> u32 {
        
        //      x      von neumann neighbourhood is calculated based on Manhatten 
        //    x x x         distance/city block distance
        //  x x o x x
        //    x x x
        //      x
        
        var activeNeighbours = 0u;
        for (var i = 0u; i < radius+1; i++){ // Row iterator
            let rowWidth = i*2+1;
            for (var j = 0u; j < rowWidth; j++){ // Column iterator
                activeNeighbours = activeNeighbours + cellActive(cell.x - i + j, cell.y - radius + i);
            }
        }
        for (var i = 0u; i < radius; i++){ // Row iterator
            let rowWidth = i*2+1;
            for (var j = 0u; j < rowWidth; j++){ // Column iterator
                activeNeighbours = activeNeighbours + cellActive(cell.x - i + j, cell.y + radius - i);
            }
        }
        activeNeighbours = activeNeighbours - cellStateIn[thisCell];
        return activeNeighbours;
    }

    fn getActiveNeighbours(neighbourhoodType:u32, cell:vec3u, thisCell:u32, radius:u32) -> u32 {
        switch (neighbourhoodType){
            case 0: {return countMooreNeighbours(cell, thisCell, radius);}
            case 1: {return countVonNeumannNeighbours(cell, thisCell, radius);}
            default: {return countMooreNeighbours(cell, thisCell, radius);}
        }
    }

    @compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE)
    fn computeMain(@builtin(global_invocation_id) cell: vec3u) {

        // Determine number of neighbours
        let thisCell = cellIndex(cell.xy);
        let activeNeighbours = getActiveNeighbours(rule[2], cell, thisCell, rule[0]);
        let states = rule[1];
        //  0  1  2       3       4    ... (no.srange items) ... no.srange+3 
        // [r, c, n, no. srange, su, sl, su, sl, ... , no. brange, bu, bl, bu, bl, ... , n]
        //                      ^if alive                         ^if dead
        // See parse for origin

        let thisState = cellStateIn[thisCell]; // State of this cell
        let ruleOffset =  3-(thisState-1)*(rule[3]+1); // Index of either no.sranges or no.branges
            // More complex than strictly necessary bc I got birth and survival the inconventient way around
        var ruleIter = ruleOffset+1; // Index of current range we are checking
        
        let valsToCheck = rule[ruleOffset]; // the value of either no.sranges of no.branges
        // No. sranges & no. of branges are always even due to parsing method

        for (var i = 0u; i < valsToCheck; i=i+2){
            if (activeNeighbours >= rule[ruleOffset+i+1] && activeNeighbours <= rule[ruleIter+1]){
                if (thisState <= 0){
                    if (states > 1){
                        cellStateOut[thisCell] = states-1;
                    } else {
                        cellStateOut[thisCell] = 1;
                    }
                } else {
                    cellStateOut[thisCell] = thisState;
                }
                return;
            }
            ruleIter = ruleIter + 2;
        }
        if (thisState > 0){
            cellStateOut[thisCell] = thisState -1;
        } else {
            cellStateOut[thisCell] = 0;
        }
    }`;