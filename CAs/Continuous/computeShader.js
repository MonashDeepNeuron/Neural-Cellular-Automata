export const computeShader =
    /*wgsl*/`
    @group(0) @binding(0) var<uniform> grid: vec2f;
    @group(0) @binding(1) var<storage> cellStateIn: array<f32>;
    @group(0) @binding(2) var<storage, read_write> cellStateOut: array<f32>;
    @group(0) @binding(3) var<storage> rule: array<f32>;
    override WORKGROUP_SIZE: u32 = 16;
    
    fn cellIndex(cell: vec2u) -> u32 {
        // Supports grid wrap-around
        // grid.x and grid.y 
        return (cell.y % u32(grid.y))*u32(grid.x)+(cell.x % u32(grid.x));
    }

    fn cellValue(x : u32, y: u32) -> f32 {
        return cellStateIn[cellIndex(vec2(x,y))];
    }

    @compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE)
    fn computeMain(@builtin(global_invocation_id) cell: vec3u) {

        let i = cellIndex(cell.xy);

        // Perform Convolution. We will use step function as activation function for now
        // k1 | k2 | k3
        // k4 | k5 | k6
        // k7 | k8 | k9
        let k1 = cellValue(cell.x-1, cell.y-1) * rule[0];
        let k2 = cellValue(cell.x, cell.y-1) * rule[1] ;
        let k3 = cellValue(cell.x+1, cell.y-1) * rule[2] ;
        let k4 = cellValue(cell.x-1, cell.y) * rule[3] ;
        let k5 = cellValue(cell.x, cell.y) * rule[4] ;
        let k6 = cellValue(cell.x+1, cell.y) * rule[5] ;
        let k7 = cellValue(cell.x-1, cell.y+1) * rule[6] ;
        let k8 = cellValue(cell.x, cell.y+1) * rule[7] ;
        let k9 = cellValue(cell.x+1, cell.y+1) * rule[8] ;

        let result = k1 + k2 + k3 + k4 + k5 + k6 + k7 + k8 + k9 ;

        cellStateOut[i] = result;
                
    }`;