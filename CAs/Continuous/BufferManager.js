

export default class BufferManager {


    // Notes for vertex buffer
    // A square that will be drawn on each cell. Vertexs loaded correspond to 
    // if the cell were a 1x1 square. This will be scaled and positioned by 
    // guiShader code (specifically vertexShader)

    /**
     * Load the verticies to be drawn
     * @param {GPUDevice} device 
     * @param {Float32Array} shapeVerticies Verticies of triangles that form the  
     *          shape.
     * @returns GPUBuffer vertexBuffer, object vertexBufferLayout
     */
    static loadShapeVertexBuffer(device, shapeVerticies){
    
        // load verticies into buffer
        const vertexBuffer = device.createBuffer({
            label: "Cell vertices", // Error message label
            size: shapeVerticies.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(vertexBuffer, /*bufferOffset=*/0, shapeVerticies);
        
        // define layout of loaded binary data
        const vertexBufferLayout = {
            arrayStride: 8, // 32bit = 4 bytes, 4x2 = 8 bytes to skip to find next vertex
            attributes: [{
                format: "float32x2", // two 32 bit floats per vertex
                offset: 0,
                shaderLocation: 0, // Position, see vertex shader
            }]
        }
    
        return {vertexBuffer, vertexBufferLayout};
    }



    /**
     * Define where and what information can be accessed by the GPU code
     * Set and make all required information available to the GPU (for the first time)
     * @param {GPUDevice} device the GPU device object 
     * @param {GPURenderPipeline} renderPipeline
     * @param {Number} gridSize sidelength of grid, where the grid is a square array of cells
     * @param {Array} initialState the initial state of the grid, filled with floating point values
     *      NOTE: This may be null, where null represents a randomised grid. If the pattern 
     *      is null, this also constitutes a randomised grid.
     * @param {Float32Array} rule The numbers from the current rule (kernel)
     * @returns bindGroups, uniformBuffer, cellStateStorage, ruleStorage
    **/
    static initialiseComputeBindgroups(device, renderPipeline, gridSize, initialState, rule){
        // Uniform grid
        const uniformArray = new Float32Array([gridSize, gridSize]);
        const uniformBuffer = device.createBuffer({
            label: "Grid Uniforms",
            size: uniformArray.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(uniformBuffer, 0, uniformArray);
    
        let cellStateStorage = BufferManager.setInitialStateBuffer(device, gridSize, initialState);

        const ruleStorage = BufferManager.setRuleBuffer(device, rule);
    
        // setup bind groups
            // NOTE: Cell updates flip flop between cell storage 0 and cell storage 1.
            // There are two bindgroup layouts available in the code, one where the 
            // input storage is storage 0, and outputs to storage 1, and one where 1 is the input
            // and 0 is the output. This means there is no need for sophisticated 
            // synchronisation.

        const bindGroups = [
            BufferManager.createBindGroup(
                    device, renderPipeline, "Cell renderer bind group A",
                    uniformBuffer, cellStateStorage[0], cellStateStorage[1],
                    ruleStorage
                ),
            BufferManager.createBindGroup(
                    device, renderPipeline, "Cell render bind group B",
                    uniformBuffer, cellStateStorage[1], cellStateStorage[0],
                    ruleStorage
                )
        ];    
        return {bindGroups, uniformBuffer, cellStateStorage, ruleStorage}  ;
    }



    /**
     * Defines the buffer sizes and sets up the buffers for the cell storages 
     * Randomises the canvas when the initialState is null
     * @param {GPUDevice} device 
     * @param {Number} gridSize sidelength of grid, where the grid is a square array of cells
     * @param {Array} initialState the initial state of the grid
     *      NOTE: This may be null, where null represents a randomised grid. If the pattern 
     *      is null, this also constitutes a randomised grid.
     * @returns GPUBuffer[2] cellStateStorage with two buffers. 
     *      cellStateStorage[0] contains the initial state, cellStateStorage[1] is blank
     */
    static setInitialStateBuffer(device, gridSize, initialState){
        // If initial state = null, assign random
        // Cell state arrays

        // This cell state array obj. is discarded, it's just used to instruct the 
        // device.writeBuffer command what values to write into the GPU resource buffers.
        const cellStateArray = new Float32Array(gridSize * gridSize);
        
        const cellStateStorage = [
            device.createBuffer({
                label: "Cell State A",
                size: cellStateArray.byteLength,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            }),
    
            device.createBuffer({
                label: "Cell State B",
                size: cellStateArray.byteLength,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            })
        ];
    
        // write to buffer A (input buffer first)
        // If no state is provided, create randomised state
        // otherwise fill with the provided template centered in the 
        // middle of the grid, (grid is a square with sidelength gridsize)
        // This NEEDS TO BE MOVED OUT OF HERE 
        if (initialState == null || initialState.pattern == null){
            for (let i = 0; i < cellStateArray.length; i++) { // Pretty sure this is the only difference between the shared BufferManager.js and this one 
                cellStateArray[i] = Math.random()*2 -1; // random starting position from 1 to -1
            }
        } else {
            for (let i = 0; i < cellStateArray.length; i++) {
                cellStateArray[i] = 0;
            }
            const centreOffset = Math.floor((gridSize-initialState.width)/2);
            for (let i = 0; i < initialState.width; i++) {
                for (let j = 0; j < initialState.height; j++){
                    cellStateArray[i+centreOffset+(j+centreOffset)*gridSize] = initialState.pattern[i+j*initialState.width];
                }
            }
        }
        device.queue.writeBuffer(cellStateStorage[0], 0, cellStateArray);
    
        // write to buffer B
        // Empty buffer B, this will be used as an output buffer first
        for (let i = 0; i < cellStateArray.length; i++) {
            cellStateArray[i] = 0;
        }
        device.queue.writeBuffer(cellStateStorage[1], 0, cellStateArray);

        return cellStateStorage;
    }



    /**
     * Convenience function, creates the bindings and sets the accessibility of 
     * GPU-available resources. Bindgroup layout defines the data visibility and 
     * permissions
     * @param {GPUDevice} device 
     * @returns bindgroup layout with 4 entries (0-3)
     */
    static createBindGroupLayout(device){
        return device.createBindGroupLayout({
            label: "Cell Bind Group Layout",
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT,
                    buffer: {} // Grid uniform buffer
                },
        
                {
                    binding: 1,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT,
                    buffer: { type: "read-only-storage" } // Cell state input buffer
                },

                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "storage" } // Cell state output buffer
                },
        
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE | GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT,
                    buffer: { type: "read-only-storage" } // Ruleset
                }
            ]
        });
    }



    /**
     * Defines the way that each resource is referenced/accessed from WGSL code
     * This function ensures that consistent binding numbers are used throughout
     * This must be kept consistent with the bindgroup layout
     * @param {GPUDevice} device 
     * @param {GPURenderPipeline} renderPipeline 
     * @param {String} label 
     * @param {GPUBuffer} uniformBuffer 
     * @param {GPUBuffer} inputStateBuffer 
     * @param {GPUBuffer} outputStateBuffer 
     * @param {GPUBuffer} ruleStorage 
     * @returns GPUBindGroupLayout
     */
    static createBindGroup(device, renderPipeline, label, uniformBuffer, inputStateBuffer, outputStateBuffer, ruleStorage) {
        return device.createBindGroup({
            label: label,
            layout: renderPipeline.getBindGroupLayout(0), // NOTE: renderpipeline and simulation pipeline use same layout
            entries: [
                { binding: 0, resource: { buffer: uniformBuffer } },
                { binding: 1, resource: { buffer: inputStateBuffer } },
                { binding: 2, resource: { buffer: outputStateBuffer } },
                { binding: 3, resource: { buffer: ruleStorage } },
            ],
        });
    }



    /**
     * Convenience function for code organisation.
     * Writes out the rule buffer such that it is consistent with expected settings
     * @param {GPUDevice} device 
     * @param {Float32Array} rule The numbers from the current rule (kernel)
     * @returns rulestorage as GPUBuffer object
     */
    static setRuleBuffer(device, rule) {
        const ruleArray = rule;
        const ruleStorage = device.createBuffer({
            label: "Rule Storage",
            size: ruleArray.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(ruleStorage, 0, ruleArray);
        return ruleStorage;
    }
}