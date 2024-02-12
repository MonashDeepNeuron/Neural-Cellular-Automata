

export default class BufferManager {
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
        const bindGroups = [
            BufferManager.createBindGroup(device, renderPipeline, "Cell renderer bind group A", uniformBuffer, cellStateStorage[0], cellStateStorage[1], ruleStorage),
            BufferManager.createBindGroup(device, renderPipeline, "Cell render bind group B", uniformBuffer, cellStateStorage[1], cellStateStorage[0], ruleStorage)
        ];    
        return {bindGroups, uniformBuffer, cellStateStorage, ruleStorage}  ;
    }

    static setInitialStateBuffer(device, gridSize, initialState){
        // If initial state = null, assign random
        // Cell state arrays
        const cellStateArray = new Uint32Array(gridSize * gridSize);
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
    
        // write to buffer A
        if (initialState == null || initialState.pattern == null){
            for (let i = 0; i < cellStateArray.length; i++) {
                cellStateArray[i] = Math.random() > 0.6 ? 1 : 0; // random starting position
            }
            console.log("Randomising canvas ...");
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
            console.log(`Implementing ${initialState.name}`);
        }
        device.queue.writeBuffer(cellStateStorage[0], 0, cellStateArray);
    
        // write to buffer B
        for (let i = 0; i < cellStateArray.length; i++) {
            cellStateArray[i] = 0;
        }
        device.queue.writeBuffer(cellStateStorage[1], 0, cellStateArray);

        return cellStateStorage;
    }


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

    static createBindGroup(device, renderPipeline, label, uniformBuffer, cellStateA, cellStateB, ruleStorage) {
        return device.createBindGroup({
            label: label,
            layout: renderPipeline.getBindGroupLayout(0), // NOTE: renderpipelne and simulation pipeline use same layout
            entries: [
                { binding: 0, resource: { buffer: uniformBuffer } },
                { binding: 1, resource: { buffer: cellStateA } },
                { binding: 2, resource: { buffer: cellStateB } },
                { binding: 3, resource: { buffer: ruleStorage } },
            ],
        });
    }

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