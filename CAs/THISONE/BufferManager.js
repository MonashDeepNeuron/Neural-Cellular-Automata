

export default class BufferManager {
    static NUM_CHANNELS = 16;
    static loadShapeVertexBuffer(device, shapeVerticies) {
        /*
        Loads vertex data into GPU buffer

        Parameters:
            device: The WebGPU device used to create buffers.
            shapeVertices: A typed array of vertex data.
        Returns:
            An object containing the vertex buffer and its layout configuration.
        */

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

        return { vertexBuffer, vertexBufferLayout };
    }


    static initialiseComputeBindgroups(device, renderPipeline, gridSize, initialState, weights) {


        // Uniform grid
        const uniformArray = new Float32Array([gridSize, gridSize]);
        const uniformBuffer = device.createBuffer({
            label: "Grid Uniforms",
            size: uniformArray.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(uniformBuffer, 0, uniformArray);

        let cellStateStorage = BufferManager.setInitialStateBuffer(device, gridSize, initialState);

        // const { w1, b1, w2 } = BufferManager.setRuleBuffer(device, weights);
        const { w1Storage: w1, b1Storage: b1, w2Storage: w2 } = BufferManager.setRuleBuffer(device, weights);



        // setup bind groups
        const bindGroups = [
            BufferManager.createBindGroup(device, renderPipeline, "Cell renderer bind group A", uniformBuffer, cellStateStorage[0], cellStateStorage[1], w1, b1, w2),
            BufferManager.createBindGroup(device, renderPipeline, "Cell render bind group B", uniformBuffer, cellStateStorage[1], cellStateStorage[0], w1, b1, w2)
        ];
        return { bindGroups, uniformBuffer, cellStateStorage, w1, b1, w2 };
    }

    static setInitialStateBuffer(device, gridSize, initialState) {
        // If initial state = null, assign random
        // Cell state arrays

        /*
        Implements double buffering - to improve simulation quality? 
        */

        const cellStateArray = new Float32Array(gridSize * gridSize * BufferManager.NUM_CHANNELS);
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
        if (initialState == null || initialState.pattern == null) {
            for (let i = 0; i < cellStateArray.length; i++) {
                cellStateArray[i] = Math.random(); // random starting position
            }
            console.log("Randomising canvas ...");
        } else {
            for (let i = 0; i < cellStateArray.length; i++) {
                cellStateArray[i] = 0;
            }
            const centreOffset = Math.floor((gridSize - initialState.width) / 2);
            
            for (let i = 0; i < initialState.width; i++) {
                for (let j = 0; j < initialState.height; j++) {
                    console.log(i + centreOffset + (j + centreOffset) * gridSize)
                    for (let k = 0; k < 16; k++){ // TODO: Not hardcode number of channels
                        cellStateArray[(i + centreOffset + (j + centreOffset) * gridSize)*16+k] = initialState.pattern[(i + j * initialState.width)*16+k];
                    }
                }
            }
            console.log(cellStateArray);
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


    static createBindGroupLayout(device) {
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
                    buffer: { type: "read-only-storage" } // w1
                },
                {
                    binding: 4,
                    visibility: GPUShaderStage.COMPUTE | GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT,
                    buffer: { type: "read-only-storage" } // b1
                },
                {
                    binding: 5,
                    visibility: GPUShaderStage.COMPUTE | GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT,
                    buffer: { type: "read-only-storage" } // w2
                }
            ]
        });
    }

    static createBindGroup(device, renderPipeline, label, uniformBuffer, cellStateA, cellStateB, w1, b1, w2) {
        return device.createBindGroup({
            label: label,
            layout: renderPipeline.getBindGroupLayout(0), // NOTE: renderpipelne and simulation pipeline use same layout
            entries: [
                { binding: 0, resource: { buffer: uniformBuffer } },
                { binding: 1, resource: { buffer: cellStateA } },
                { binding: 2, resource: { buffer: cellStateB } },
                { binding: 3, resource: { buffer: w1 } },
                { binding: 4, resource: { buffer: b1 } },
                { binding: 5, resource: { buffer: w2 } }
            ],
        });
    }

    static setRuleBuffer(device, modelWeights) {

        const w1 = modelWeights.slice(0, (128 * 48));
        const b1 = modelWeights.slice(0, (128 * 1));
        const w2 = modelWeights.slice(0, (128 * 16));

        const w1Storage = device.createBuffer({
            label: "W1 Storage",
            size: w1.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        const b1Storage = device.createBuffer({
            label: "B1 Storage",
            size: b1.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        const w2Storage = device.createBuffer({
            label: "W2 Storage",
            size: w2.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        device.queue.writeBuffer(w1Storage, 0, w1);
        device.queue.writeBuffer(b1Storage, 0, b1);
        device.queue.writeBuffer(w2Storage, 0, w2);

        return { w1Storage, b1Storage, w2Storage };
    }
}