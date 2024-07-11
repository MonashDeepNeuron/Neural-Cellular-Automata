// SET VARIABLES
export const GRID_SIZE = 64;
export const UPDATE_INTERVAL = 200;
import { WORKGROUP_SIZE, computeShaderWGSL, guiShaderWGSL } from './shader.js';

export async function main() {

    // HEHEHEHE NYAN WAS HERE
    // Josh was here too
    let step = 0

    // Using a purely random initial state for now (see below)
    // let initialState = [
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0,
    //     1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,
    //     0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0,
    // ];

    // DEVICE SETUP - could prob be  a function. yes please make this a function
    const canvas = document.querySelector("canvas");
    if (!navigator.gpu) {
        throw new Error("WebGPU not supported on this browser");
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        throw new Error("No appropriate GPUAdapter found.");
    }

    const device = await adapter.requestDevice();

    // Uniform grid
    const uniformArray = new Float32Array([GRID_SIZE, GRID_SIZE]);
    const uniformBuffer = device.createBuffer({
        label: "Grid Uniforms",
    size: uniformArray.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    device.queue.writeBuffer(uniformBuffer, 0, uniformArray);

    // Cell state arrays
    const cellStateArray = new Uint32Array(GRID_SIZE * GRID_SIZE);
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
    for (let i = 0; i < cellStateArray.length; ++i) {
        cellStateArray[i] = Math.random() > 0.6 ? 1 : 0;
        // cellStateArray[i] = initialState[i];
    }
    device.queue.writeBuffer(cellStateStorage[0], 0, cellStateArray);

    // write to buffer B
    for (let i = 0; i < cellStateArray.length; i++) {
        cellStateArray[i] = i % 2;
    }
    device.queue.writeBuffer(cellStateStorage[1], 0, cellStateArray);

    // FORMAT CANVAS
    const context = canvas.getContext("webgpu");
    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
        device: device,
    format: canvasFormat,
    });

    // VERTEX SETUP for a square
    const verticies = new Float32Array([
    // X,    Y,
    -0.8, -0.8, // Triangle 1
    -0.8, 0.8,
    0.8, 0.8,

    0.8, 0.8, // Triangle 2
    0.8, -0.8,
    -0.8, -0.8,
    ]);

    const vertexBuffer = device.createBuffer({
        label: "Cell verticies", // Error message label
    size: verticies.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });

    device.queue.writeBuffer(vertexBuffer, /*bufferOffset=*/0, verticies);

    const vertexBufferLayout = {
        arrayStride: 8, // 32bit = 4 bytes, 4x2 = 8 bytes to skip to find next vertex
    attributes: [{
        format: "float32x2", // two 32 bit floats per vertex
    offset: 0,
    shaderLocation: 0, // Position, see vertex shader
        }]
    }

    // COMPUTE SHADER MODULE
    const simulationShaderModule = device.createShaderModule(computeShaderWGSL);

    // CELL SHADER MODULE
    const cellShaderModule = device.createShaderModule(guiShaderWGSL);

    // LAYOUTS 
    // COMPUTE SHADER RESOURCE BINDINGS
    const bindGroupLayout = device.createBindGroupLayout({
        label: "Cell Bind Group Layout",
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT,
                buffer: { } // Grid uniform buffer
            },

            {
                binding: 1,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT,
                buffer: {type: "read-only-storage" } // Cell state input buffer
            },

            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {type: "storage" } // Cell state output buffer
            }
        ]
    });


    // PIPE LAYOUT
    const pipelineLayout = device.createPipelineLayout({
        label: "Cell Pipeline Layout",
        bindGroupLayouts: [bindGroupLayout],
    });

    // SIMULATION PIPELINE
    const simulationPipeline = device.createComputePipeline({
        label: "Simulation pipeline",
        layout: pipelineLayout,
        compute: {
            module: simulationShaderModule,
            entryPoint: "computeMain",
        }
    });

    //RENDER PIPELINE            
    const cellPipeline = device.createRenderPipeline({
        label: "Cell pipeline",
        layout: pipelineLayout,
        vertex: {
            module: cellShaderModule,
            entryPoint: "vertexMain",
            buffers: [vertexBufferLayout],
        },
        fragment: {
            module: cellShaderModule,
            entryPoint: "fragmentMain",
            targets: [{
                format: canvasFormat
            }],
        }
    });

    // GROUP BINDING
    const bindGroups = [

        device.createBindGroup({
            label: "Cell renderer bind group A",
            layout: cellPipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {buffer: uniformBuffer }
                },

                {
                    binding: 1,
                    resource: {buffer: cellStateStorage[0] },
                },

                {
                    binding: 2,
                    resource: {buffer: cellStateStorage[1] },
                }
            ],
        }),

        device.createBindGroup({
            label: "Cell render bind group B",
            layout: cellPipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {buffer: uniformBuffer },
                },

                {
                    binding: 1,
                    resource: {buffer: cellStateStorage[1] }
                },

                {
                    binding: 2,
                    resource: {buffer: cellStateStorage[0] }
                }
            ],
        })
    ];

    function updateLoop(){
    
        const encoder = device.createCommandEncoder();
    
        // CREATE COMPUTE TOOL & PERFORM COMPUTATION TASKS
        const computePass = encoder.beginComputePass();
    
        computePass.setPipeline(simulationPipeline);
        computePass.setBindGroup(0, bindGroups[step % 2]);
    
        const workgroupCount = Math.ceil(GRID_SIZE / WORKGROUP_SIZE);
        computePass.dispatchWorkgroups(workgroupCount, workgroupCount);
    
        computePass.end();
    
        // CREATE DRAW TOOL & SET DEFAULT COLOR (BACKGROUND COLOR)
        step++;
        const pass = encoder.beginRenderPass({
            colorAttachments:
            [{
                view: context.getCurrentTexture().createView(),
                loadOp: "clear",
                clearValue: {r: 0, g: 0, b: 0, a: 1 }, // New line
                storeOp: "store",
            }]
        });
    
        // DRAW THE FEATURES
        pass.setPipeline(cellPipeline);
        pass.setVertexBuffer(0, vertexBuffer);
    
        pass.setBindGroup(0, bindGroups[step % 2]);
    
        pass.draw(verticies.length / 2, GRID_SIZE * GRID_SIZE); // 6 verticies, 12 floats
    
        pass.end();
    
        // const commandBuffer = encoder.finish();
        // device.queue.submit([commandBuffer]);
    
        // Finish the command buffer and immediately submit it.
        device.queue.submit([encoder.finish()]);
    }

    setInterval(updateLoop, UPDATE_INTERVAL)

    //setInterval(updateLoop(device, simulationPipeline, bindGroups,step,context,cellPipeline,vertexBuffer,verticies), UPDATE_INTERVAL)
}

export default main