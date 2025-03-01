import { loadWeights } from "./readBinary.js";
import { guiShader } from "./guiShader.js";
import { ComputeShaderManager } from "./ComputeManager.js";

/* Import custom event and buffer managers for GCS*/
import EventManager from "./EventManager.js";
import BufferManager from "./BufferManager.js";

import DeviceManager from "../Shared/managers/DeviceManager.js";
import startingPatterns from "./startingPatterns.js";
await DeviceManager.staticConstructor();

const device = DeviceManager.device;
const canvas = DeviceManager.canvas;

/* Set global variables*/
const WORKGROUP_SIZE = 2; /* only 1, 2, 4, 8, 16 work. higher is smoother*/
const INITIAL_TEMPLATE_NO = 1;
const INITIAL_STATE = startingPatterns[INITIAL_TEMPLATE_NO - 1];
const GRID_SIZE = 32;/* from canvas size in life.html*/

EventManager.submitSpeed();

const SQUARE_VERTICIES = new Float32Array([
    // X,    Y,
    -0.8, -0.8, // Triangle 1
    -0.8, 0.8,
    0.8, 0.8,

    0.8, 0.8, // Triangle 2
    0.8, -0.8,
    -0.8, -0.8,
]);

/* Number of compute passes*/
let step = 0;

/* Set up device*/
const context = canvas.getContext("webgpu");
const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
context.configure({
    device: device,
    format: canvasFormat,
});

/* Rendering/drawing set up*/
const { vertexBuffer, vertexBufferLayout } = BufferManager.loadShapeVertexBuffer(device, SQUARE_VERTICIES);

// load shader code for drawing operations
const cellShaderModule = device.createShaderModule({
    label: 'shader that draws',
    code: guiShader
});

/* GPU set up*/
const bindGroupLayout = BufferManager.createBindGroupLayout(device);


/* Pipeline layout*/
const pipelineLayout = device.createPipelineLayout({
    label: "Cell Pipeline Layout",
    bindGroupLayouts: [bindGroupLayout],
});



/* Set up computer shaders*/
ComputeShaderManager.setWorkgroupSize(WORKGROUP_SIZE);
ComputeShaderManager.setPipelineLayout(pipelineLayout);
ComputeShaderManager.initialSetup(device);
ComputeShaderManager.compileNewSimulationPipeline(device);

const renderPipeline = device.createRenderPipeline({
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

/* Attach actions to inputs (buttons, keys)*/
EventManager.bindEvents();

/* Load model weights*/
let weights = await loadWeights('/persist_cat20000.bin');
console.log(weights)



let stochasticMaskSeed = Math.floor(Math.random() * Number.MAX_SAFE_INTEGER); // TODO: ensure datatype is correct
let stochasticMaskArray = new Int32Array([stochasticMaskSeed])
let { bindGroups, uniformBuffer, cellStateStorage, w1, b1, w2, stochasticMaskBuffer } = BufferManager.initialiseComputeBindgroups(device, renderPipeline, GRID_SIZE, INITIAL_STATE, weights, stochasticMaskArray);


/* First render pass to initalise canvas*/
const encoder = device.createCommandEncoder();
renderPass(encoder);
device.queue.submit([encoder.finish()]);



/* Animation rendering and calculation instructions*/
const updateLoop =  () => { //async () => { // TODO: REMOVE ASYNC WHEN REMOVE LOGGING
    const encoder = device.createCommandEncoder();

    renderPass(encoder);
    // Console logging for state if needed for debugging
    // console.log(step)
    // Log cell state storage
    // Only log every 10 frames (or adjust the interval)
    // if (step % 10 === 0) {
    //     logCellStateStorage(device, cellStateStorage, GRID_SIZE).then(() => {
    //         console.log("Logged cell state storage");
    //     });
    // }

    stochasticMaskBuffer = BufferManager.changeStochasticMaskSeed(device, stochasticMaskBuffer)

    /* Perform multiple updates per render pass */
    if (EventManager.running) {
        for (let i = 0; i < EventManager.framesPerUpdateLoop; i++) {
            computePass(encoder);
            step++;
            EventManager.incrementCycleCount();
        }

    }

    /* Someone pressed "one frame" so do one compute pass*/
    else {
        computePass(encoder);
        step++;
        EventManager.incrementCycleCount();
        if (EventManager.skipEvenFrames) {
            computePass(encoder);
            step++;
            EventManager.incrementCycleCount();
        }
    }


    /* Update number of cycles count*/
    EventManager.updateCyclesDisplay();

    /* Finish the command buffer and immediately submit it*/
    device.queue.submit([encoder.finish()]);
}



/* Start update animation*/
EventManager.setUpdateLoop(updateLoop);
EventManager.playPause();

function renderPass(encoder) {
    const renderPass = encoder.beginRenderPass({
        colorAttachments:
            [{
                view: context.getCurrentTexture().createView(),
                loadOp: "clear",
                clearValue: { r: 0, g: 0, b: 0, a: 1 }, /* Clear existing values*/
                storeOp: "store",
            }]
    });

    /* Draw the grid state*/
    renderPass.setPipeline(renderPipeline);
    renderPass.setVertexBuffer(0, vertexBuffer);
    renderPass.setBindGroup(0, bindGroups[step % 2]);
    renderPass.draw(SQUARE_VERTICIES.length / 2, GRID_SIZE * GRID_SIZE);
    renderPass.end();
}


function computePass(encoder) {
    const computePass = encoder.beginComputePass();

    computePass.setPipeline(ComputeShaderManager.simulationPipeline);
    computePass.setBindGroup(0, bindGroups[step % 2]);

    const workgroupCount = Math.ceil(GRID_SIZE / WORKGROUP_SIZE);
    computePass.dispatchWorkgroups(workgroupCount, workgroupCount);

    computePass.end();
}





/* FUNCTIONS FOR READING CELL STATE STORAGE BUFFER*/
// Function to read and log the contents of a GPU buffer
async function readBuffer(device, buffer, size) {
    // Create a buffer for reading with the MAP_READ flag
    const readBuffer = device.createBuffer({
        size: size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    // Create a command encoder and copy the contents of the buffer to the readBuffer
    const commandEncoder = device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(buffer, 0, readBuffer, 0, size);
    device.queue.submit([commandEncoder.finish()]);

    // Wait for the buffer to be mapped to the CPU
    await readBuffer.mapAsync(GPUMapMode.READ);

    // Get the data from the buffer
    const arrayBuffer = readBuffer.getMappedRange();
    const data = new Float32Array(arrayBuffer);

    // Log the buffer contents
    console.log(data);

    // Unmap the buffer
    readBuffer.unmap();
}

// Function to trigger reading the cell state storage
async function logCellStateStorage(device, cellStateStorage, gridSize) {
    const bufferSize = gridSize * gridSize * BufferManager.NUM_CHANNELS * Float32Array.BYTES_PER_ELEMENT;

    // Call readBuffer for both cellStateStorage buffers (double buffering)
    console.log("Cell State A:");
    await readBuffer(device, cellStateStorage[0], bufferSize);

    console.log("Cell State B:");
    await readBuffer(device, cellStateStorage[1], bufferSize);
}