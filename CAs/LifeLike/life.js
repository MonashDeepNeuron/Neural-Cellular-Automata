// import shaders
import { guiShader } from "./guiShader.js";
import { computeShader } from "./computeShader.js";

// import static manager classes
import EventManager from "./managers/EventManager.js";
import DeviceManager from "./managers/DeviceManager.js";

import BufferManager from "./managers/BufferManager.js";
// import PipelineManager from "./managers/PipelineManager.js";
// construct static classes lol
await DeviceManager.staticConstructor();
const device = DeviceManager.device
const canvas = DeviceManager.canvas

// Set global variables
const GRID_SIZE = document.getElementById("canvas").getAttribute("width"); // from canvas size in life.html
const WORKGROUP_SIZE = 16; // only 1, 2, 4, 8, 16 work. higher is smoother.
const INITIAL_STATE = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1,
    0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0,
    0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

const SQUARE_VERTICIES = new Float32Array([
    // X,    Y,
    -0.8, -0.8, // Triangle 1
    -0.8, 0.8,
    0.8, 0.8,

    0.8, 0.8, // Triangle 2
    0.8, -0.8,
    -0.8, -0.8,
]);


let step = 0; // How many compute passes have been made

// DEVICE SETUP - ran into issues making it a func
// CONFIGURE CANVAS
const context = canvas.getContext("webgpu");
const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
context.configure({
    device: device,
    format: canvasFormat,
});


// DRAWING STUFF setup code 
// vertex setup for a square
const {vertexBuffer, vertexBufferLayout} = BufferManager.loadShapeVertexBuffer(device, SQUARE_VERTICIES);

// load shader code for drawing operations
const cellShaderModule = device.createShaderModule({
    label: 'shader that draws',
    code: guiShader
});


// COMPUTE SHADER setup code
// 1. load grid data into buffers etc. relevant to the simulation inc. parse rulestring 
// 2. define the layout of loaded binary data 
const bindGroupLayout = BufferManager.createBindGroupLayout(device);

// load shader module for running simulation
const simulationShaderModule = device.createShaderModule({
    label: 'shader that computes next state',
    code: computeShader,
    constants: { WORKGROUP_SIZE: WORKGROUP_SIZE }
}
);

// PIPE LAYOUT
const pipelineLayout = device.createPipelineLayout({
    label: "Cell Pipeline Layout",
    bindGroupLayouts: [bindGroupLayout],
});

//RENDER PIPELINE            
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

// SIMULATION PIPELINE
const simulationPipeline = device.createComputePipeline({
    label: "Simulation pipeline",
    layout: pipelineLayout,
    compute: {
        module: simulationShaderModule,
        entryPoint: "computeMain",
    }
});

const bindGroups = BufferManager.initialiseComputeBindgroups(device, renderPipeline, GRID_SIZE, INITIAL_STATE, EventManager.ruleString);


// INITIAL CANVAS SETUP
const encoder = device.createCommandEncoder();
const pass = encoder.beginRenderPass({
    colorAttachments:
        [{
            view: context.getCurrentTexture().createView(),
            loadOp: "clear",
            clearValue: { r: 0, g: 0, b: 0, a: 1 }, // New line
            storeOp: "store",
        }]
});

// Draw the features
pass.setPipeline(renderPipeline);
pass.setVertexBuffer(0, vertexBuffer);
pass.setBindGroup(0, bindGroups[step % 2]);
pass.draw(SQUARE_VERTICIES.length / 2, GRID_SIZE * GRID_SIZE); // 6 vertices, 12 floats
pass.end();

// Finish the command buffer and immediately submit it.
device.queue.submit([encoder.finish()]);

// get and bind events from html
document.getElementById('play').addEventListener('click', EventManager.playPause);  // play pause button
document.getElementById('next').addEventListener('click', EventManager.moveOneFrame); // move one frame button
document.getElementsByTagName("body")[0].addEventListener("keydown", EventManager.keyListener); // key presses
document.getElementById('submitInput').addEventListener('click', EventManager.updateRuleString); // new rule string input button
document.getElementById('speedInput').addEventListener('click', () => {
    EventManager.updateSpeed();
    clearInterval(interval);
    interval = setInterval(updateLoop, EventManager.updateInterval)
}

); // change speed


// iterative update for cells
var interval = setInterval(updateLoop, EventManager.updateInterval); // Interval is accessed from an externally called function
EventManager.forcedUpdate = updateLoop;

function updateLoop() {

    // Only permitted to run if one frame is wanted or
    if (!EventManager.oneFrame) {
        if (!EventManager.running) { return; }
        // Continue if running = true
    }
    else {
        EventManager.oneFrame = false; // Cross-script variable, do not add let,var or const
    }

    // check for new rule string
    if (EventManager.newRuleString) {
        const { ruleStorage } = BufferManager.setRuleBuffer(device, parseRuleString(EventManager.ruleString));
        bindGroups[0] = BufferManager.createBindGroup(device, renderPipeline, "Cell renderer bind group A", uniformBuffer, cellStateStorage[0], cellStateStorage[1], ruleStorage);
        bindGroups[1] = BufferManager.createBindGroup(device, renderPipeline, "Cell render bind group B", uniformBuffer, cellStateStorage[1], cellStateStorage[0], ruleStorage);
  
        EventManager.newRuleString = false // toggle off

    }

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
                clearValue: { r: 0, g: 0, b: 0, a: 1 }, // New line
                storeOp: "store",
            }]
    });

    // DRAW THE FEATURES
    pass.setPipeline(renderPipeline);
    pass.setVertexBuffer(0, vertexBuffer);
    pass.setBindGroup(0, bindGroups[step % 2]);
    pass.draw(SQUARE_VERTICIES.length / 2, GRID_SIZE * GRID_SIZE); // 6 vertices, 12 floats
    pass.end();

    // Finish the command buffer and immediately submit it.
    device.queue.submit([encoder.finish()]);
}
