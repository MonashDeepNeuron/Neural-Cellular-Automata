// import shaders
import { guiShader } from "./guiShader.js";
import { computeShader } from "./computeShader.js";

// import static manager classes
import EventManager from "../Shared/managers/EventManager.js";
import DeviceManager from "../Shared/managers/DeviceManager.js";

import BufferManager from "../Shared/managers/BufferManager.js";
import { parseRuleString, displayRule } from "./Parse.js";

import startingPatterns from "./startingPatterns.js";
// import PipelineManager from "./managers/PipelineManager.js";
// construct static classes lol
await DeviceManager.staticConstructor();
const device = DeviceManager.device
const canvas = DeviceManager.canvas

// Set global variables
const WORKGROUP_SIZE = 16; // only 1, 2, 4, 8, 16 work. higher is smoother. // There is a limitation though to some pcs/graphics cards
const INITIAL_STATE = startingPatterns[2];
const GRID_SIZE = INITIAL_STATE.minGrid*2;//document.getElementById("canvas").getAttribute("width"); // from canvas size in life.html

EventManager.ruleString = INITIAL_STATE.rule;
displayRule(EventManager.ruleString);

EventManager.getRule = () => {
    let ruleString = "";
    for (let i = 1; i < 10; i++){
        ruleString += (document.getElementById(`kernel${i}`).value + ',');
    }
    console.log(`The rule string is ${ruleString}`);
    return ruleString;
}


let select = document.getElementById("templateSelect");

let template = document.createElement("option");
template.text = "Random";
template.value = -1;
select.add(template);

for (let i = 0; i < startingPatterns.length; i++){
    let template = document.createElement("option");
    template.text = startingPatterns[i].name;
    template.value = i;
    select.add(template);
}


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

let { bindGroups, uniformBuffer, cellStateStorage, ruleStorage } = BufferManager.initialiseComputeBindgroups(device, renderPipeline, GRID_SIZE, INITIAL_STATE, parseRuleString(EventManager.ruleString) );


// INITIAL CANVAS SETUP, 1st render pass
const encoder = device.createCommandEncoder();
renderPass(encoder);

// Finish the command buffer and immediately submit it.
device.queue.submit([encoder.finish()]);

EventManager.bindEvents();

EventManager.updateLoop = () => {

    // Only permitted to run if one frame is wanted or
    if (!EventManager.oneFrame) {
        if (!EventManager.running) { return; }
        // Continue if running = true
    }
    else {
        EventManager.oneFrame = false; // Cross-script variable, do not add let,var or const
    }

    if (EventManager.resetTemplate){
        console.log(`Resetting canvas bump`)
        let initialState = null;
        if (EventManager.templateNo >= 0){
            initialState = startingPatterns[EventManager.templateNo];
        }

        if (initialState != null){
            EventManager.ruleString = initialState.rule;
        }

        ruleStorage = BufferManager.setRuleBuffer(device, parseRuleString(EventManager.ruleString));
        cellStateStorage = BufferManager.setInitialStateBuffer(device, GRID_SIZE, initialState);
        
        bindGroups[0] = BufferManager.createBindGroup(device, renderPipeline, "Cell renderer bind group A", uniformBuffer, cellStateStorage[0], cellStateStorage[1], ruleStorage);
        bindGroups[1] = BufferManager.createBindGroup(device, renderPipeline, "Cell render bind group B", uniformBuffer, cellStateStorage[1], cellStateStorage[0], ruleStorage);

        EventManager.resetTemplate = false;
        step = 0;
        EventManager.running = true;
        
        displayRule(EventManager.ruleString);
    }


    // check for new rule string
    if (EventManager.newRuleString) {
        ruleStorage = BufferManager.setRuleBuffer(device, parseRuleString(EventManager.ruleString));
        bindGroups[0] = BufferManager.createBindGroup(device, renderPipeline, "Cell renderer bind group A", uniformBuffer, cellStateStorage[0], cellStateStorage[1], ruleStorage);
        bindGroups[1] = BufferManager.createBindGroup(device, renderPipeline, "Cell render bind group B", uniformBuffer, cellStateStorage[1], cellStateStorage[0], ruleStorage);
  
        EventManager.newRuleString = false // toggle off

    }

    const encoder = device.createCommandEncoder();

    // CREATE COMPUTE TOOL & PERFORM COMPUTATION TASKS
    computePass(encoder);

    // CREATE DRAW TOOL & SET DEFAULT COLOR (BACKGROUND COLOR)
    step++;
    renderPass(encoder);

    // Finish the command buffer and immediately submit it.
    device.queue.submit([encoder.finish()]);
}



// start iterative update for cells
EventManager.currentTimer = setInterval(EventManager.updateLoop, EventManager.updateInterval); // Interval is accessed from an externally called function
EventManager.forcedUpdate = () => {
    EventManager.oneFrame = true;
    EventManager.updateLoop();
}




// FUNCTIONS for convenient break-up of code
    // Can't be removed because relies on a ton of 
    // definitions from this chunk of code.
function renderPass(encoder){
    const renderPass = encoder.beginRenderPass({
        colorAttachments:
            [{
                view: context.getCurrentTexture().createView(),
                loadOp: "clear",
                clearValue: { r: 0, g: 0, b: 0, a: 1 }, // New line
                storeOp: "store",
            }]
    });
    
    // Draw the features
    renderPass.setPipeline(renderPipeline);
    renderPass.setVertexBuffer(0, vertexBuffer);
    renderPass.setBindGroup(0, bindGroups[step % 2]);
    renderPass.draw(SQUARE_VERTICIES.length / 2, GRID_SIZE * GRID_SIZE); // 6 vertices, 12 floats
    renderPass.end();
}


function computePass(encoder){
    const computePass = encoder.beginComputePass();

    computePass.setPipeline(simulationPipeline);
    computePass.setBindGroup(0, bindGroups[step % 2]);

    const workgroupCount = Math.ceil(GRID_SIZE / WORKGROUP_SIZE);
    computePass.dispatchWorkgroups(workgroupCount, workgroupCount);

    computePass.end();
}