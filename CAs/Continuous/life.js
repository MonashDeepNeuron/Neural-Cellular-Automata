// import shaders
import { guiShader } from "./guiShader.js";
import { ComputeShaderManager } from "./computeShader.js";

// import static manager classes
import EventManager from "../Shared/managers/EventManager.js";
import DeviceManager from "../Shared/managers/DeviceManager.js";

import BufferManager from "./BufferManager.js";
import { parseRuleString, displayRule } from "./Parse.js";

import startingPatterns from "./startingPatterns.js";
// import PipelineManager from "./managers/PipelineManager.js";
// construct static classes lol
await DeviceManager.staticConstructor();
const device = DeviceManager.device;
const canvas = DeviceManager.canvas;

// Set global variables
const WORKGROUP_SIZE = 16; // only 1, 2, 4, 8, 16 work. higher is smoother. // There is a limitation though to some pcs/graphics cards
const INITIAL_TEMPLATE_NO = 2;
const INITIAL_STATE = startingPatterns[INITIAL_TEMPLATE_NO - 1];
const GRID_SIZE = 1024;//INITIAL_STATE.minGrid*2;//document.getElementById("canvas").getAttribute("width"); // from canvas size in life.html

EventManager.ruleString = INITIAL_STATE.rule;
EventManager.submitSpeed();

displayRule(EventManager.ruleString);
console.log(INITIAL_STATE)
const SQUARE_VERTICIES = new Float32Array([
    // X,    Y,
    -0.8, -0.8, // Triangle 1
    -0.8, 0.8,
    0.8, 0.8,

    0.8, 0.8, // Triangle 2
    0.8, -0.8,
    -0.8, -0.8,
]);

EventManager.getRule = () => {
    let ruleString = "";
    for (let i = 1; i < 10; i++) {
        let nextVal = document.getElementById(`kernel${i}`).value;
        if (nextVal == null || nextVal == "") {
            nextVal = 0;
        }
        ruleString += (nextVal + ',');

    }
    console.log(`The rule string is ${ruleString}`);
    return ruleString;
}

let select = document.getElementById("templateSelect");

for (let i = 0; i < startingPatterns.length; i++) {
    let template = document.createElement("option");
    template.text = startingPatterns[i].name;
    template.value = i;
    template.id = startingPatterns[i].name;
    select.add(template);
}
document.getElementById("templateSelect").value = INITIAL_TEMPLATE_NO - 1;

ComputeShaderManager.updateActivationField(INITIAL_STATE.activation);

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
const { vertexBuffer, vertexBufferLayout } = BufferManager.loadShapeVertexBuffer(device, SQUARE_VERTICIES);

// load shader code for drawing operations
const cellShaderModule = device.createShaderModule({
    label: 'shader that draws',
    code: guiShader
});


// GPU setup code
// 1. load grid data into buffers etc. relevant to the simulation inc. parsed rulestring 
// 2. define the layout of loaded binary data 
const bindGroupLayout = BufferManager.createBindGroupLayout(device);


// PIPE LAYOUT
const pipelineLayout = device.createPipelineLayout({
    label: "Cell Pipeline Layout",
    bindGroupLayouts: [bindGroupLayout],
});


//COMPUTE SHADER SETUP AND PIPELINE
// Set up compute shader with custom activation function
ComputeShaderManager.setWorkgroupSize(WORKGROUP_SIZE);
ComputeShaderManager.setPipelineLayout(pipelineLayout);
ComputeShaderManager.updateActivationField(INITIAL_STATE.activation);
ComputeShaderManager.initialSetup(device);
ComputeShaderManager.compileNewSimulationPipeline(device);

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

// SET BUFFERS
let { bindGroups, uniformBuffer, cellStateStorage, ruleStorage } = BufferManager.initialiseComputeBindgroups(device, renderPipeline, GRID_SIZE, INITIAL_STATE, parseRuleString(EventManager.ruleString));


// INITIAL CANVAS SETUP, 1st render pass
const encoder = device.createCommandEncoder();
renderPass(encoder);

// Finish the command buffer and immediately submit it.
device.queue.submit([encoder.finish()]);

// Attatch actions to inputs (buttons, keys)
EventManager.bindEvents();

// Animation rendering and calculation instructions
const updateLoop = () => {

    // The user has set a new tmeplate
    if (EventManager.resetTemplate || EventManager.randomiseGrid) {

        // Assume that reset template and radomise grid are mutually exclusive events
        // Prioritise resetTemplate

        console.log(`Resetting canvas bump`)
        let newRuleStorage = null;
        let initialState = null;

        if (EventManager.resetTemplate) {
            initialState = startingPatterns[EventManager.templateNo];
            EventManager.ruleString = initialState.rule;
            newRuleStorage = BufferManager.setRuleBuffer(device, parseRuleString(EventManager.ruleString));

            // Doin a sneaky here, this means the template reset has to go before the activation setup
            ComputeShaderManager.updateActivationField(initialState.activation);
            ComputeShaderManager.submitActivation(); // Equivalent of what happens when someone pressing submit activation button

        } else {
            newRuleStorage = ruleStorage;
        }

        const newCellStateStorage = BufferManager.setInitialStateBuffer(device, GRID_SIZE, initialState);
        bindGroups[0] = BufferManager.createBindGroup(device, renderPipeline, "Cell renderer bind group A", uniformBuffer, newCellStateStorage[0], newCellStateStorage[1], newRuleStorage);
        bindGroups[1] = BufferManager.createBindGroup(device, renderPipeline, "Cell render bind group B", uniformBuffer, newCellStateStorage[1], newCellStateStorage[0], newRuleStorage);

        cellStateStorage = newCellStateStorage;
        ruleStorage = newRuleStorage;
        EventManager.resetTemplate = false;
        EventManager.randomiseGrid = false;
        step = 0;
        EventManager.resetCycleCount();

        displayRule(EventManager.ruleString);
    }

    // Set new activation function and recompile shader if required
    if (ComputeShaderManager.newActivation) {
        ComputeShaderManager.compileNewSimulationPipeline(device);
        ComputeShaderManager.newActivation = false;
    }

    // check for new rule string
    if (EventManager.newRuleString) {
        const newRuleStorage = BufferManager.setRuleBuffer(device, parseRuleString(EventManager.ruleString));
        ruleStorage = newRuleStorage;
        bindGroups[0] = BufferManager.createBindGroup(device, renderPipeline, "Cell renderer bind group A", uniformBuffer, cellStateStorage[0], cellStateStorage[1], newRuleStorage);
        bindGroups[1] = BufferManager.createBindGroup(device, renderPipeline, "Cell render bind group B", uniformBuffer, cellStateStorage[1], cellStateStorage[0], newRuleStorage);

        EventManager.newRuleString = false // toggle off
    }

    const encoder = device.createCommandEncoder();

    // CREATE COMPUTE TOOL & PERFORM COMPUTATION TASKS
    if (EventManager.running) {
        for (let i = 0; i < EventManager.framesPerUpdateLoop; i++) {
            computePass(encoder);
            step++; // Note this counter primarily indicates which cell state should be used
            // In this case the output of the compute pass will be used as input, thus the opposite of
            // what was used for the compute pass. Hence increment after compute pass but before rendering frame
            EventManager.incrementCycleCount();
        }

    } else { // Someone pressed do one frame, so update once

        computePass(encoder);
        step++; // Note this counter primarily indicates which cell state should be used
        // In this case the output of the compute pass will be used as input, thus the opposite of
        // what was used for the compute pass. Hence increment after compute pass but before rendering frame
        EventManager.incrementCycleCount();

        if (EventManager.skipEvenFrames) {
            computePass(encoder);
            step++;
            EventManager.incrementCycleCount();
        }
    }

    // CREATE DRAW TOOL & SET DEFAULT COLOR (BACKGROUND COLOR)
    renderPass(encoder);

    EventManager.updateCyclesDisplay();
    // Finish the command buffer and immediately submit it.
    device.queue.submit([encoder.finish()]);
}



// start iterative update for cells
EventManager.setUpdateLoop(updateLoop);
EventManager.playPause();




// FUNCTIONS for convenient break-up of code
// Can't be removed because relies on a ton of 
// definitions from this chunk of code.
function renderPass(encoder) {
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


function computePass(encoder) {
    const computePass = encoder.beginComputePass();

    computePass.setPipeline(ComputeShaderManager.simulationPipeline);
    computePass.setBindGroup(0, bindGroups[step % 2]);

    const workgroupCount = Math.ceil(GRID_SIZE / WORKGROUP_SIZE);
    computePass.dispatchWorkgroups(workgroupCount, workgroupCount);

    computePass.end();
}