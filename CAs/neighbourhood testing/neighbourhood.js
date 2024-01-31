import { guiShader } from "./guiShader.js";
import { computeShader } from "./computeShader.js";
import EventManager from "../Shared/managers/EventManager.js";
import DeviceManager from "../Shared/managers/DeviceManager.js";
import BufferManager from "../Shared/managers/BufferManager.js";
import { parseRuleString, displayRule } from "./Parse.js";

// grid of cells
// "this cell" to be anything
// "neighbouring cells" to be highlighted

await DeviceManager.staticConstructor();
const device = DeviceManager.device
const canvas = DeviceManager.canvas

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
const { vertexBuffer, vertexBufferLayout } = BufferManager.loadShapeVertexBuffer(device, SQUARE_VERTICIES);

// load shader code for drawing operations
const cellShaderModule = device.createShaderModule({
    label: 'shader that draws',
    code: guiShader
});

// compute shader

const computeShader = device.createShaderModule({
    label: 'shader that computes neighbourness',
    code:
        /*wgsl*/`
        fn cellIndex(cell: vec2u) -> u32 {
            // Supports grid wrap-around
            // grid.x and grid.y 
            return (cell.y % u32(grid.y))*u32(grid.x)+(cell.x % u32(grid.x));
        }
        
        fn isNeighbour(origin: vec2, target: vec2) -> bool{
            let x_offset = abs(origin.x - target.x)
            let y_offset = abs(origin.y - target.y)
            let distance = x_offset + y_offset
            return distance < 5
        }
        `,
});

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
        module: computeShader,
        entryPoint: "isNeighbour",
    }
});
