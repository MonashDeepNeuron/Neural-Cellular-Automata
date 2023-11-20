// wrong bad ugly imports
import vertex from "./vertex.wgsl"

// synopsis: canvas, device, buffer, bindings, pipeline, encoder.
// SET UP 

// CANVAS
// the thing on the html page
const canvas = document.querySelector("canvas");

// DEVICE
// these things let us talk about/to the GPU
const adapter = await navigator.gpu.requestAdapter(); // returns 
const device = await adapter.requestDevice();

// i don't know what this is. canvas related
const context = canvas.getContext("webgpu"); // this is where we specify that its webGPU
const format = navigator.gpu.getPreferredCanvasFormat(); // idk bruh
context.configure({
    device: device,
    format: format,
});

// BUFFER
// buffer is raw binary data that gets sent to the gpu

// write predetermined vertices to buffer

// declare information that will be written
const vertices = new Float32Array([
    // X,    Y,
    -0.8, -0.8, // Triangle 1
    -0.8, 0.8,
    0.8, 0.8,
]);

// create (initialise) a buffer, its empty
const vertexBuffer = device.createBuffer({
    label: "a triangle",
    size: vertices.byteLength,
    usage: GPUBufferUsage.VERTEX, // vertex buffers are kinda special
});

// put the binary information into the buffer
device.queue.writeBuffer(vertexBuffer, 0, vertices); // dst, offset, src

// BIND GROUPS
// bind groups instruct the GPU how to interpret things in the buffer
// they "binds" the binary data in the buffer to a data type
// note: bind groups also have other purposes (e.g. textures?? and samplers??)
const bindGroupLayout = device.createBindGroupLayout({
    label: "basic bind group layout",
    entries: [{
        binding: 0,
        visibility: GPUShaderStage.VERTEX, // the gpu sees it at different points depending on flag
        buffer: vertexBuffer,
    }],
});
const bindGroup = device.createBindGroup({
    label: "basic bind group thing",
    layout: bindGroupLayout,
    entries: [{
        binding: 0,
        resource: { buffer: vertexBuffer },
    }], // comma for consistency i guess
});

// PIPELINE CREATION
// a pipeline specifies the bindings to the GPU
const pipelineLayout = device.createPipelineLayout({
    label: "basic layout",
    entries: { bindGroupLayouts: [bindGroupLayout] }
});
const pipeline = device.createRenderPipeline({
    label: "main pipeline",
    layout: pipelineLayout,
    vertex: {}, // necessary, this is the .wgsl module which I will write later
}); // p

// ENCODER
// makes the GPU actually do things by queuing to the command buffer
// command buffer != buffer 
