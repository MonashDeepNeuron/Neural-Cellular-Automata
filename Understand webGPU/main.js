// wrong bad ugly imports
import shader from "./vertex.wgsl"

// Setup a variety of things
// synopsis: canvas, device, buffer, bindings, pipeline, encoder.

// CANVAS
// the thing on the html page
const canvas = document.querySelector("canvas");

// DEVICE
//  these things let us talk about the GPUdevice
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();

// i don't know what this is. canvas related
const context = canvas.getContext("webgpu"); // this is where we specify that its webGPU
const format = navigator.gpu.getPreferredCanvasFormat(); // idk bruh
context.configure({
    device: device,
    format: format,
});

// create things and write them to buffer
// the buffer raw binary data that gets sent to the gpu
// write uniform to buffer

// write predetermined vertices to buffer

// set up the information that will be written
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
    usage: GPUBufferUsage.VERTEX
});

// put the binary information into the buffer
device.queue.writeBuffer(vertexBuffer, /*bufferOffset=*/ 0, vertices);

// bindings instruct the GPU how to interpret things in the buffer
// it "binds" the binary data in the buffer to a data type
const bindGroupLayout = device.createBindGroupLayout({
    label: "basic bind group layout",
    entries: [
        {
            binding: 0,
            visibility: 0, // i don't know what this is
            buffer: vertexBuffer,
        }
    ]
});
const bindGroup = device.createBindGroup({
    label: "basic bind group thing",
    layout: bindGroupLayout
});

// a pipeline specifies the bindings to the GPU
const pipelineLayout = createPipelineLayout(
    // something needs to go in here, presumably the binding information
);
const pipeline = device.createRenderPipeline({
    label: "main pipeline",
    layout: pipelineLayout,
    vertex: {}, // necessary
}); // p

// I believe setup is complete here 

// encoder actually makes the GPU do things
