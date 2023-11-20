// wrong bad ugly imports
import shader from "./vertex.wgsl"

// Setup a variety of things
// synopsis: canvas, device, buffer, bindings, pipeline, encoder.

// the thing on the html page
const canvas = document.querySelector("canvas");

//  these things let us talk about the GPUdevice
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();

// i don't know what this is
const context = canvas.getContext("webgpu"); // this is where we specify that its webGPU
const format = navigator.gpu.getPreferredCanvasFormat(); // idk bruh
context.configure({
    device: device,
    format: format,
});

// create things and write them to buffer
// the buffer is the GPUs memory

// bindings instruct the GPU how to interpret things in the buffer
// it "binds" the binary data in the buffer to a data type
const bindGroupLayout = device.createBindGroupLayout({
    label: "basic layout for bind group",
    entries: [
        {
            binding: 0,
            /* visibility: ,
            buffer: ,
            resource: // the type of resource in the buffer, */
        }
    ]
});
const bindGroup = ({
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
