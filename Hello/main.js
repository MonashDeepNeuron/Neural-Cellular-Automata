// Imports
import { shaderSrc } from "./shaders.js";

// Broad Overview
// Canvas, Device, Texture, 
// Buffer, Bind Groups, Shaders, (this part might be different between models)
// Pipeline, Encoder, Pass

// SETUP
const canvas = document.querySelector("canvas");
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();
const context = canvas.getContext("webgpu");
const format = navigator.gpu.getPreferredCanvasFormat();
context.configure({
    device: device,
    format: format,
});

// TEXTURE, buffer with special operations (good for 2D and 3D data)
const texture = context.getCurrentTexture();

// BUFFER, raw binary data that gets sent to the gpu
const vertices = new Float32Array([
    // X,    Y,
    -0.8, -0.8,
    -0.8, 0.8,
    0.8, 0.8,
]);

const vertexBufferLayout = {
    arrayStride: 8,
    attributes: [{
        format: "float32x2",
        offset: 0,
        shaderLocation: 0,
    }]
};

const vertexBuffer = device.createBuffer({
    label: "vertex buffer",
    size: vertices.byteLength,
    usage:
        GPUBufferUsage.VERTEX |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.STORAGE,
});

device.queue.writeBuffer(vertexBuffer, 0, vertices);

// BIND GROUPS, bunch of data bound together
const bindGroupLayout = device.createBindGroupLayout({
    label: "basic bind group layout",
    entries: [{
        binding: 0,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: "read-only-storage" },
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

// SHADER, code that runs on GPU
const shaders = device.createShaderModule(shaderSrc);

// PIPELINE, specifies the bindings and modules to the GPU
const pipelineLayout = device.createPipelineLayout({
    label: "basic pipeline layout",
    bindGroupLayouts: [bindGroupLayout],
});

const pipeline = device.createRenderPipeline({
    label: "main pipeline",
    vertex: {
        module: shaders,
        entryPoint: "vertexMain",
        buffers: [vertexBufferLayout],
    },
    fragment: {
        module: shaders,
        entryPoint: "fragmentMain",
        targets: [{ format: format }],
    },
    primitive: { topology: "triangle-list" },
    layout: pipelineLayout,
});

// ENCODER, queues instructions to the GPU via command buffer
const commandEncoder = device.createCommandEncoder();

// RENDER PASS, one sequence of instructions
const renderPass = commandEncoder.beginRenderPass({
    label: "pass encoder",
    colorAttachments: [{
        view: texture.createView(), // to view the texture
        loadOp: "clear", // better for performance than load.
        clearValue: { r: 0, g: 0, b: 0.4, a: 1.0 }, // what is empty
        storeOp: "store", // try discard later
    }],
});

renderPass.setPipeline(pipeline);
renderPass.setVertexBuffer(0, vertexBuffer);
renderPass.setBindGroup(0, bindGroup);
renderPass.draw(3, 1);
renderPass.end();

device.queue.submit([commandEncoder.finish()]); 