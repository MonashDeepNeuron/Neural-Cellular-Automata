// Imports
import { shaderSrc } from "./shaders.js";
import * as initialise from "./functions.js";
import { performRenderPass } from "./other.js";
import * as resources from "./resources.js";

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
const vertices = resources.vertices;
const vertexBufferLayout = resources.vertexBufferLayout;
const vertexBufferDescriptor = resources.vertexBufferDescriptor(vertices);
const vertexBuffer = device.createBuffer(vertexBufferDescriptor);
device.queue.writeBuffer(vertexBuffer, 0, vertices);

// BIND GROUPS, bunch of data bound together
const bindGroupLayout = initialise.bindGroupLayout(device);
const bindGroup = initialise.bindGroup(device, bindGroupLayout, vertexBuffer);

// SHADER, code that runs on GPU
const shaders = device.createShaderModule(shaderSrc);

// PIPELINE, specifies the bindings and modules to the GPU
const pipelineLayout = initialise.pipelineLayout(device, bindGroupLayout);
const pipeline = initialise.pipeline(device, format, shaders, vertexBufferLayout, pipelineLayout);

// ENCODER, queues instructions to the GPU via command buffer
const commandEncoder = device.createCommandEncoder();

// RENDER PASS, one sequence of instructions
const renderPass = initialise.renderPass(commandEncoder, texture);

renderPass.setPipeline(pipeline);
renderPass.setVertexBuffer(0, vertexBuffer);
renderPass.setBindGroup(0, bindGroup);
renderPass.draw(3, 1);
renderPass.end();

device.queue.submit([commandEncoder.finish()]); 