import { shaderSrc } from "./shaders.js";
import * as initialise from "./functions.js";
import { performRenderPass } from "./other.js";

// CANVAS; the thing on the html page
const canvas = document.querySelector("canvas");

// DEVICE; how to talk about/to the GPU
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();

// auxiliary setup
const context = canvas.getContext("webgpu");
const format = navigator.gpu.getPreferredCanvasFormat();
context.configure({
    device: device,
    format: format,
});

// TEXTURE, buffer with special operations (good for 2D and 3D data)
const texture = context.getCurrentTexture();

// BUFFER, raw binary data that gets sent to the gpu
const vertices = initialise.vertices;
const vertexBufferLayout = initialise.vertexBufferLayout();
const vertexBuffer = initialise.vertexBuffer(device, vertices);
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
performRenderPass(renderPass, pipeline, vertexBuffer, bindGroup)

device.queue.submit([commandEncoder.finish()]); // THIS IS INVALID?