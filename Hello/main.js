// Imports
import { shaderSrc } from "./shaders.js";
import * as initialise from "./functions.js";
import { triangleBuffer as buffer } from "./buffer.js";
import { bindGroup as bGroup } from "./bindGroup.js";

// Broad Overview
// Canvas, Device, Texture, 
// Buffer, Bind Groups, Shaders, (this part might be different between models)
// Pipeline, Encoder, Pass

// SETUP
const canvas = document.querySelector("canvas");
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();
const format = navigator.gpu.getPreferredCanvasFormat();
const context = initialise.context(canvas, device, format);

// TEXTURE, buffer with special operations (good for 2D and 3D data)
const texture = context.getCurrentTexture();

// BUFFER, raw binary data that gets sent to the gpu
const vertexBuffer = device.createBuffer(buffer.descriptor);
device.queue.writeBuffer(vertexBuffer, 0, buffer.vertices);

// BIND GROUPS, bunch of data bound together
const bindGroupLayout = device.createBindGroupLayout(bGroup.layoutDescriptor);
bGroup.descriptor.layout = bindGroupLayout;
bGroup.descriptor.entries[0].resource.buffer = vertexBuffer;
const bindGroup = device.createBindGroup(bGroup.descriptor);

// SHADER, code that runs on GPU
const shaders = device.createShaderModule(shaderSrc);

// PIPELINE, specifies the bindings and modules to the GPU
const pipelineLayout = initialise.pipelineLayout(device, bindGroupLayout);
const pipeline = initialise.pipeline(device, format, shaders, buffer.layout, pipelineLayout);

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


/*
idea:
1. just leave the dependent things as null in the classes
2. set the final details

i want to make it

update descriptors
execute device methods
repeat


idea 2:
passing device as an argument so the static methods can use it
*/