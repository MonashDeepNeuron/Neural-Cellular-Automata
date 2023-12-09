// Imports
import setupManager from "./managers/setup.js";
import shaderManager from "./managers/shader.js";
import bufferManager from "./managers/buffer.js"
import bindGroupManager from "./managers/bindGroup.js"
import pipelineManager from "./managers/pipeline.js"
import renderManager from "./managers/render.js"

// Broad Overview
// Canvas, Device, Texture, 
// Buffer, Bind Groups, Shaders, (this part might be different between models)
// Pipeline, Encoder, Pass

// SETUP, not sure what most terms are
const device = await setupManager.device();
const format = navigator.gpu.getPreferredCanvasFormat();
const context = setupManager.context({ device: device, format: format });

// TEXTURE, buffer with special operations (good for 2D and 3D data)
const texture = context.getCurrentTexture();

// BUFFER, binary data, sent to GPU
const vertices = bufferManager.triangle;
const vertexBufferLayout = bufferManager.layout;
const vertexBufferDescriptor = bufferManager.createDescriptor({
    size: vertices.byteLength
});
const vertexBuffer = device.createBuffer(vertexBufferDescriptor);
device.queue.writeBuffer(vertexBuffer, 0, vertices);

// BIND GROUP, bunch of data bound together
const bindGroupLayout = device.createBindGroupLayout(bindGroupManager.layoutDescriptor);
const bindGroupDescriptor = bindGroupManager.createDescriptor({
    layout: bindGroupLayout,
    buffer: vertexBuffer
});
const bindGroup = device.createBindGroup(bindGroupDescriptor);

// SHADER, GPU code 
const shaders = {
    vertex: device.createShaderModule(shaderManager.vertexShader),
    fragment: device.createShaderModule(shaderManager.fragmentShader),
};

// PIPELINE, specifies location of everything to GPU
const pipelineLayoutDescriptor = pipelineManager.createLayoutDescriptor({ bindGroupLayout: bindGroupLayout });
const pipelineLayout = device.createPipelineLayout(pipelineLayoutDescriptor);

const pipelineDescriptor = pipelineManager.createDescriptor({
    shaders: shaders,
    vertexBufferLayout: vertexBufferLayout,
    format: format,
    layout: pipelineLayout,
});
const pipeline = device.createRenderPipeline(pipelineDescriptor);

// ENCODER, queues instructions to the GPU via command buffer
const commandEncoder = device.createCommandEncoder();

// RENDER PASS, one sequence of instructions
const renderPassDescriptor = renderManager.createDescriptor({ texture: texture })
const renderPass = commandEncoder.beginRenderPass(renderPassDescriptor);
renderPass.setPipeline(pipeline);
renderPass.setVertexBuffer(0, vertexBuffer);
renderPass.setBindGroup(0, bindGroup);
renderPass.draw(vertices.length / 2, 1);
renderPass.end();

device.queue.submit([commandEncoder.finish()]); 