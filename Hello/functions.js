export function bindGroupLayout(device) {
    const bindGroupLayout = device.createBindGroupLayout({
        label: "basic bind group layout",
        entries: [{
            binding: 0,
            visibility: GPUShaderStage.VERTEX,
            buffer: { type: "read-only-storage" },
        }],
    });
    return bindGroupLayout;
};

export function bindGroup(device, bindGroupLayout, vertexBuffer) {
    const bindGroup = device.createBindGroup({
        label: "basic bind group thing",
        layout: bindGroupLayout,
        entries: [{
            binding: 0,
            resource: { buffer: vertexBuffer },
        }], // comma for consistency i guess
    });

    return bindGroup
}

export function pipelineLayout(device, bindGroupLayout) {
    const pipelineLayout = device.createPipelineLayout({
        label: "basic pipeline layout",
        bindGroupLayouts: [bindGroupLayout],
    });

    return pipelineLayout
}

export function pipeline(device, format, shaders, vertexBufferLayout, pipelineLayout) {
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
    return pipeline
}

export function renderPass(commandEncoder, texture) {
    const renderPass = commandEncoder.beginRenderPass({
        label: "pass encoder",
        colorAttachments: [{
            view: texture.createView(), // to view the texture
            loadOp: "clear", // better for performance than load.
            clearValue: { r: 0, g: 0, b: 0.4, a: 1.0 }, // what is empty
            storeOp: "store", // try discard later
        }],
    });
    return renderPass;
}