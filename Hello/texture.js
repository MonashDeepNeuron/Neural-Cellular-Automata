function fail(msg) {
    // eslint-disable-next-line no-alert
    alert(msg);
}

const adapter = await navigator.gpu?.requestAdapter();
const device = await adapter?.requestDevice();

// Get a WebGPU context from the canvas and configure it
const canvas = document.querySelector('canvas');
const context = canvas.getContext('webgpu');
const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
context.configure({
    device,
    format: presentationFormat,
});

const module = device.createShaderModule({
    label: 'our hardcoded textured quad shaders',
    code:
        /*wgsl*/`
        struct OurVertexShaderOutput {
            @builtin(position) position: vec4f,
            @location(0) texcoord: vec2f,
        };

        @vertex fn vs(
            @builtin(vertex_index) vertexIndex : u32
        ) -> OurVertexShaderOutput {
            let pos = array(
            // 1st triangle
            vec2f( 0.0,  0.0),  // center
            vec2f( 1.0,  0.0),  // right, center
            vec2f( 0.0,  1.0),  // center, top

            // 2st triangle
            vec2f( 0.0,  1.0),  // center, top
            vec2f( 1.0,  0.0),  // right, center
            vec2f( 1.0,  1.0),  // right, top
            );

            var vsOutput: OurVertexShaderOutput;
            let xy = pos[vertexIndex];
            vsOutput.position = vec4f(xy, 0.0, 1.0);
            vsOutput.texcoord = xy;
            return vsOutput;
        }

        @group(0) @binding(0) var ourSampler: sampler;
        @group(0) @binding(1) var ourTexture: texture_2d<f32>;

        @fragment fn fs(fsInput: OurVertexShaderOutput) -> @location(0) vec4f {
            return textureSample(ourTexture, ourSampler, fsInput.texcoord);
        }
        `,
});

const pipeline = device.createRenderPipeline({
    label: 'hardcoded textured quad pipeline',
    layout: 'auto',
    vertex: {
        module,
        entryPoint: 'vs',
    },
    fragment: {
        module,
        entryPoint: 'fs',
        targets: [{ format: presentationFormat }],
    },
});

const response = await fetch('./doge.png');
const dogeBitmap = await createImageBitmap(await response.blob());

const texture = device.createTexture({
    label: 'doge texture',
    size: [dogeBitmap.width, dogeBitmap.height],
    format: 'rgba8unorm',
    usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
});

device.queue.copyExternalImageToTexture(
    { source: dogeBitmap, flipY: true },
    { texture: texture },
    [dogeBitmap.width, dogeBitmap.height]
);

const sampler = device.createSampler();

const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
        { binding: 0, resource: sampler },
        { binding: 1, resource: texture.createView() },
    ],
});

const renderPassDescriptor = {
    label: 'our basic canvas renderPass',
    colorAttachments: [
        {
            // view: <- to be filled out when we render
            clearValue: [0.3, 0.3, 0.3, 1],
            loadOp: 'clear',
            storeOp: 'store',
        },
    ],
};

function render() {
    // Get the current texture from the canvas context and
    // set it as the texture to render to.
    renderPassDescriptor.colorAttachments[0].view =
        context.getCurrentTexture().createView();

    const encoder = device.createCommandEncoder({
        label: 'render quad encoder',
    });
    const pass = encoder.beginRenderPass(renderPassDescriptor);
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(6);  // call our vertex shader 6 times
    pass.end();

    const commandBuffer = encoder.finish();
    device.queue.submit([commandBuffer]);
}

const observer = new ResizeObserver(entries => {
    for (const entry of entries) {
        const canvas = entry.target;
        const width = entry.contentBoxSize[0].inlineSize;
        const height = entry.contentBoxSize[0].blockSize;
        canvas.width = Math.max(1, Math.min(width, device.limits.maxTextureDimension2D));
        canvas.height = Math.max(1, Math.min(height, device.limits.maxTextureDimension2D));
        // re-render
        render();
    }
});
observer.observe(canvas);