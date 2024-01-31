// Boiler Plate
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();
const format = navigator.gpu.getPreferredCanvasFormat();
const canvas = document.querySelector("canvas");
const context = canvas.getContext("webgpu");
context.configure({
    device: device,
    format: format,
})

// Textures
const response = await fetch('./doge.png');
const dogeBitmap = await createImageBitmap(await response.blob());
const dogeTexture = device.createTexture({
    label: 'doge texture',
    size: [dogeBitmap.width, dogeBitmap.height],
    format: 'rgbaunorm',
    usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
});
device.queue.copyExternalImageToTexture(
    { source: dogeBitmap, flipY: true },
    { texture: dogeTexture },
    [dogeBitmap.width, dogeBitmap.height]
)

// sampler, which is related to texture
const sampler = device.createSampler();

// Bind groups
const bindGroup = device.createBindGroup({
    entries: [
        { binding: 0, resource: dogeTexture.getView() },
        { binding: 1, resource: sampler }
    ]
})

// Shaders
const shaders = {
    vertex: device.createShaderModule({
        label: 'vertex shader',
        code: /*wgsl*/`
            // HERE LIES VERTEX SHADER CODE
            `
    }),
    fragment: device.createShaderModule({
        label: 'fragment shader',
        code: /*wgsl*/`
            // HERE LIES FRAGMENT SHADER CODE
            `
    }),
}

// Pipeline