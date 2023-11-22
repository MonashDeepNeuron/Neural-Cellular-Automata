// synopsis: canvas, device, texture, buffer, bindings, pipeline, encoder
// SET UP 

// CANVAS; the thing on the html page
const canvas = document.querySelector("canvas");

// DEVICE; how to talk about/to the GPU
const adapter = await navigator.gpu.requestAdapter(); // returns 
const device = await adapter.requestDevice();

// i don't know what this is. canvas related
const context = canvas.getContext("webgpu"); // this is where we specify that its webGPU
const format = navigator.gpu.getPreferredCanvasFormat(); // idk bruh
context.configure({
    device: device,
    format: format,
});
// TEXTURE; buffer with special operations (good for 2D and 3D data)
const texture = context.getCurrentTexture(); // basic

// BUFFER; raw binary data that gets sent to the gpu

// write predetermined vertices to buffer
// declare information that will be written
const vertices = new Float32Array([ // whats with this new shit?
    // X,    Y,
    -0.8, -0.8, // Triangle 1
    -0.8, 0.8,
    0.8, 0.8,
]);

// create (initialise) a buffer, its empty
const vertexBuffer = device.createBuffer({
    label: "a triangle",
    size: vertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST, // vertex buffers are kinda special
});

// put the binary information into the buffer
device.queue.writeBuffer(vertexBuffer, 0, vertices); // dst, offset, src

// BIND GROUPS; instruct the GPU how to interpret things in the buffer

// they "binds" the binary data in the buffer to a data type
// note: bind groups also have other purposes (e.g. textures?? and samplers??)
const bindGroupLayout = device.createBindGroupLayout({
    label: "basic bind group layout",
    entries: [{
        binding: 0,
        visibility: GPUShaderStage.VERTEX, // the gpu sees it at different stages depending on flag
        buffer: {},
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

// SHADER, WILL BE REMOVED WHEN THE PROGRAM WORKS
const shaders = device.createShaderModule({
    label: "shaders",
    code:
        /*wgsl*/`
        @vertex
        fn vertexMain(@location(0) pos: vec2f) ->
          @builtin(position) vec4f {
          return vec4f(pos, 0, 1);
        }

        @fragment
        fn fragmentMain() -> @location(0) vec4f {
            return vec4f(1, 0, 0, 1); // (Red, Green, Blue, Alpha)
        }`,
});


// PIPELINE CREATION; specifies the bindings and modules to the GPU
/*
const pipelineLayout = device.createPipelineLayout({
    label: "basic layout",
    entries: { bindGroupLayouts: [bindGroupLayout] }
});
*/
const pipeline = device.createRenderPipeline({
    label: "main pipeline",
    layout: "auto",
    vertex: { // FILL THIS IN
        module: shaders,
        entryPoint: "vertexMain", // not this
        buffer: 0, // not this(?)
    },
    fragment: {
        module: shaders,
        entryPoint: "fragmentMain", // not this
        targets: [{ format: format }], // not this(?)
    }
}); // p

// ENCODER; queues instructions to the GPU via command buffer
// command buffer != buffer 

const commandEncoder = device.createCommandEncoder();
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
renderPass.setBindGroup(0, bindGroup)
renderPass.draw(3, 1);
renderPass.end();

device.queue.submit([commandEncoder.finish()]);


/*
setup
// Device, Canvas, Context // setupManager (might not be a class)
// Textures, Buffers, Bindings // Resource Manager
// Pipeline // PipelineManager


// all event manager things
detect play click -> starts looping (with set interval() )
detect pause click -> stops the loop (actually fully stopped) (clearInterval() )
detect one frame button -> executes oneFrame() 
detect rule string submission -> write to buffer
speed -> changes interval

// maybe the encoder manager handles this
oneFrame(a,b,c) { 
    encoder
    render pass
    submit
    return Null
}; 
*/

