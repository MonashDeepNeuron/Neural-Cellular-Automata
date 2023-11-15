// import shaders
import { guiShaderWGSL } from "http://localhost:5500/CAs/LifeLike/guiShader.js";
import { computeShaderWGSL } from "http://localhost:5500/CAs/LifeLike/computeShader.js";
// SET VARIABLES
const GRID_SIZE = 16;
const UPDATE_INTERVAL = 200;
const WORKGROUP_SIZE = 8;

/** Life like CA rulestring using Survival/Birth notation */
const RULESTRING = "/2";

/**  Number of possible neighbour states: 0-8*/
const POSSIBLE_NEIGHBOURS = 9;

const INITIAL_STATE = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1,
    0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0,
    0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

export default async function main() {
    const RULE = parseRulestring(RULESTRING);
    console.log(RULE);

    let step = 0

    // DEVICE SETUP - could prob be  a function. yes please make this a function
    const canvas = document.querySelector("canvas");
    if (!navigator.gpu) {
        throw new Error("WebGPU not supported on this browser");
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        throw new Error("No appropriate GPUAdapter found.");
    }

    const device = await adapter.requestDevice();

    // Uniform grid
    const uniformArray = new Float32Array([GRID_SIZE, GRID_SIZE]);
    const uniformBuffer = device.createBuffer({
        label: "Grid Uniforms",
        size: uniformArray.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    device.queue.writeBuffer(uniformBuffer, 0, uniformArray);

    // Cell state arrays
    const cellStateArray = new Uint32Array(GRID_SIZE * GRID_SIZE);
    const cellStateStorage = [
        device.createBuffer({
            label: "Cell State A",
            size: cellStateArray.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        }),

        device.createBuffer({
            label: "Cell State B",
            size: cellStateArray.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        })
    ];

    // write to buffer A
    for (let i = 0; i < cellStateArray.length; ++i) {
        // cellStateArray[i] = Math.random() > 0.6 ? 1 : 0;
        cellStateArray[i] = INITIAL_STATE[i];
    }
    device.queue.writeBuffer(cellStateStorage[0], 0, cellStateArray);

    // write to buffer B
    for (let i = 0; i < cellStateArray.length; i++) {
        cellStateArray[i] = i % 2;
    }
    device.queue.writeBuffer(cellStateStorage[1], 0, cellStateArray);


    function rules(rule) {
        console.log("rules")
        const ruleArray = new Uint32Array(rule.length * rule[0].length);
        const ruleStorage = device.createBuffer({
            label: "Rule Storage",
            size: ruleArray.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        for (let i = 0; i < rule.length; ++i) {
            for (let j = 0; j < rule[i].length; j++) {
                ruleArray[i * POSSIBLE_NEIGHBOURS + j] = rule[i][j];
            }
        }

        return { ruleArray, ruleStorage };
    }

    const { ruleArray, ruleStorage } = rules(RULE);
    console.log(ruleArray)
    device.queue.writeBuffer(ruleStorage, 0, ruleArray);

    // FORMAT CANVAS
    const context = canvas.getContext("webgpu");
    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
        device: device,
        format: canvasFormat,
    });

    // VERTEX SETUP for a square
    const vertices = new Float32Array([
        // X,    Y,
        -0.8, -0.8, // Triangle 1
        -0.8, 0.8,
        0.8, 0.8,

        0.8, 0.8, // Triangle 2
        0.8, -0.8,
        -0.8, -0.8,
    ]);

    const vertexBuffer = device.createBuffer({
        label: "Cell vertices", // Error message label
        size: vertices.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });

    device.queue.writeBuffer(vertexBuffer, /*bufferOffset=*/0, vertices);

    const vertexBufferLayout = {
        arrayStride: 8, // 32bit = 4 bytes, 4x2 = 8 bytes to skip to find next vertex
        attributes: [{
            format: "float32x2", // two 32 bit floats per vertex
            offset: 0,
            shaderLocation: 0, // Position, see vertex shader
        }]
    }

    // COMPUTE SHADER MODULE
    const simulationShaderModule = device.createShaderModule(computeShaderWGSL);

    // CELL SHADER MODULE
    const cellShaderModule = device.createShaderModule(guiShaderWGSL);


    // COMPUTE SHADER RESOURCE BINDING LAYOUT
    const bindGroupLayout = device.createBindGroupLayout({
        label: "Cell Bind Group Layout",
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT,
                buffer: {} // Grid uniform buffer
            },

            {
                binding: 1,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT,
                buffer: { type: "read-only-storage" } // Cell state input buffer
            },

            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "storage" } // Cell state output buffer
            },

            {
                binding: 3,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "read-only-storage" } // Ruleset
            }
        ]
    });


    // PIPE LAYOUT
    const pipelineLayout = device.createPipelineLayout({
        label: "Cell Pipeline Layout",
        bindGroupLayouts: [bindGroupLayout],
    });

    // SIMULATION PIPELINE
    const simulationPipeline = device.createComputePipeline({
        label: "Simulation pipeline",
        layout: pipelineLayout,
        compute: {
            module: simulationShaderModule,
            entryPoint: "computeMain",
        }
    });

    //RENDER PIPELINE            
    const cellPipeline = device.createRenderPipeline({
        label: "Cell pipeline",
        layout: pipelineLayout,
        vertex: {
            module: cellShaderModule,
            entryPoint: "vertexMain",
            buffers: [vertexBufferLayout],
        },
        fragment: {
            module: cellShaderModule,
            entryPoint: "fragmentMain",
            targets: [{
                format: canvasFormat
            }],
        }
    });

    function createBindGroup(label, uniformBuffer, cellStateA, cellStateB, ruleStorage) {
        return device.createBindGroup({
            label: label,
            layout: cellPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: uniformBuffer } },
                { binding: 1, resource: { buffer: cellStateA } },
                { binding: 2, resource: { buffer: cellStateB } },
                { binding: 3, resource: { buffer: ruleStorage } },
            ],
        });
    }


    let bindGroups = [
        createBindGroup("Cell renderer bind group A", uniformBuffer, cellStateStorage[0], cellStateStorage[1], ruleStorage),
        createBindGroup("Cell render bind group B", uniformBuffer, cellStateStorage[1], cellStateStorage[0], ruleStorage)
    ];


    // INITIAL CANVAS SETUP
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
        colorAttachments:
            [{
                view: context.getCurrentTexture().createView(),
                loadOp: "clear",
                clearValue: { r: 0, g: 0, b: 0, a: 1 }, // New line
                storeOp: "store",
            }]
    });

    // Draw the features
    pass.setPipeline(cellPipeline);
    pass.setVertexBuffer(0, vertexBuffer);
    pass.setBindGroup(0, bindGroups[step % 2]);
    pass.draw(vertices.length / 2, GRID_SIZE * GRID_SIZE); // 6 vertices, 12 floats
    pass.end();

    // Finish the command buffer and immediately submit it.
    device.queue.submit([encoder.finish()]);


    // Called at set interval as callback and on keyPressed
    function updateLoop() {

        // Only permitted to run if one frame is wanted or
        if (!oneFrame) {
            if (!running) { return; }
            // Continue if running = true
        }
        else {
            oneFrame = false; // Cross-script variable, do not add let,var or const
        }

        if (newRuleString) {
            const { ruleArray, ruleStorage } = rules(parseRulestring(ruleString))
            console.log(ruleArray)
            device.queue.writeBuffer(ruleStorage, 0, ruleArray);
            bindGroups = [
                createBindGroup("Cell renderer bind group A", uniformBuffer, cellStateStorage[0], cellStateStorage[1], ruleStorage),
                createBindGroup("Cell render bind group B", uniformBuffer, cellStateStorage[1], cellStateStorage[0], ruleStorage)
            ];
            newRuleString = false // toggle off

        }


        const encoder = device.createCommandEncoder();

        // CREATE COMPUTE TOOL & PERFORM COMPUTATION TASKS
        const computePass = encoder.beginComputePass();

        computePass.setPipeline(simulationPipeline);
        computePass.setBindGroup(0, bindGroups[step % 2]);

        const workgroupCount = Math.ceil(GRID_SIZE / WORKGROUP_SIZE);
        computePass.dispatchWorkgroups(workgroupCount, workgroupCount);

        computePass.end();

        // CREATE DRAW TOOL & SET DEFAULT COLOR (BACKGROUND COLOR)
        step++;
        const pass = encoder.beginRenderPass({
            colorAttachments:
                [{
                    view: context.getCurrentTexture().createView(),
                    loadOp: "clear",
                    clearValue: { r: 0, g: 0, b: 0, a: 1 }, // New line
                    storeOp: "store",
                }]
        });

        // DRAW THE FEATURES
        pass.setPipeline(cellPipeline);
        pass.setVertexBuffer(0, vertexBuffer);

        pass.setBindGroup(0, bindGroups[step % 2]);

        pass.draw(vertices.length / 2, GRID_SIZE * GRID_SIZE); // 6 vertices, 12 floats

        pass.end();


        // Finish the command buffer and immediately submit it.
        device.queue.submit([encoder.finish()]);
    }

    setInterval(updateLoop, UPDATE_INTERVAL);
    forcedUpdate = updateLoop;
    // Cross-script variable, enables other scripts to force an update cylce
}


function parseRulestring(rulestring) {
    // Output structure:
    // [
    //    SURVIVE[number of neighbours] = 1/0 (true/false),
    //    BIRTH  [number of neighbours] = 1/0 (true/false)
    // ]
    // Where length of each SURVIVE and BIRTH is 9 (0-8)
    // Default fill all conditions as False then parse rule string

    const RULE = [Array(POSSIBLE_NEIGHBOURS).fill(0), Array(POSSIBLE_NEIGHBOURS).fill(0)];

    let slashFound = false;
    // Parse rulestring. digits before slash indicate when a cell should 
    // survive and digits after indicate when a cell should be born
    for (let i = 0; i < rulestring.length; i++) {
        if (rulestring.charAt(i) == '/') {
            slashFound = true;
        }
        else {
            const x = parseInt(rulestring.charAt(i));
            if (Number.isNaN(x)) {
                alert("ERROR: Invalid Rulestring");
                return;
            }
            else {
                if (!slashFound) {
                    RULE[0][x] = 1;
                    // More vigourous validity checks could be used
                    // eg. is this index already filled as true?
                }
                else {
                    RULE[1][x] = 1;
                    // More vigourous validity checks could be used
                    // eg. is this index already filled as true?
                }
            }
        }
    }

    return RULE;
}
