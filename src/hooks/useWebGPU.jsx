import { useEffect, useRef } from "react";
import BufferManager from "../managers/BufferManager";
import { guiShader } from "../shaders/guiShader";
import { computeShader } from "../shaders/computeShader";
import { parseRuleString } from '../util/Parse'

const SQUARE_VERTICIES = new Float32Array([
    // X,    Y,
    -0.8, -0.8, // Triangle 1
    -0.8, 0.8,
    0.8, 0.8,

    0.8, 0.8, // Triangle 2
    0.8, -0.8,
    -0.8, -0.8,
]);

export default function useWebGPU(canvasRef, settings) {
    const {
        workgroupSize,
        initialState,
        gridSize,
        ruleString
    } = settings;

    const deviceRef = useRef(null);
    const contextRef = useRef(null);
    const pipelinesRef = useRef(null);
    const buffersRef = useRef(null);

    useEffect(() => {
        const initializeWebGPU = async () => {
            if (!canvasRef.current) return;

            // Check if WebGPU is supported
            if (!navigator.gpu) {
                console.error("WebGPU not supported on this browser.");
                return;
            }

            try {


                /**
                * INIITIALISING GPU CONNECTION: DEVICE, CONTEXT
                */


                // Request GPU adapter and device
                const adapter = await navigator.gpu.requestAdapter();
                if (!adapter) {
                    console.error("No suitable GPUAdapter found.");
                    return;
                }
                const device = await adapter.requestDevice();
                deviceRef.current = device;

                // Configure the canvas context
                const canvas = canvasRef.current;
                const context = canvas.getContext("webgpu");
                const canvasFormat = navigator.gpu.getPreferredCanvasFormat();

                context.configure({
                    device,
                    format: canvasFormat,
                });
                contextRef.current = context;

                console.log("WebGPU initialized");



                /**
                 * INIITIALISING WEBGPU RESOURCES: BUFFERS AND PIPELINES
                 */




                const { vertexBuffer, vertexBufferLayout } = BufferManager.loadShapeVertexBuffer(device, SQUARE_VERTICIES);

                // Create shaders
                const cellShaderModule = device.createShaderModule({
                    label: 'shader that draws',
                    code: guiShader
                });

                const simulationShaderModule = device.createShaderModule({
                    label: 'shader that computes next state',
                    code: computeShader,
                    constants: { WORKGROUP_SIZE: workgroupSize }
                });

                // Create pipeline layout and bind group layout
                const bindGroupLayout = BufferManager.createBindGroupLayout(device);
                const pipelineLayout = device.createPipelineLayout({
                    bindGroupLayouts: [bindGroupLayout],
                });

                // Create render and compute pipelines
                const renderPipeline = device.createRenderPipeline({
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

                const simulationPipeline = device.createComputePipeline({
                    label: "Simulation pipeline",
                    layout: pipelineLayout,
                    compute: {
                        module: simulationShaderModule,
                        entryPoint: "computeMain",
                    }
                });


                // Initialize buffers and bind groups
                const { bindGroups, uniformBuffer, cellStateStorage, ruleStorage } = BufferManager.initialiseComputeBindgroups(
                    device,
                    renderPipeline,
                    gridSize,
                    initialState,
                    parseRuleString(ruleString)
                );

                pipelinesRef.current = { renderPipeline, simulationPipeline };
                buffersRef.current = { bindGroups, uniformBuffer, cellStateStorage, ruleStorage };

                console.log('configured gpu resources', pipelinesRef.current, buffersRef.current);
                console.log('configured device and context', contextRef.current, deviceRef.current);

            } catch (error) {
                console.error("Failed to initialize WebGPU:", error);
            }
        };

        initializeWebGPU();

        return () => {
            // Cleanup (if needed)
            deviceRef.current = null;
            contextRef.current = null;
            pipelinesRef.current = null;
            buffersRef.current = null;
        };
    }, [canvasRef]);
    console.log('about to return');
    return { device: deviceRef.current, context: contextRef.current };
};
