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

/**
 * Sets up buffers and pipelines and returns 
 * @param {*} device 
 * @param {*} canvasFormat 
 * @param {*} settings 
 * @returns 
 */

const useGPUResources = (device, context, settings) => {
    const {
        workgroupSize,
        initialState,
        gridSize,
        ruleString
    } = settings;

    const pipelinesRef = useRef(null);
    const buffersRef = useRef(null);

    useEffect(() => {


        console.log('initialising web gpu resources');

        const canvasFormat = navigator.gpu.getPreferredCanvasFormat();

        context.configure({
            device,
            format: canvasFormat,
        });

        const setupResources = () => {
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
        };

        setupResources();

        return () => {
            // Cleanup logic if needed
            pipelinesRef.current = null;
            buffersRef.current = null;
        };
    }, [device, settings]);

    return { pipelines: pipelinesRef.current, buffers: buffersRef.current };
};

export default useGPUResources;
