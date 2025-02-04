import { type RefObject, useEffect, useRef } from 'react';
import BufferManager from '../managers/BufferManager';
import startingPatterns from '../patterns/startingPatterns';
import { computeShader } from '../shaders/computeShader';
import { guiShader } from '../shaders/guiShader';
import { parseRuleString } from '../util/Parse';
import useTypedSelector from './useTypedSelector';

const SQUARE_VERTICIES = new Float32Array([-0.8, -0.8, -0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, -0.8, -0.8, -0.8]);

export interface WebGPUSettings {
	workgroupSize: number;
	gridSize: number;
	ruleString: string;
}

export interface WebGPUResources {
	device: GPUDevice;
	context: GPUCanvasContext;
	pipelines: {
		renderPipeline: GPURenderPipeline;
		simulationPipeline: GPUComputePipeline;
	};
	buffers: {
		bindGroups: GPUBindGroup[];
		uniformBuffer: GPUBuffer;
		cellStateStorage: GPUBuffer[];
		ruleStorage: GPUBuffer;
		vertexBuffer: GPUBuffer;
	};
}

export default function useWebGPU(canvasRef: RefObject<HTMLCanvasElement | null>, settings: WebGPUSettings) {
	const { workgroupSize, gridSize, ruleString } = settings;
	const template = useTypedSelector(state => state.webGPU.template);
	const initialState = startingPatterns[template];
	const resources = useRef<WebGPUResources>(null);

	// biome-ignore lint/correctness/useExhaustiveDependencies: test missing dependencies
	useEffect(() => {
		const initializeWebGPU = async () => {
			if (!canvasRef.current) return;

			// Check if WebGPU is supported
			if (!navigator.gpu) {
				console.error('WebGPU not supported on this browser.');
				return;
			}

			try {
				/**
				 * Step 1: Initialize GPU Connection (Device and Context)
				 */
				const adapter = await navigator.gpu.requestAdapter();
				if (!adapter) {
					console.error('No suitable GPUAdapter found.');
					return;
				}

				const device = await adapter.requestDevice();

				const canvas = canvasRef.current;
				const context = canvas.getContext('webgpu');
				const canvasFormat = navigator.gpu.getPreferredCanvasFormat();

				if (!context) {
					console.error('No context found.');
					return;
				}

				context.configure({
					device,
					format: canvasFormat
				});

				console.log('WebGPU initialized');

				/**
				 * Step 2: Initialize GPU Resources (Pipelines and Buffers)
				 */
				const bufferManager = new BufferManager(device);
				const { vertexBuffer, vertexBufferLayout } = bufferManager.loadShapeVertexBuffer(SQUARE_VERTICIES);

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
				const bindGroupLayout = bufferManager.createBindGroupLayout();
				const pipelineLayout = device.createPipelineLayout({
					bindGroupLayouts: [bindGroupLayout]
				});

				// Create render and compute pipelines
				const renderPipeline = device.createRenderPipeline({
					label: 'Cell pipeline',
					layout: pipelineLayout,
					vertex: {
						module: cellShaderModule,
						entryPoint: 'vertexMain',
						buffers: [vertexBufferLayout]
					},
					fragment: {
						module: cellShaderModule,
						entryPoint: 'fragmentMain',
						targets: [{ format: canvasFormat }]
					}
				});

				const simulationPipeline = device.createComputePipeline({
					label: 'Simulation pipeline',
					layout: pipelineLayout,
					compute: {
						module: simulationShaderModule,
						entryPoint: 'computeMain'
					}
				});

				console.log('Configured GPU resources');

				// Initialize buffers and bind groups
				const { bindGroups, uniformBuffer, cellStateStorage, ruleStorage } = bufferManager.initialiseComputeBindgroups(
					renderPipeline,
					gridSize,
					initialState,
					parseRuleString(ruleString)
				);

				/**
				 * Step 3: Update refs to point at WebGPU state
				 */
				resources.current = {
					device,
					context,
					pipelines: {
						renderPipeline,
						simulationPipeline
					},
					buffers: {
						bindGroups,
						uniformBuffer,
						cellStateStorage,
						ruleStorage,
						vertexBuffer
					}
				};
			} catch (error) {
				console.error('Failed to initialize WebGPU:', error);
			}
		};

		initializeWebGPU();

		return () => {
			resources.current = null;
		};
	}, [template]);
	return resources.current;
}
