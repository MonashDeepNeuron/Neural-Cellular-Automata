import { type RefObject, useEffect, useRef, useState } from 'react';
import { useSelector } from 'react-redux';
import BufferManager from '../managers/BufferManager';
import startingPatterns from '../patterns/startingPatterns';
import { computeShader } from '../shaders/computeShader';
import { guiShader } from '../shaders/guiShader';
import { parseRuleString } from '../util/Parse';

const SQUARE_VERTICIES = new Float32Array([-0.8, -0.8, -0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, -0.8, -0.8, -0.8]);

export default function useWebGPU(canvasRef: RefObject<HTMLCanvasElement | null>, settings) {
	const { workgroupSize, gridSize, ruleString } = settings;
	const template = useSelector(state => state.webGPU.template);
	const initialState = startingPatterns[template];

	const [initialised, setInitialised] = useState(false);
	const resources = useRef(null);

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
				const { bindGroups, uniformBuffer, cellStateStorage, ruleStorage } = BufferManager.initialiseComputeBindgroups(
					device,
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
			resources.current = {
				device: null,
				context: null,
				pipelines: null,
				buffers: null
			};
		};
	}, [template]);
	return resources.current;
}
