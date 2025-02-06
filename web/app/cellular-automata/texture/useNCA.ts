'use client';

import cell from '@/shaders/nca/cell';
import loadWeights from '@/util/loadWeights';
import { useEffect, useRef, useState } from 'react';

export enum NCAStatus {
	ALLOCATING_RESOURCES = 'Allocating Resources',
	READY = 'Ready',
	FAILED = 'Failed'
}

export interface GPUResources {
	device: GPUDevice | null;
	context: GPUCanvasContext | null;
	encoder: GPUCommandEncoder | null;
}

export interface NCASettings {
	size: number;
	channels: number;
	hiddenChannels: number;
	convolutions: number;
	weightsURL: string;
	shaders: {
		simulation: string;
	};
}

export type CellStateBufferPair = [GPUBuffer, GPUBuffer];
export type CellStateBindGroupPair = [GPUBindGroup, GPUBindGroup];

// Optimise drawing of square to 1 large triangle that will be constrained to clip space
const SHAPE_VERTICES = new Float32Array([-1, 2, -1, -1, 2, -1]);

export default function useNCA({ size, channels, hiddenChannels, convolutions, shaders, weightsURL }: NCASettings) {
	const [status, setStatus] = useState(NCAStatus.ALLOCATING_RESOURCES);
	const [error, setError] = useState('');
	const [resources, setResources] = useState<GPUResources>({
		device: null,
		context: null,
		encoder: null
	});
	const [play, setPlay] = useState(false);
	const [step, setStep] = useState(0);

	const canvasRef = useRef<HTMLCanvasElement>(null);

	// biome-ignore lint/correctness/useExhaustiveDependencies: depending upon `resources.device?.destroy` would cause an infinite loop
	useEffect(() => {
		async function init() {
			// Check for WebGPU support
			if (!navigator.gpu) {
				setStatus(NCAStatus.FAILED);
				setError('WebGPU is not supported on this browser.');
			}

			// Request GPU Adapter
			let adapter: GPUAdapter | null = null;
			try {
				adapter = await navigator.gpu.requestAdapter();
			} catch (error) {
				setStatus(NCAStatus.FAILED);
				setError(`Failed to get a GPU adapter: ${(error as Error).message}`);
			}
			if (!adapter) return;

			// Request GPU Device
			let device: GPUDevice | null = null;
			try {
				device = await adapter.requestDevice();
			} catch (error) {
				setStatus(NCAStatus.FAILED);
				setError(`Failed to get a GPU device: ${(error as Error).message}`);
				return;
			}
			setResources(prev => ({ ...prev, device }));

			// Configure canvas context
			const context = canvasRef.current?.getContext('webgpu');
			if (!context) {
				setStatus(NCAStatus.FAILED);
				setError('Failed to get canvas context.');
				return;
			}
			setResources(prev => ({ ...prev, context }));
			context.configure({
				device,
				format: navigator.gpu.getPreferredCanvasFormat(),
				alphaMode: 'premultiplied'
			});

			// Load weights
			let weights: Float32Array | null = null;
			try {
				weights = await loadWeights(weightsURL);
			} catch (error) {
				setStatus(NCAStatus.FAILED);
				setError(`Failed to load weights from URL: ${(error as Error).message}`);
				return;
			}
			const parameters = [channels * convolutions * hiddenChannels, hiddenChannels, hiddenChannels * channels];
			const total = parameters.reduce((acc, p) => acc + p, 0);
			if (total !== weights.length) {
				setStatus(NCAStatus.FAILED);
				setError(`Loaded weights do not match provided model shape. Expected (${parameters.join(', ')})`);
				return;
			}

			// Create vertex buffer
			const vertexBuffer = device.createBuffer({
				label: 'Cell Vertices',
				size: SHAPE_VERTICES.byteLength,
				usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
			});
			device.queue.writeBuffer(vertexBuffer, 0, SHAPE_VERTICES);

			const vertexBufferLayout = {
				arrayStride: 8,
				attributes: [
					{
						format: 'float32x2' as const, // x, y
						offset: 0,
						shaderLocation: 0
					}
				]
			};

			// Create shaders
			const cellShader = device.createShaderModule({
				label: 'Cell Shader',
				code: cell
			});

			const simulationShader = device.createShaderModule({
				label: 'Simulation Shader',
				code: shaders.simulation
			});

			// Create Global Binding Layout
			const bindGroupLayout = device.createBindGroupLayout({
				entries: [
					{
						binding: 0, // Grid size
						visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
						buffer: { type: 'uniform' }
					},
					{
						binding: 1, // State / Input State (Compute)
						visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
						buffer: { type: 'uniform' }
					},
					{
						binding: 2, // Output State (Compute)
						visibility: GPUShaderStage.COMPUTE,
						buffer: { type: 'uniform' }
					},
					{
						binding: 3, // Layer 1 Weights
						visibility: GPUShaderStage.COMPUTE,
						buffer: { type: 'uniform' }
					},
					{
						binding: 4, // Layer 1 Biases
						visibility: GPUShaderStage.COMPUTE,
						buffer: { type: 'uniform' }
					},
					{
						binding: 5, // Layer 2 Weights
						visibility: GPUShaderStage.COMPUTE,
						buffer: { type: 'uniform' }
					}
				]
			});

			// Create pipeline layout
			const pipelineLayout = device.createPipelineLayout({
				label: 'Global Pipeline Layout',
				bindGroupLayouts: [bindGroupLayout]
			});

			// Create pipelines
			const cellPipeline = device.createRenderPipeline({
				label: 'Cell Pipeline',
				layout: pipelineLayout,
				vertex: {
					module: cellShader,
					entryPoint: 'vertex_main',
					buffers: [vertexBufferLayout]
				},
				fragment: {
					module: cellShader,
					entryPoint: 'fragment_main',
					targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }]
				}
			});

			const simulationPipeline = device.createComputePipeline({
				label: 'Simulation Pipeline',
				layout: pipelineLayout,
				compute: {
					module: simulationShader,
					entryPoint: 'compute_main'
				}
			});

			// Initialise buffers
			const shapeArray = new Float32Array([channels, convolutions, hiddenChannels, size]);
			const shapeBuffer = device.createBuffer({
				label: 'Size Buffer',
				size: shapeArray.byteLength,
				usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
			});

			const cellState = new Float32Array(channels * size * size).fill(0);
			const cellStateBuffers: CellStateBufferPair = [
				device.createBuffer({
					label: 'Cell State A',
					size: cellState.byteLength,
					usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
				}),
				device.createBuffer({
					label: 'Cell State B',
					size: cellState.byteLength,
					usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
				})
			];

			const weightBuffers = [
				device.createBuffer({
					label: 'Layer 1 Weights',
					size: parameters[0] * Float32Array.BYTES_PER_ELEMENT,
					usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
				}),
				device.createBuffer({
					label: 'Layer 1 Biases',
					size: parameters[1] * Float32Array.BYTES_PER_ELEMENT,
					usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
				}),
				device.createBuffer({
					label: 'Layer 2 Weights',
					size: parameters[2] * Float32Array.BYTES_PER_ELEMENT,
					usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
				})
			];

			// Write buffers
			device.queue.writeBuffer(shapeBuffer, 0, shapeArray);
			device.queue.writeBuffer(cellStateBuffers[0], 0, cellState);
			device.queue.writeBuffer(cellStateBuffers[1], 0, cellState);
			device.queue.writeBuffer(weightBuffers[0], 0, weights.slice(0, parameters[0]));
			device.queue.writeBuffer(weightBuffers[1], 0, weights.slice(parameters[0], parameters[0] + parameters[1]));
			device.queue.writeBuffer(weightBuffers[2], 0, weights.slice(parameters[0] + parameters[1]));

			// All done!
			setStatus(NCAStatus.READY);
		}

		init();

		// Cleanup
		return () => {
			// Destroy device & buffers
			resources.device?.destroy();
		};
	}, [size, channels]);

	return {
		play,
		setPlay,
		step,
		setStep,
		error,
		setError,
		status,
		setStatus,
		canvasRef
	};
}
