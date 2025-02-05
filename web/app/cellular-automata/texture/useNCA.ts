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
	weightsURL: string;
	shaders: {
		simulation: string;
	};
}

export type CellStateBufferPair = [GPUBuffer, GPUBuffer];
export type CellStateBindGroupPair = [GPUBindGroup, GPUBindGroup];

export default function useNCA({ size, channels, shaders, weightsURL }: NCASettings) {
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

			// Create vertex buffer
			const vertexBuffer = device.createBuffer({
				label: 'Cell vertices', // Error message label
				size: shapeVertices.byteLength,
				usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
			});
			device.queue.writeBuffer(vertexBuffer, 0, shapeVertices);

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
						buffer: { type: 'read-only-storage' }
					},
					{
						binding: 1, // State / Input State (Compute)
						visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
						buffer: { type: 'storage' }
					},
					{
						binding: 2, // Output State (Compute)
						visibility: GPUShaderStage.COMPUTE,
						buffer: { type: 'storage' }
					},
					{
						binding: 3, // Stage 1 Weights
						visibility: GPUShaderStage.COMPUTE,
						buffer: { type: 'read-only-storage' }
					},
					{
						binding: 4, // Stage 1 Biases
						visibility: GPUShaderStage.COMPUTE,
						buffer: { type: 'read-only-storage' }
					},
					{
						binding: 5, // Stage 2 Biases
						visibility: GPUShaderStage.COMPUTE,
						buffer: { type: 'read-only-storage' }
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
			const cellState = new Float32Array(channels * size * size).fill(0);
			const cellStateBuffers: CellStateBufferPair = [
				device.createBuffer({
					label: 'Cell State A',
					size: cellState.byteLength,
					usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
				}),
				device.createBuffer({
					label: 'Cell State B',
					size: cellState.byteLength,
					usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
				})
			];

			// Write buffers

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

	return { play, setPlay, step, setStep, error, setError, status, setStatus, canvasRef };
}
