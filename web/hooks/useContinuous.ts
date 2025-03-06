'use client';

import cell from '@/shaders/continuous/cell';
import { useEffect, useRef, useState } from 'react';
import { CAStatus } from './useNCA';

export interface GPUResources {
	device: GPUDevice;
	context: GPUCanvasContext;
	bindGroups: CellStateBindGroupPair;
	pipelines: {
		cell: GPURenderPipeline;
		simulation: GPUComputePipeline;
	};
	buffers: {
		vertex: GPUBuffer;
	};
}

export interface ContinuousSettings {
	size: number;
	shaders: {
		simulation: string;
	};
}

export type CellStateBufferPair = [GPUBuffer, GPUBuffer];
export type CellStateBindGroupPair = [GPUBindGroup, GPUBindGroup];

const SHAPE_VERTICES = new Float32Array([-1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1]);
const WORKGROUP_SIZE = 8;

export default function useContinuous({ size, shaders }: ContinuousSettings) {
	const [status, setStatus] = useState(CAStatus.ALLOCATING_RESOURCES);
	const [error, setError] = useState('');
	const [resources, setResources] = useState<GPUResources | null>(null);
	const [play, setPlay] = useState(true);
	const [step, setStep] = useState(0);
	const [FPS, setFPS] = useState(60);
	const [stepsPerFrame, setStepsPerFrame] = useState(2);

	const canvasRef = useRef<HTMLCanvasElement>(null);

	useEffect(() => {
		async function init() {
			if (resources) return;
			console.log('Initialising Shaders');

			// Secure context check
			if (!window.isSecureContext) {
				setStatus(CAStatus.FAILED);
				setError('WebGPU is not allowed in non-secure contexts.  Please access this website over HTTPS.');
				return;
			}

			// Check for WebGPU support
			if (!navigator.gpu) {
				setStatus(CAStatus.FAILED);
				setError('WebGPU is not supported on this browser.');
				return;
			}

			// Request GPU Adapter
			let adapter: GPUAdapter | null = null;
			try {
				adapter = await navigator.gpu.requestAdapter();
			} catch (error) {
				setStatus(CAStatus.FAILED);
				setError(`Failed to get a GPU adapter: ${(error as Error).message}`);
			}
			if (!adapter) return;

			// Request GPU Device
			let device: GPUDevice | null = null;
			try {
				device = await adapter.requestDevice();
			} catch (error) {
				setStatus(CAStatus.FAILED);
				setError(`Failed to get a GPU device: ${(error as Error).message}`);
				return;
			}

			// Configure canvas context
			const context = canvasRef.current?.getContext('webgpu');
			if (!context) {
				setStatus(CAStatus.FAILED);
				setError('Failed to get canvas context.');
				return;
			}
			context.configure({
				device,
				format: navigator.gpu.getPreferredCanvasFormat(),
				alphaMode: 'opaque'
			});

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
						buffer: { type: 'read-only-storage' }
					},
					{
						binding: 1, // State / Input State (Compute)
						visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
						buffer: { type: 'read-only-storage' }
					},
					{
						binding: 2, // Output State (Compute)
						visibility: GPUShaderStage.COMPUTE,
						buffer: { type: 'storage' }
					},
					{
						binding: 3, // Kernel
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
			const sizeArray = new Uint32Array([size]);
			const sizeBuffer = device.createBuffer({
				label: 'Size Buffer',
				size: sizeArray.byteLength,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
			});

			const cellState = new Float32Array(size * size).fill(0).map(() => Math.random());
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

			const kernelArray = new Float32Array([0.68, -0.9, 0.68, -0.9, -0.66, -0.9, 0.68, -0.9, 0.68]);
			const kernelBuffer = device.createBuffer({
				label: 'Kernel Buffer',
				size: kernelArray.byteLength,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
			});

			// Write buffers
			device.queue.writeBuffer(sizeBuffer, 0, sizeArray);
			device.queue.writeBuffer(cellStateBuffers[0], 0, cellState);
			device.queue.writeBuffer(cellStateBuffers[1], 0, cellState);
			device.queue.writeBuffer(kernelBuffer, 0, kernelArray);

			// Create Bind Group
			const bindGroups: CellStateBindGroupPair = [
				device.createBindGroup({
					label: 'Bind Group A',
					layout: bindGroupLayout,
					entries: [
						{ binding: 0, resource: { buffer: sizeBuffer } },
						{ binding: 1, resource: { buffer: cellStateBuffers[0] } },
						{ binding: 2, resource: { buffer: cellStateBuffers[1] } },
						{ binding: 3, resource: { buffer: kernelBuffer } }
					]
				}),
				device.createBindGroup({
					label: 'Bind Group B',
					layout: bindGroupLayout,
					entries: [
						{ binding: 0, resource: { buffer: sizeBuffer } },
						{ binding: 1, resource: { buffer: cellStateBuffers[1] } },
						{ binding: 2, resource: { buffer: cellStateBuffers[0] } },
						{ binding: 3, resource: { buffer: kernelBuffer } }
					]
				})
			];

			// All done!
			setResources({
				device,
				context,
				bindGroups,
				pipelines: {
					cell: cellPipeline,
					simulation: simulationPipeline
				},
				buffers: {
					vertex: vertexBuffer
				}
			});
			setStatus(CAStatus.READY);
		}

		init();

		// Cleanup
		return () => {
			// Destroy device & associated buffers
			resources?.device.destroy();
		};
	}, [size, shaders.simulation, resources]);

	useEffect(() => {
		if (status !== CAStatus.READY || !resources) return;

		let animationFrameId: number;
		let lastFrameTime = performance.now();

		const renderLoop = () => {
			if (!(play && resources)) {
				animationFrameId = requestAnimationFrame(renderLoop);
				return;
			}

			const now = performance.now();
			const deltaTime = now - lastFrameTime;
			const frameTime = 1000 / FPS;

			if (deltaTime >= frameTime) {
				lastFrameTime = now - (deltaTime % frameTime);

				// Create command encoder
				const encoder = resources.device.createCommandEncoder();
				const textureView = resources.context.getCurrentTexture().createView();

				// Compute Passes
				for (let i = 0; i < stepsPerFrame; i++) {
					const computePass = encoder.beginComputePass();
					computePass.setPipeline(resources.pipelines.simulation);
					computePass.setBindGroup(0, resources.bindGroups[(step + i) % 2]);
					const workgroupCount = Math.ceil(size / WORKGROUP_SIZE);
					computePass.dispatchWorkgroups(workgroupCount, workgroupCount);
					computePass.end();
				}

				// Render pass
				const renderPass = encoder.beginRenderPass({
					colorAttachments: [
						{
							view: textureView,
							clearValue: [1, 1, 1, 1],
							loadOp: 'clear',
							storeOp: 'store'
						}
					]
				});
				renderPass.setPipeline(resources.pipelines.cell);
				renderPass.setVertexBuffer(0, resources.buffers.vertex);
				renderPass.setBindGroup(0, resources.bindGroups[(step + stepsPerFrame - 1) % 2]);
				renderPass.draw(SHAPE_VERTICES.length / 2, size * size);
				renderPass.end();

				// Submit commands
				resources.device.queue.submit([encoder.finish()]);
				setStep(prev => prev + stepsPerFrame);
			}

			animationFrameId = requestAnimationFrame(renderLoop);
		};

		// Start render loop
		animationFrameId = requestAnimationFrame(renderLoop);

		return () => {
			cancelAnimationFrame(animationFrameId);
		};
	}, [play, FPS, resources, status, size, step, stepsPerFrame]);

	return {
		play,
		setPlay,
		step,
		setStep,
		error,
		setError,
		status,
		setStatus,
		FPS,
		setFPS,
		stepsPerFrame,
		setStepsPerFrame,
		canvasRef
	};
}

export type NCAControls = ReturnType<typeof useContinuous>;
