'use client';

import type { Pattern } from '@/patterns';
import guiShader from '@/shaders/discrete/guiShader';
import { parseRuleString } from '@/util/Parse';
import { useEffect, useRef, useState } from 'react';

export enum LTLStatus {
	ALLOCATING_RESOURCES = 'Allocating Resources',
	READY = 'Ready',
	FAILED = 'Failed'
}

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

export interface LTLSettings {
	size: number;
	pattern: Pattern;
	shaders: {
		simulation: string;
	};
}

export type CellStateBufferPair = [GPUBuffer, GPUBuffer];
export type CellStateBindGroupPair = [GPUBindGroup, GPUBindGroup];

const SHAPE_VERTICES = new Float32Array([-1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1]);
const WORKGROUP_SIZE = 8;
const defaultRule : Uint32Array = new Uint32Array([1, 0, 2, 2, 3, 2, 3, 3, 0]); // Conway's game of life

export default function useLTL({ size, pattern, shaders }: LTLSettings) {
	const [status, setStatus] = useState(LTLStatus.ALLOCATING_RESOURCES);
	const [error, setError] = useState('');
	const [resources, setResources] = useState<GPUResources | null>(null);
	const [play, setPlay] = useState(true);
	const [step, setStep] = useState(0);
	const [FPS, setFPS] = useState(1);
	const [stepsPerFrame, setStepsPerFrame] = useState(1);

	const canvasRef = useRef<HTMLCanvasElement>(null);

	useEffect(() => {
		async function init() {
			if (resources) return;
			console.log('Initialising Shaders');

			// Check for WebGPU support
			if (!navigator.gpu) {
				setStatus(LTLStatus.FAILED);
				setError('WebGPU is not supported on this browser.');
			}

			// Request GPU Adapter
			let adapter: GPUAdapter | null = null;
			try {
				adapter = await navigator.gpu.requestAdapter();
			} catch (error) {
				setStatus(LTLStatus.FAILED);
				setError(`Failed to get a GPU adapter: ${(error as Error).message}`);
			}
			if (!adapter) return;

			// Request GPU Device
			let device: GPUDevice | null = null;
			try {
				device = await adapter.requestDevice();
			} catch (error) {
				setStatus(LTLStatus.FAILED);
				setError(`Failed to get a GPU device: ${(error as Error).message}`);
				return;
			}

			// Configure canvas context
			const context = canvasRef.current?.getContext('webgpu');
			if (!context) {
				setStatus(LTLStatus.FAILED);
				setError('Failed to get canvas context.');
				return;
			}
			context.configure({
				device,
				format: navigator.gpu.getPreferredCanvasFormat(),
				alphaMode: 'opaque'
			});

			// Get rule 

			const parsedRule : Uint32Array | null = parseRuleString(pattern.rule);
			let rule: Uint32Array = new Uint32Array([]);
			if (parsedRule != null) {
				rule = parsedRule;
			} else {
				rule = defaultRule;
				// Throw an error for the user to show their rule did not load successfully
				setStatus(LTLStatus.FAILED);
				setError('Failed to parse rule. Rule set to Conway\' life');
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
				code: guiShader
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
						buffer: { type: 'read-only-storage' }
					},
					{
						binding: 2, // Output State (Compute)
						visibility: GPUShaderStage.COMPUTE,
						buffer: { type: 'storage' }
					},
					{
						binding: 3, // Rule
						visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT, // Fragment needs to see number of possible cell states to gradiate colours
						buffer: { type: 'read-only-storage' }
					},
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
			const shapeArray = new Float32Array([size, size]);
			const shapeBuffer = device.createBuffer({
				label: 'Size Buffer',
				size: shapeArray.byteLength,
				usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
			});

			const cellState = new Float32Array(size * size).fill(0);
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

			const ruleBuffer =
				device.createBuffer({
					label: 'Rule',
					size: rule.byteLength,
					usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
				})
			;

			// Write buffers
			device.queue.writeBuffer(shapeBuffer, 0, shapeArray);
			device.queue.writeBuffer(cellStateBuffers[1], 0, cellState);
			if (pattern.pattern == null) {
				// If there is no pre-determined pattern, randomise the grid
				for (let i = 0; i < cellState.length; i++) {
					cellState[i] = Math.random() > 0.6 ? 1 : 0; // random starting position
				}
			} else {
				// Copy the starting pattern in
				for (let i = 0; i < cellState.length; i++) {
					cellState[i] = 0;
				}
				const centreOffset = Math.floor((size-pattern.cols)/2);
				for (let i = 0; i < pattern.cols; i++) {
					for (let j = 0; j < pattern.rows; j++){
						cellState[i+centreOffset+(j+centreOffset)*size] = pattern.pattern[i+j*pattern.cols];
					}
				}
			}
			device.queue.writeBuffer(cellStateBuffers[0], 0, cellState);
			device.queue.writeBuffer(ruleBuffer, 0, rule);

			// Create Bind Group
			const bindGroups: CellStateBindGroupPair = [
				device.createBindGroup({
					label: 'Bind Group A',
					layout: bindGroupLayout,
					entries: [
						{ binding: 0, resource: { buffer: shapeBuffer } },
						{ binding: 1, resource: { buffer: cellStateBuffers[0] } },
						{ binding: 2, resource: { buffer: cellStateBuffers[1] } },
						{ binding: 3, resource: { buffer: ruleBuffer } },
					]
				}),
				device.createBindGroup({
					label: 'Bind Group B',
					layout: bindGroupLayout,
					entries: [
						{ binding: 0, resource: { buffer: shapeBuffer } },
						{ binding: 1, resource: { buffer: cellStateBuffers[1] } },
						{ binding: 2, resource: { buffer: cellStateBuffers[0] } },
						{ binding: 3, resource: { buffer: ruleBuffer } },
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
					vertex: vertexBuffer,
				}
			});
			setStatus(LTLStatus.READY);
		}

		init();

		// Cleanup
		return () => {
			// Destroy device & associated buffers
			resources?.device.destroy();
		};
	}, [size, pattern, shaders.simulation, resources]);

	useEffect(() => {
		if (status !== LTLStatus.READY || !resources) return;

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

				// Compute Pass
				const computePass = encoder.beginComputePass();
				computePass.setPipeline(resources.pipelines.simulation);
				computePass.setBindGroup(0, resources.bindGroups[step % 2]);
				const workgroupCount = Math.ceil(size / WORKGROUP_SIZE);
				computePass.dispatchWorkgroups(workgroupCount, workgroupCount);
				computePass.end();

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
				renderPass.setBindGroup(0, resources.bindGroups[step % 2]);
				renderPass.draw(SHAPE_VERTICES.length / 2, size * size);
				renderPass.end();

				// Submit commands
				resources.device.queue.submit([encoder.finish()]);
				setStep(prev => prev + 1);
			}

			animationFrameId = requestAnimationFrame(renderLoop);
		};

		// Start render loop
		animationFrameId = requestAnimationFrame(renderLoop);

		return () => {
			cancelAnimationFrame(animationFrameId);
		};
	}, [play, FPS, resources, status, size, step]);

	return {
		play,
		setPlay,
		step,
		setStep,
		error,
		setError,
		status,
		stepsPerFrame,
		setStepsPerFrame,
		setStatus,
		FPS,
		setFPS,
		canvasRef
	};
}


export type LTLControls = ReturnType<typeof useLTL>;
