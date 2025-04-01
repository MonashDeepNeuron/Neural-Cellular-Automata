'use client';

import { useEffect, useRef, useState } from 'react';

export enum CAStatus {
	ALLOCATING_RESOURCES = 'Allocating Resources',
	READY = 'Ready',
	FAILED = 'Failed'
}

export interface GPUResources {
	device: GPUDevice;
	context: GPUCanvasContext;
	bindGroup: GPUBindGroup;
	pipelines: {
		render: GPURenderPipeline;
	};
	buffers: {
		vertex: GPUBuffer;
	};
}

export type CellStateBufferPair = [GPUBuffer, GPUBuffer];

const SHAPE_VERTICES = new Float32Array([-1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1]);

interface Use3DSettings {
	size: number;
}

export default function use3D({ size }: Use3DSettings) {
	const [status, setStatus] = useState(CAStatus.ALLOCATING_RESOURCES);
	const [error, setError] = useState('');
	const [resources, setResources] = useState<GPUResources | null>(null);
	const [play, setPlay] = useState(true);
	const [step, setStep] = useState(0);
	const [FPS, setFPS] = useState(60);
	const [stepsPerFrame, setStepsPerFrame] = useState(1);
	const [camera, setCamera] = useState({
		x: 0,
		y: 0,
		z: 0,
		rho: 1,
		theta: 0,
		phi: Math.PI / 2,
		f: 10
	});
	const dragging = useRef(false);
	const prevMouse = useRef({ x: 0, y: 0 });
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
				code: ''
			});

			// Create Global Binding Layout
			const bindGroupLayout = device.createBindGroupLayout({
				entries: []
			});

			// Create pipeline layout
			const pipelineLayout = device.createPipelineLayout({
				label: 'Global Pipeline Layout',
				bindGroupLayouts: [bindGroupLayout]
			});

			// Create pipelines
			const cellPipeline = device.createRenderPipeline({
				label: 'Render Pipeline',
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

			const bindGroup = device.createBindGroup({
				label: 'Bind Group',
				layout: bindGroupLayout,
				entries: []
			});

			// All done!
			setResources({
				device,
				context,
				bindGroup,
				pipelines: {
					render: cellPipeline
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
	}, [resources]);

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
				renderPass.setPipeline(resources.pipelines.render);
				renderPass.setVertexBuffer(0, resources.buffers.vertex);
				renderPass.setBindGroup(0, resources.bindGroup);
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

	useEffect(() => {
		const canvas = canvasRef.current;
		if (!canvas) return;

		const handleMouseDown = (e: MouseEvent) => {
			dragging.current = true;
			prevMouse.current = { x: e.clientX, y: e.clientY };
		};

		const handleMouseMove = (e: MouseEvent) => {
			if (!dragging.current) return;
			const deltaX = e.clientX - prevMouse.current.x;
			const deltaY = e.clientY - prevMouse.current.y;
			prevMouse.current = { x: e.clientX, y: e.clientY };

			setCamera(prev => {
				// Clamp phi
				const phi = Math.max(Number.EPSILON, Math.min(Math.PI - Number.EPSILON, prev.phi - deltaY * 0.01));
				return {
					...prev,
					theta: prev.theta - deltaX * 0.01,
					phi: phi
				};
			});
		};

		const handleMouseUp = () => {
			dragging.current = false;
		};

		const handleWheel = (e: WheelEvent) => {
			setCamera(c => ({
				...c,
				rho: Math.max(50, Math.min(1000, c.rho + e.deltaY * 0.2))
			}));
		};

		canvas.addEventListener('mousedown', handleMouseDown);
		canvas.addEventListener('mousemove', handleMouseMove);
		canvas.addEventListener('mouseup', handleMouseUp);
		canvas.addEventListener('wheel', handleWheel);

		return () => {
			canvas.removeEventListener('mousedown', handleMouseDown);
			canvas.removeEventListener('mousemove', handleMouseMove);
			canvas.removeEventListener('mouseup', handleMouseUp);
			canvas.removeEventListener('wheel', handleWheel);
		};
	}, [canvasRef.current]);

	useEffect(() => {
		const { rho, theta, phi } = camera;
		const x = camera.x + rho * Math.cos(theta) * Math.sin(phi);
		const y = camera.y + rho * Math.sin(theta) * Math.sin(phi);
		const z = camera.z + rho * Math.cos(phi);

		console.log(`x: ${x.toPrecision(4)}\ty: ${y.toPrecision(4)}\tz: ${z.toPrecision(4)}`);
	}, [camera]);

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

export type NCAControls = ReturnType<typeof use3D>;
