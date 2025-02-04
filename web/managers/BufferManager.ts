import type { Pattern } from '@/patterns';

export default class BufferManager {
	private device: GPUDevice;

	constructor(device: GPUDevice) {
		this.device = device;
	}
	// Notes for vertex buffer
	// A square that will be drawn on each cell. Vertexs loaded correspond to
	// if the cell were a 1x1 square. This will be scaled and positioned by
	// guiShader code (specifically vertexShader)

	/**
	 * Load the verticies to be drawn
	 * @param shapeVerticies Verticies of triangles that form the
	 *          shape.
	 * @returns vertexBuffer, vertexBufferLayout
	 */
	loadShapeVertexBuffer(shapeVerticies: Float32Array) {
		// load verticies into buffer
		// This is currently used only for a square
		const vertexBuffer = this.device.createBuffer({
			label: 'Cell vertices', // Error message label
			size: shapeVerticies.byteLength,
			usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
		});
		this.device.queue.writeBuffer(vertexBuffer, /*bufferOffset=*/ 0, shapeVerticies);

		// define layout of loaded binary data
		const vertexBufferLayout = {
			arrayStride: 8, // 32bit = 4 bytes, 4x2 = 8 bytes to skip to find next vertex
			attributes: [
				{
					format: 'float32x2', // two 32 bit floats per vertex
					offset: 0,
					shaderLocation: 0 // Position, see vertex shader
				}
			]
		};

		return { vertexBuffer, vertexBufferLayout };
	}

	/**
	 * Define where and what information can be accessed by the GPU code
	 * Set and make all required information available to the GPU (for the first time)
	 * @param renderPipeline
	 * @param gridSize sidelength of grid, where the grid is a square array of cells
	 * @param initialState the initial state of the grid, filled with floating point values
	 *      NOTE: This may be null, where null represents a randomised grid. If the pattern
	 *      is null, this also constitutes a randomised grid.
	 * @param rule The numbers from the current rule (kernel)
	 * @returns bindGroups, uniformBuffer, cellStateStorage, ruleStorage
	 **/
	initialiseComputeBindgroups(renderPipeline: GPURenderPipeline, gridSize: number, initialState: Pattern, rule: Float32Array) {
		// Uniform grid
		const uniformArray = new Float32Array([gridSize, gridSize]);
		const uniformBuffer = this.device.createBuffer({
			label: 'Grid Uniforms',
			size: uniformArray.byteLength,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});
		this.device.queue.writeBuffer(uniformBuffer, 0, uniformArray);

		const cellStateStorage = this.setInitialStateBuffer(gridSize, initialState);

		const ruleStorage = this.setRuleBuffer(rule);

		// setup bind groups
		const bindGroups = [
			this.createBindGroup(
				renderPipeline,
				'Cell renderer bind group A',
				uniformBuffer,
				cellStateStorage[0],
				cellStateStorage[1],
				ruleStorage
			),
			this.createBindGroup(renderPipeline, 'Cell render bind group B', uniformBuffer, cellStateStorage[1], cellStateStorage[0], ruleStorage)
		];
		return { bindGroups, uniformBuffer, cellStateStorage, ruleStorage };
	}

	/**
	 * Defines the buffer sizes and sets up the buffers for the cell storages
	 * Randomises the canvas when the initialState is null
	 * @param gridSize sidelength of grid, where the grid is a square array of cells
	 * @param initialState the initial state of the grid
	 *      NOTE: This may be null, where null represents a randomised grid. If the pattern
	 *      is null, this also constitutes a randomised grid.
	 * @returns GPUBuffer[2] cellStateStorage with two buffers.
	 *      cellStateStorage[0] contains the initial state, cellStateStorage[1] is blank
	 */
	setInitialStateBuffer(gridSize: number, initialState: Pattern | null) {
		// If initial state = null, assign random
		// Cell state arrays
		const cellStateArray = new Uint32Array(gridSize * gridSize);
		const cellStateStorage = [
			this.device.createBuffer({
				label: 'Cell State A',
				size: cellStateArray.byteLength,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
			}),

			this.device.createBuffer({
				label: 'Cell State B',
				size: cellStateArray.byteLength,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
			})
		];

		// write to buffer A
		if (initialState === null || initialState.pattern == null) {
			for (let i = 0; i < cellStateArray.length; i++) {
				cellStateArray[i] = Math.random() > 0.6 ? 1 : 0; // random starting position
			}
			// console.log("Randomising canvas ...");
		} else {
			for (let i = 0; i < cellStateArray.length; i++) {
				cellStateArray[i] = 0;
			}
			const centreOffset = Math.floor((gridSize - initialState.width) / 2);
			for (let i = 0; i < initialState.width; i++) {
				for (let j = 0; j < initialState.height; j++) {
					cellStateArray[i + centreOffset + (j + centreOffset) * gridSize] = initialState.pattern[i + j * initialState.width];
				}
			}
			// console.log(`Implementing ${initialState.name}`);
		}
		this.device.queue.writeBuffer(cellStateStorage[0], 0, cellStateArray);

		// write to buffer B
		for (let i = 0; i < cellStateArray.length; i++) {
			cellStateArray[i] = 0;
		}
		this.device.queue.writeBuffer(cellStateStorage[1], 0, cellStateArray);

		return cellStateStorage;
	}

	/**
	 * Convenience function, creates the bindings and sets the accessibility of
	 * GPU-available resources. Bindgroup layout defines the data visibility and
	 * permissions
	 * @param device
	 * @returns bindgroup layout with 4 entries (0-3)
	 */
	createBindGroupLayout() {
		return this.device.createBindGroupLayout({
			label: 'Cell Bind Group Layout',
			entries: [
				{
					binding: 0,
					visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT,
					buffer: {} // Grid uniform buffer
				},

				{
					binding: 1,
					visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT,
					buffer: { type: 'read-only-storage' } // Cell state input buffer
				},

				{
					binding: 2,
					visibility: GPUShaderStage.COMPUTE,
					buffer: { type: 'storage' } // Cell state output buffer
				},

				{
					binding: 3,
					visibility: GPUShaderStage.COMPUTE | GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT,
					buffer: { type: 'read-only-storage' } // Ruleset
				}
			]
		});
	}

	/**
	 * Defines the way that each resource is referenced/accessed from WGSL code
	 * This function ensures that consistent binding numbers are used throughout
	 * This must be kept consistent with the bindgroup layout
	 * @param renderPipeline
	 * @param label
	 * @param uniformBuffer
	 * @param inputStateBuffer
	 * @param outputStateBuffer
	 * @param ruleStorage
	 * @returns GPUBindGroupLayout
	 */
	createBindGroup(
		renderPipeline: GPURenderPipeline,
		label: string,
		uniformBuffer: GPUBuffer,
		cellStateA: GPUBuffer,
		cellStateB: GPUBuffer,
		ruleStorage: GPUBuffer
	) {
		return this.device.createBindGroup({
			label: label,
			layout: renderPipeline.getBindGroupLayout(0), // NOTE: renderpipelne and simulation pipeline use same layout
			entries: [
				{ binding: 0, resource: { buffer: uniformBuffer } },
				{ binding: 1, resource: { buffer: cellStateA } },
				{ binding: 2, resource: { buffer: cellStateB } },
				{ binding: 3, resource: { buffer: ruleStorage } }
			]
		});
	}

	/**
	 * Convenience function for code organisation.
	 * Writes out the rule buffer such that it is consistent with expected settings
	 * @param rule The numbers from the current rule (kernel)
	 * @returns rulestorage as GPUBuffer object
	 */
	setRuleBuffer(rule: Float32Array) {
		const ruleArray = rule;
		const ruleStorage = this.device.createBuffer({
			label: 'Rule Storage',
			size: ruleArray.byteLength,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		});
		this.device.queue.writeBuffer(ruleStorage, 0, ruleArray);
		return ruleStorage;
	}
}
