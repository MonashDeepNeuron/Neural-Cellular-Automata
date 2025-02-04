import type { Pattern } from '@/patterns';

export type CellStateBufferPair = [GPUBuffer, GPUBuffer];
export type CellStateBindGroupPair = [GPUBindGroup, GPUBindGroup];

export default class BufferManager {
	private device: GPUDevice;

	constructor(device: GPUDevice) {
		this.device = device;
	}

	// Notes for vertex buffer
	// A square that will be drawn on each cell. Vertices loaded correspond to
	// if the cell were a 1x1 square. This will be scaled and positioned by
	// guiShader code (specifically vertexShader)

	/**
	 * Load the vertices to be drawn
	 * @param shapeVertices Vertices of triangles that form the
	 *          shape.
	 * @returns vertexBuffer, vertexBufferLayout
	 */
	loadShapeVertexBuffer(shapeVertices: Float32Array) {
		// load vertices into buffer
		// This is currently used only for a square
		const vertexBuffer = this.device.createBuffer({
			label: 'Cell vertices', // Error message label
			size: shapeVertices.byteLength,
			usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
		});
		this.device.queue.writeBuffer(vertexBuffer, 0, shapeVertices);

		// define layout of loaded binary data
		const vertexBufferLayout = {
			arrayStride: 8, // 32bit = 4 bytes, 4x2 = 8 bytes to skip to find next vertex
			attributes: [
				{
					format: 'float32x2' as const, // two 32 bit floats per vertex
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
	 * @returns bindGroups, uniformBuffer, cellStateBuffers, ruleBuffer
	 **/
	initialiseComputeBindgroups(renderPipeline: GPURenderPipeline, gridSize: number, initialState: Pattern, rule: Uint32Array) {
		// Uniform grid
		const uniformArray = new Float32Array([gridSize, gridSize]);
		const uniformBuffer = this.device.createBuffer({
			label: 'Grid Uniforms',
			size: uniformArray.byteLength,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});
		this.device.queue.writeBuffer(uniformBuffer, 0, uniformArray);

		const cellStateBuffers = this.setInitialStateBuffer(gridSize, initialState);
		const ruleBuffer = this.setRuleBuffer(rule);

		// Setup alternating bind groups
		const bindGroups: CellStateBindGroupPair = [
			this.createBindGroup(renderPipeline, 'Renderer A', uniformBuffer, cellStateBuffers, ruleBuffer),
			this.createBindGroup(renderPipeline, 'Renderer B', uniformBuffer, cellStateBuffers.toReversed() as CellStateBufferPair, ruleBuffer)
		];

		return { bindGroups, uniformBuffer, cellStateBuffers, ruleBuffer };
	}

	/**
	 * Creates and initialises the cell state buffers.
	 * @param size Size of the square grid.
	 * @param pattern The initial state pattern.  If not defined, the grid will instead be initialised with a random grid.
	 * @returns A pair of cell state GPUBuffers.
	 */
	setInitialStateBuffer(size: number, pattern: Pattern | null): CellStateBufferPair {
		// Cell state arrays
		const cellStateArray = new Uint32Array(size * size);
		const cellStateBuffers: CellStateBufferPair = [
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

		// Write initial state to 1st buffer
		if (pattern?.pattern) {
			// Initialise the pattern in the middle of the grid
			cellStateArray.fill(0);
			const rowOffset = Math.floor((size - pattern.rows) / 2);
			const colOffset = Math.floor((size - pattern.cols) / 2);
			for (let r = 0; r < pattern.rows; r++) {
				for (let c = 0; c < pattern.cols; c++) {
					cellStateArray[(r + rowOffset) * size + c + colOffset] = pattern.pattern[r * pattern.cols + c];
				}
			}
		} else {
			// Create a random state if no initial state is provided
			const THRESHOLD = 0.6;
			for (let i = 0; i < cellStateArray.length; i++) {
				cellStateArray[i] = Math.random() > THRESHOLD ? 1 : 0;
			}
		}
		this.device.queue.writeBuffer(cellStateBuffers[0], 0, cellStateArray);

		// Write zeros to 2nd buffer
		cellStateArray.fill(0);
		this.device.queue.writeBuffer(cellStateBuffers[1], 0, cellStateArray);

		return cellStateBuffers;
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
	 * @param ruleBuffer
	 * @returns GPUBindGroupLayout
	 */
	createBindGroup(
		renderPipeline: GPURenderPipeline,
		label: string,
		uniformBuffer: GPUBuffer,
		cellStates: CellStateBufferPair,
		ruleBuffer: GPUBuffer
	) {
		return this.device.createBindGroup({
			label: label,
			layout: renderPipeline.getBindGroupLayout(0), // NOTE: renderpipelne and simulation pipeline use same layout
			entries: [
				{ binding: 0, resource: { buffer: uniformBuffer } },
				{ binding: 1, resource: { buffer: cellStates[0] } },
				{ binding: 2, resource: { buffer: cellStates[1] } },
				{ binding: 3, resource: { buffer: ruleBuffer } }
			]
		});
	}

	/**
	 * Convenience function for code organisation.
	 * Writes out the rule buffer such that it is consistent with expected settings
	 * @param rule The numbers from the current rule (kernel)
	 * @returns ruleBuffer as GPUBuffer object
	 */
	setRuleBuffer(rule: Uint32Array) {
		const ruleBuffer = this.device.createBuffer({
			label: 'Rule Storage',
			size: rule.byteLength,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		});
		this.device.queue.writeBuffer(ruleBuffer, 0, rule);
		return ruleBuffer;
	}
}
