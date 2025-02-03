// This class has not been implemented anywhere yet
/** @todo Finish writing this Manager */
export default class PipelineManager {
	// SIMULATION PIPELINE
	static simulationPipeline = device.createComputePipeline({
		label: 'Simulation pipeline',
		layout: pipelineLayout,
		compute: {
			module: simulationShaderModule,
			entryPoint: 'computeMain'
		}
	});

	static cellPipeline = device.createRenderPipeline({
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
			targets: [
				{
					format: canvasFormat
				}
			]
		}
	});

	static pipelines = new Map();

	static createRenderPipeline(device, label, layout, vertexInfo, fragmentInfo, canvasFormat) {
		const pipeline = DeviceManager.getDevice().createRenderPipeline({
			label,
			layout,
			vertex: vertexInfo,
			fragment: fragmentInfo,
			targets: [{ format: canvasFormat }]
		});

		this.pipelines.set(label, pipeline);
		return pipeline;
	}

	static getPipeline(label) {
		return this.pipelines.get(label);
	}
}
