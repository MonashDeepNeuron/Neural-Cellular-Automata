import type { WebGPUResources, WebGPUSettings } from '@/hooks/useWebGPU';

const SQUARE_VERTICIES = new Float32Array([-1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1]);

/**
 * Factory function to create a parameterized renderPass function.
 */
export const createRenderPass = ({ gridSize }: WebGPUSettings, { context, pipelines, buffers }: WebGPUResources) => {
	return (encoder: GPUCommandEncoder, step: number) => {
		const renderPass = encoder.beginRenderPass({
			colorAttachments: [
				{
					view: context.getCurrentTexture().createView(),
					loadOp: 'clear',
					clearValue: { r: 0, g: 0, b: 0, a: 1 },
					storeOp: 'store'
				}
			]
		});

		renderPass.setPipeline(pipelines.renderPipeline);
		renderPass.setVertexBuffer(0, buffers.vertexBuffer);
		renderPass.setBindGroup(0, buffers.bindGroups[step % 2]);
		renderPass.draw(SQUARE_VERTICIES.length / 2, gridSize * gridSize);
		renderPass.end();
	};
};

/**
 * Factory function to create a parameterized computePass function.
 */
export const createComputePass = ({ gridSize, workgroupSize }: WebGPUSettings, { pipelines, buffers }: WebGPUResources) => {
	return (encoder: GPUCommandEncoder, step: number) => {
		const computePass = encoder.beginComputePass();
		computePass.setPipeline(pipelines.simulationPipeline);
		computePass.setBindGroup(0, buffers.bindGroups[step % 2]);
		const workgroupCount = Math.ceil(gridSize / workgroupSize);
		computePass.dispatchWorkgroups(workgroupCount, workgroupCount);
		computePass.end();
	};
};
