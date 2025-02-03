import { useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { incrementStep } from '../store/webGPUSlice';
import { createComputePass, createRenderPass } from '../util/ShaderPass';

export const useRenderLoop = ({ settings, resources }) => {
	const { workgroupSize, gridSize } = settings;
	const { running, framesPerSecond, step, template } = useSelector(state => state.webGPU);

	const dispatch = useDispatch();

	useEffect(() => {
		if (!resources) {
			console.log('WebGPU initialization incomplete.');
			return;
		}
		const { device, context, pipelines, buffers } = resources;

		/**
		 * Defining renderPass and computePass
		 */
		const renderPass = createRenderPass(context, pipelines, buffers, gridSize);
		const computePass = createComputePass(pipelines, buffers, gridSize, workgroupSize);
		let intervalID;

		/**
		 * Defining callback renderLoop. This updates every 50ms until running state changes
		 * Then we clear up by clearing the interval
		 */
		const renderLoop = () => {
			const encoder = device.createCommandEncoder();
			for (let i = 0; i < 1; i++) {
				// TODO: CHANGE THIS TO number of computer passes
				computePass(encoder, step);
				dispatch(incrementStep());
			}

			renderPass(encoder, step);
			device.queue.submit([encoder.finish()]);
		};

		if (running) {
			intervalID = setInterval(renderLoop, 1000 / framesPerSecond);
		}

		return () => {
			clearInterval(intervalID);
		};
	}, [running, step, template, framesPerSecond]);
};
