import { useEffect } from 'react';
import { useDispatch } from 'react-redux';
import { incrementStep } from '../store/webGPUSlice';
import { createComputePass, createRenderPass } from '../util/ShaderPass';
import useTypedSelector from './useTypedSelector';
import type { WebGPUResources, WebGPUSettings } from './useWebGPU';

interface UseRenderLoop {
	settings: WebGPUSettings;
	resources: WebGPUResources | null;
}

export const useRenderLoop = ({ settings, resources }: UseRenderLoop) => {
	const { running, framesPerSecond, step, template } = useTypedSelector(state => state.webGPU);

	const dispatch = useDispatch();

	// biome-ignore lint/correctness/useExhaustiveDependencies: test missing dependencies
	useEffect(() => {
		if (!resources) {
			console.log('WebGPU initialization incomplete.');
			return;
		}

		/**
		 * Defining renderPass and computePass
		 */
		const renderPass = createRenderPass(settings, resources);
		const computePass = createComputePass(settings, resources);
		let intervalID: ReturnType<typeof setInterval>;

		/**
		 * Defining callback renderLoop. This updates every 50ms until running state changes
		 * Then we clear up by clearing the interval
		 */
		const renderLoop = () => {
			const encoder = resources.device.createCommandEncoder();
			for (let i = 0; i < 1; i++) {
				// TODO: CHANGE THIS TO number of computer passes
				computePass(encoder, step);
				dispatch(incrementStep());
			}

			renderPass(encoder, step);
			resources.device.queue.submit([encoder.finish()]);
		};

		if (running) {
			intervalID = setInterval(renderLoop, 1000 / framesPerSecond);
		}

		return () => {
			clearInterval(intervalID);
		};
	}, [running, step, template, framesPerSecond]);
};
