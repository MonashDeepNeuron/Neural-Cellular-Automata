'use client';

import useTypedSelector from '@/hooks/useTypedSelector';
import { useRef, useState } from 'react';
import { useDispatch } from 'react-redux';
import { useRenderLoop } from '../hooks/useRenderLoop';
import useWebGPU, { type WebGPUResources } from '../hooks/useWebGPU';
import startingPatterns from '../patterns';
import { toggleRunning } from '../store/webGPUSlice';
import { SpeedSlider } from './SpeedSlider';
import TemplateDropdown from './TemplateDropdown';

export default function Canvas() {
	/**
	 * Obtain state from WebGPU slice of redux store
	 */
	const dispatch = useDispatch();
	const running = useTypedSelector(state => state.webGPU.running);
	const template = useTypedSelector(state => state.webGPU.template);
	const step = useTypedSelector(state => state.webGPU.step);

	const [toolkitOpen, setToolkitOpen] = useState(false);

	/**
	 * Derived state for configuring WebGPU side effects
	 */
	const selectedTemplate = template ? startingPatterns[template] : startingPatterns[8];
	const settings = {
		workgroupSize: 16,
		initialState: selectedTemplate,
		gridSize: selectedTemplate.minGrid,
		ruleString: selectedTemplate.rule
	};

	/**
	 * Create a ref to the canvas element, pass it into
	 * useWebGPU effect custom hook to initialise web gpu resources and state
	 * We use ref here for persistence between re-renders
	 */
	const canvasRef = useRef<HTMLCanvasElement>(null);
	const resourceRef = useRef<WebGPUResources>(null);

	resourceRef.current = useWebGPU(canvasRef, settings);

	useRenderLoop({
		settings: settings,
		resources: resourceRef.current
	});

	return (
		<div className='flex flex-col lg:flex-row items-center lg:items-start justify-between w-full h-screen p-4 gap-4 bg-background overflow-auto'>
			{/* Controls Panel */}
			<div className='flex flex-col bg-white rounded-md shadow-lg p-4 w-full lg:w-1/3 relative'>
				<button
					type='button'
					onClick={() => setToolkitOpen(!toolkitOpen)}
					className='absolute bottom-2 right-2 bg-blue-500 text-white px-2 py-1 text-sm rounded hover:bg-blue-600'
				>
					{toolkitOpen ? 'Collapse' : 'Expand'}
				</button>
				<h1 className='text-2xl font-bold mb-2'>
					Simulating <span className='text-pink-500 font-bold'>{startingPatterns[template].name}</span>
				</h1>
				{toolkitOpen && (
					<div className='mt-4'>
						<TemplateDropdown />
						<SpeedSlider />
					</div>
				)}
				<button
					type='button'
					onClick={() => dispatch(toggleRunning())}
					className={`mt-4 px-4 py-2 rounded-md shadow ${
						running ? 'bg-red-500 text-white hover:bg-red-600' : 'bg-green-500 text-white hover:bg-green-600'
					}`}
				>
					{running ? 'Pause' : 'Start'}
				</button>
				<h2 className='mt-4 text-lg font-medium'>Step: {step}</h2>
			</div>
			{/* Canvas */}
			<div className='flex items-center justify-center p-4'>
				<canvas
					ref={canvasRef}
					width={1024}
					height={1024}
					className='mt-8 border border-gray-300 rounded-md shadow-lg w-full lg:w-2/3 h-auto'
				/>
			</div>
		</div>
	);
}
