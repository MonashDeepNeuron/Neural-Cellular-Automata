'use client';

import useTypedSelector from '@/hooks/useTypedSelector';
import { type ChangeEventHandler, useState } from 'react';
import { useDispatch } from 'react-redux';
import { setFramesPerSecond } from '../store/webGPUSlice';

export const SpeedSlider = () => {
	const dispatch = useDispatch();
	const framesPerSecond = useTypedSelector(state => state.webGPU.framesPerSecond);
	const [fpsReference, setFpsReference] = useState(framesPerSecond);

	const handleChange: ChangeEventHandler<HTMLInputElement> = event => {
		const fps = Number(event.target.value);
		setFpsReference(fps);
		dispatch(setFramesPerSecond(fps));
	};

	return (
		<div className='mt-4 flex flex-col items-center rounded-md shadow-lg bg-blue-100 p-8'>
			<label htmlFor='speed-slider' className='text-lg font-semibold mb-2'>
				Frames Per Second
			</label>
			<input id='speed-slider' type='range' min='1' max='60' value={framesPerSecond} onChange={handleChange} className='w-full max-w-md' />
			<div className='flex justify-between w-full max-w-md text-sm mt-2'>
				<span>1 FPS</span>
				<span>{fpsReference} FPS</span>
				<span>60 FPS</span>
			</div>
		</div>
	);
};
