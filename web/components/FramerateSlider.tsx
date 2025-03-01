'use client';

import { type Dispatch, type SetStateAction, useId } from 'react';

interface FramerateSliderProps {
	state: number;
	setState: Dispatch<SetStateAction<number>>;
	min?: number;
	max?: number;
}

export default function FramerateSlider({ state, setState, min = 1, max = 120 }: FramerateSliderProps) {
	const id = useId();

	return (
		<div className='w-96 max-w-full'>
			<label htmlFor={id} className='text-md mb-2'>
				<b>Framerate</b>:
			</label>
			<input id={id} type='range' min={min} max={max} value={state} onChange={e => setState(Number(e.target.value))} className='w-full accent-purple-mdn' />
			<div className='flex justify-between w-full text-sm mt-2'>
				<span>{min} FPS</span>
				<span>{state} FPS</span>
				<span>{max} FPS</span>
			</div>
		</div>
	);
}
