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
		<div className='mt-4 flex flex-col items-center bg-background p-8 w-96'>
			<label htmlFor={id} className='text-md font-semibold mb-2'>
				Frames Per Second
			</label>
			<input
				id={id}
				type='range'
				min={min}
				max={max}
				value={state}
				onChange={e => setState(Number(e.target.value))}
				className='w-full max-w-md'
			/>
			<div className='flex justify-between w-full max-w-md text-sm mt-2'>
				<span>{min} FPS</span>
				<span>{state} FPS</span>
				<span>{max} FPS</span>
			</div>
		</div>
	);
}
