'use client';
import { CAStatus, type NCAControls } from '@/hooks/useNCA';
import clsx from 'clsx';
import { useId } from 'react';
import Card from '../Card';
import FramerateSlider from '../FramerateSlider';

interface SimulatorProps extends NCAControls {
	name: string;
	className?: string;
	size: number;
}

export default function Simulator({
	name,
	FPS,
	setFPS,
	setPlay,
	play,
	error,
	canvasRef,
	size,
	step,
	status,
	stepsPerFrame,
	setStepsPerFrame,
	className
}: SimulatorProps) {
	const checkboxId = useId();

	return (
		<div className='grid gap-4 grid-rows-[1fr,auto] grid-cols-1 max-w-full lg:grid-rows-1 lg:grid-cols-[24rem,1fr] lg:h-[calc(100vh-6rem)]'>
			<Card>
				<h1 className='font-bold'>{name}</h1>
				<p>
					<b>Status</b>: {status}
				</p>
				<p>
					<b>Step</b>: {step}
				</p>
				<FramerateSlider state={FPS} setState={setFPS} max={240} />
				<input
					type='checkbox'
					id={checkboxId}
					checked={stepsPerFrame === 2}
					className='mr-2'
					onChange={() => setStepsPerFrame(stepsPerFrame === 1 ? 2 : 1)}
				/>
				<label htmlFor={checkboxId}>Skip every second frame</label>
				<button
					type='button'
					disabled={status !== CAStatus.READY}
					className={clsx('mt-4 block px-4 py-2 text-white min-w-24 rounded-md font-bold', play ? 'bg-red-700' : 'bg-green-700')}
					onClick={() => setPlay(!play)}
				>
					{play ? 'Pause' : 'Play'}
				</button>
			</Card>
			<Card className='flex justify-center'>
				<div className='relative aspect-square w-full lg:w-auto lg:h-full max-h-full max-w-full overflow-hidden'>
					<div className='absolute top-0 left-0 flex items-center justify-center overflow-hidden w-full h-full'>
						{error && <p className='text-red-500 px-4 text-center'>{error}</p>}
					</div>
					<canvas
						height={size}
						width={size}
						className={clsx('h-full w-full', className)}
						ref={canvasRef}
						style={{ imageRendering: 'pixelated' }}
					/>
				</div>
			</Card>
		</div>
	);
}
