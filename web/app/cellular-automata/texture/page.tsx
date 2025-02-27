'use client';

import Card from '@/components/Card';
import FramerateSlider from '@/components/FramerateSlider';
import useNCA, { NCAStatus } from '@/hooks/useNCA';
import { texture as simulation } from '@/shaders/nca/simulation';
import clsx from 'clsx';

const SIZE = 256;

export default function Texture() {
	const { error, status, play, setPlay, step, FPS, setFPS, canvasRef } = useNCA({
		size: SIZE,
		channels: 12,
		hiddenChannels: 96,
		convolutions: 4,
		weightsURL: '/weights/texture-bubbles.bin',
		shaders: {
			simulation
		}
	});

	return (
		<div className='grid gap-4 grid-rows-2 grid-cols-1 max-w-full lg:grid-rows-1 lg:grid-cols-[24rem,1fr] lg:h-[calc(100vh-6rem)]'>
			<Card>
				<h1>Texture</h1>
				<p>
					<b>Status</b>: {status}
				</p>
				<p>
					<b>Step</b>: {step}
				</p>
				<FramerateSlider state={FPS} setState={setFPS} max={240} />
				<button
					type='button'
					disabled={status !== NCAStatus.READY}
					className={clsx('mt-4 px-4 py-2 text-white min-w-24 rounded-md font-bold', play ? 'bg-red-700' : 'bg-green-700')}
					onClick={() => setPlay(!play)}
				>
					{play ? 'Pause' : 'Play'}
				</button>
			</Card>
			<Card className='flex justify-center'>
				<div className='relative aspect-square h-full max-h-full max-w-full overflow-hidden'>
					<div className='absolute top-0 left-0 flex items-center justify-center overflow-hidden w-full h-full'>
						{error && <p className='text-red-500 px-4 text-center'>{error}</p>}
					</div>
					<canvas height={SIZE} width={SIZE} className='h-full w-full' ref={canvasRef} />
				</div>
			</Card>
		</div>
	);
}
