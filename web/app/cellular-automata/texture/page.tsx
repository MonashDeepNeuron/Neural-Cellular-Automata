'use client';

import Card from '@/components/Card';
import FramerateSlider from '@/components/FramerateSlider';
import clsx from 'clsx';
import simulation from './simulation';
import useNCA, { NCAStatus } from './useNCA';

const SIZE = 512;

export default function Texture() {
	const { error, status, play, setPlay, step, FPS, setFPS, canvasRef } = useNCA({
		size: SIZE,
		channels: 12,
		hiddenChannels: 96,
		convolutions: 4,
		weightsURL: '/weights/texture.bin',
		shaders: {
			simulation
		}
	});

	return (
		<div>
			<h1>Texture</h1>
			<div className='flex gap-4 flex-row'>
				<Card>
					<p>
						<b>Status</b>: {status}
					</p>
					<p>
						<b>Step</b>: {step}
					</p>
					<FramerateSlider state={FPS} setState={setFPS} />
					<button
						type='button'
						disabled={status !== NCAStatus.READY}
						className={clsx('mt-4 px-4 py-2 text-white min-w-24 rounded-md font-bold', play ? 'bg-red-700' : 'bg-green-700')}
						onClick={() => setPlay(!play)}
					>
						{play ? 'Pause' : 'Play'}
					</button>
				</Card>
				<Card className='relative inline-block'>
					<div className='absolute top-0 left-0 flex items-center justify-center overflow-hidden w-full h-full'>
						{error && <p className='text-red-500 px-4 text-center'>{error}</p>}
					</div>
					<canvas height={SIZE} width={SIZE} ref={canvasRef} />
				</Card>
			</div>
		</div>
	);
}
