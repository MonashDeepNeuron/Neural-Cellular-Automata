'use client';

import Card from '@/components/Card';
import FramerateSlider from '@/components/FramerateSlider';
import useNCA, { NCAStatus } from '@/hooks/useNCA';
import { persisting as simulation } from '@/shaders/nca/simulation';
import clsx from 'clsx';

const SIZE = 60;

export default function PersistingGCA() {
	const { error, status, play, setPlay, step, FPS, setFPS, canvasRef } = useNCA({
		size: SIZE,
		channels: 16,
		hiddenChannels: 128,
		convolutions: 3,
		weightsURL: '/weights/persisting.bin',
		seed: true,
		shaders: {
			simulation
		}
	});

	return (
		<div>
			<h1>Persisting GCA</h1>
			<div className='flex gap-4 flex-row'>
				<Card>
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
				<Card className='relative inline-block'>
					<div className='absolute top-0 left-0 flex items-center justify-center overflow-hidden w-full h-full'>
						{error && <p className='text-red-500 px-4 text-center'>{error}</p>}
					</div>
					<canvas height={SIZE} width={SIZE} className='w-96 h-96' ref={canvasRef} style={{ imageRendering: 'pixelated' }} />
				</Card>
			</div>
		</div>
	);
}
