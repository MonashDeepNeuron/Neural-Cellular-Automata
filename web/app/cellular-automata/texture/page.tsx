'use client';

import clsx from 'clsx';
import simulation from './simulation';
import useNCA from './useNCA';

const SIZE = 512;

export default function Texture() {
	const { error, status, play, setPlay, step, canvasRef } = useNCA({
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
			<button
				type='button'
				className={clsx('px-4 py-2 text-white min-w-24 rounded-md', play ? 'bg-red-700' : 'bg-green-700')}
				onClick={() => setPlay(!play)}
			>
				{play ? 'Pause' : 'Play'}
			</button>
			<p>Status: {status}</p>
			<p>Step: {step}</p>
			<div className='relative inline-block'>
				<div className='absolute top-0 left-0 flex items-center justify-center overflow-hidden w-full h-full'>
					{error && <p className='text-red-500 px-4 text-center'>{error}</p>}
				</div>
				<canvas height={SIZE} width={SIZE} ref={canvasRef} />
			</div>
		</div>
	);
}
