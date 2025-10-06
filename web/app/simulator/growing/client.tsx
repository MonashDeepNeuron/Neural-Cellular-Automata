'use client';
import Simulator from '@/components/layout/Simulator';
import useNCA from '@/hooks/useNCA';
import { growing as simulation } from '@/shaders/nca/simulation';

const SIZE = 48;

export default function PersistingGCA() {
	const controls = useNCA({
		size: SIZE,
		channels: 16,
		hiddenChannels: 128,
		convolutions: 3,
		weightsURL: '/weights/growing.bin',
		seed: true,
		shaders: {
			simulation
		}
	});

	return <Simulator name='Growing' size={SIZE} className='-rotate-90' resetStateStep={500} {...controls} />;
}
