'use client';
import Simulator from '@/components/layout/Simulator';
import useNCA from '@/hooks/useNCA';
import { growing as simulation } from '@/shaders/nca/simulation';

const SIZE = 60;

export default function PersistingGCA() {
	const controls = useNCA({
		size: SIZE,
		channels: 16,
		hiddenChannels: 128,
		convolutions: 3,
		weightsURL: '/weights/persist_cat20000.bin',
		seed: true,
		shaders: {
			simulation
		}
	});

	return <Simulator name='Growing' size={SIZE} {...controls} />;
}
