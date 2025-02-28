'use client';
import Simulator from '@/components/layout/Simulator';
import useNCA from '@/hooks/useNCA';
import { texture as simulation } from '@/shaders/nca/simulation';

const SIZE = 256;

export default function Texture() {
	const controls = useNCA({
		size: SIZE,
		channels: 12,
		hiddenChannels: 96,
		convolutions: 4,
		weightsURL: '/weights/texture-bubbles.bin',
		shaders: {
			simulation
		}
	});

	return <Simulator name='Texture' size={SIZE} {...controls} />;
}
