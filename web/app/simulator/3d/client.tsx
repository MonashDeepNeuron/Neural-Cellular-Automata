'use client';
import Simulator from '@/components/layout/Simulator';
import use3D from '@/hooks/use3D';

const SIZE = 1024;

export default function Client() {
	const controls = use3D({
		size: SIZE
	});

	return <Simulator name='3D' size={SIZE} {...controls} />;
}
