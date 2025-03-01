'use client';
import Simulator from '@/components/layout/Simulator';
import useContinuous from '@/hooks/useContinuous';
import simulation from '@/shaders/continuous/simulation';

const SIZE = 512;

export default function PersistingGCA() {
	const controls = useContinuous({
		size: SIZE,
		shaders: {
			simulation: simulation()
		}
	});

	return <Simulator name='Continuous' size={SIZE} {...controls} />;
}
