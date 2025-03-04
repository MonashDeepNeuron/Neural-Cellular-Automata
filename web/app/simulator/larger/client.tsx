'use client';
import Simulator from '@/components/layout/SimulatorDiscrete';
import useLTL from '@/hooks/useLTL';
import { computeShader as simulation } from '@/shaders/discrete/simulation';
import patterns from '@/patterns'
const SIZE = 60;

export default function Client() {
	const controls = useLTL({
		size: SIZE,
		pattern: patterns[3],
		shaders: {
			simulation
		}
	});

	return <Simulator name='LTL' size={SIZE} {...controls} />;
}
