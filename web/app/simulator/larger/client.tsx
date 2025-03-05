'use client';
import Simulator from '@/components/layout/Simulator';
import useLTL from '@/hooks/useLTL';
import patterns from '@/patterns';
import { computeShader as simulation } from '@/shaders/discrete/simulation';
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
