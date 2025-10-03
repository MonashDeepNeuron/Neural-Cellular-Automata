'use client';
import Simulator from '@/components/layout/Simulator';
import useLTL from '@/hooks/useLTL';
import patterns from '@/patterns/larger';
import { simulation } from '@/shaders/discrete/simulation';
import { parseLTLRule } from '@/util/Parse';

const SIZE = 64;

export default function Client() {
	const controls = useLTL({
		size: SIZE,
		pattern: patterns[3],
		shaders: {
			simulation
		},
		parseRuleString: parseLTLRule
	});

	return <Simulator name='Larger than Life' size={SIZE} {...controls} />;
}
