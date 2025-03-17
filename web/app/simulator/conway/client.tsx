'use client';
import Simulator from '@/components/layout/Simulator';
import useLTL from '@/hooks/useLTL'; // Note: Conway's life is a subset of Larger than Life
import patterns from '@/patterns/conway'; // Different patterns for Conway's life
import { CONWAYS_LIFE } from '@/patterns/conway'; // Different rulestring format for Conway's life
import { simulation } from '@/shaders/discrete/simulation';
const SIZE = 64;

export default function Client() {
	const controls = useLTL({
		size: SIZE,
		pattern: patterns[0],
		shaders: {
			simulation
		},
		parseRuleString: (input: string) => {
			return CONWAYS_LIFE; // Conway's game of life
		}
	});

	return <Simulator name='Conways life' size={SIZE} {...controls} />;
}
