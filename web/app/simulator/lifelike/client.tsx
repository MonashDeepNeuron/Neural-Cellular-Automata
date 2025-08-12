'use client';
import Simulator from '@/components/layout/Simulator';
import useLTL from '@/hooks/useLTL'; // NOTE: Life-like is a subset of LTL
import patterns from '@/patterns/lifelike'; // Different patterns for Life-like
import { simulation } from '@/shaders/discrete/simulation';
import { parseLifeLikeRule } from '@/util/Parse'; // Different rulestring format for Life-like

const SIZE = 64;

export default function Client() {
	const controls = useLTL({
		size: SIZE,
		pattern: patterns[4],
		shaders: {
			simulation
		},
		parseRuleString: parseLifeLikeRule
	});

	return <Simulator name='Life-like' size={SIZE} {...controls} />;
}
