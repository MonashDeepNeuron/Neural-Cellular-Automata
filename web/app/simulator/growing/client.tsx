'use client';
import { useEffect, useId, useState } from 'react';
import Simulator from '@/components/layout/Simulator';
import { Checkbox } from '@/components/ui/checkbox';
import { Label } from '@/components/ui/label';
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

	const [reset, setReset] = useState(false);
	const checkboxId = useId();

	useEffect(() => {
		if (reset && controls.step > 500) controls.resetState();
	}, [reset, controls.step, controls.resetState]);

	return (
		<Simulator name='Growing' size={SIZE} className='-rotate-90' {...controls}>
			<div className='flex items-center gap-3 p-3 rounded-lg bg-muted/50 border border-border/50 cursor-pointer'>
				<Checkbox id={checkboxId} checked={reset} onCheckedChange={checked => setReset(Boolean(checked))} className='cursor-pointer' />
				<Label htmlFor={checkboxId} className='cursor-pointer'>
					Reset state periodically
				</Label>
			</div>
		</Simulator>
	);
}
