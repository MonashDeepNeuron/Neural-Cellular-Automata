'use client';
import clsx from 'clsx';
import { AlertCircle, Pause, Play } from 'lucide-react';
import { type ReactNode, useEffect, useId, useState } from 'react';
import { CAStatus, type NCAControls } from '@/hooks/useNCA';
import { Alert, AlertDescription } from '../ui/alert';
import { Button } from '../ui/button';
import { Card } from '../ui/card';
import { Checkbox } from '../ui/checkbox';
import { Label } from '../ui/label';
import { Slider } from '../ui/slider';

interface SimulatorProps extends NCAControls {
	name: string;
	className?: string;
	size: number;
	children?: ReactNode;
	resetStateStep?: number;
}

export default function Simulator({
	name,
	FPS,
	setFPS,
	setPlay,
	play,
	error,
	canvasRef,
	size,
	step,
	status,
	stepsPerFrame,
	setStepsPerFrame,
	className,
	children,
	resetState,
	resetStateStep = 0
}: SimulatorProps) {
	const skipId = useId();
	const resetId = useId();

	const [reset, setReset] = useState(false);

	useEffect(() => {
		if (reset && resetStateStep && step > resetStateStep) resetState();
	}, [reset, step, resetState, resetStateStep]);

	return (
		<div className='container mx-auto px-6 py-24'>
			<div className='grid lg:grid-cols-[20rem_1fr] gap-6 max-w-7xl mx-auto'>
				{/* Controls Panel */}
				<Card className='p-6 bg-card/50 backdrop-blur-sm border-primary/20 h-full'>
					<h1 className='text-2xl font-bold'>{name}</h1>

					{/* Status */}
					<div className='space-y-2 pb-6 border-b border-border'>
						<div className='flex items-center justify-between text-sm'>
							<span className='text-muted-foreground'>Status</span>
							<span
								className={clsx(
									'font-medium',
									status === CAStatus.READY && 'text-green-500',
									status === CAStatus.ALLOCATING_RESOURCES && 'text-yellow-500',
									status === CAStatus.FAILED && 'text-destructive'
								)}
							>
								{status}
							</span>
						</div>
						<div className='flex items-center justify-between text-sm'>
							<span className='text-muted-foreground'>Step</span>
							<span className='font-mono'>{step}</span>
						</div>
					</div>

					{/* Frame Rate */}
					<div className='space-y-3'>
						<div className='flex items-center justify-between'>
							<Label>Frame Rate</Label>
							<span className='text-sm font-mono text-muted-foreground'>{FPS} FPS</span>
						</div>
						<Slider value={[FPS]} onValueChange={value => setFPS(value[0])} max={240} min={1} step={1} />
					</div>

					{/* Skip Frame */}
					<div className='flex items-center gap-3 p-3 rounded-lg bg-muted/50 border border-border/50 cursor-pointer'>
						<Checkbox
							id={skipId}
							checked={stepsPerFrame === 2}
							onCheckedChange={checked => setStepsPerFrame(checked ? 2 : 1)}
							className='cursor-pointer'
						/>
						<Label htmlFor={skipId} className='cursor-pointer'>
							Skip every second frame
						</Label>
					</div>

					{/* Reset State */}
					{resetStateStep > 0 && (
						<div className='flex items-center gap-3 p-3 rounded-lg bg-muted/50 border border-border/50 cursor-pointer'>
							<Checkbox id={resetId} checked={reset} onCheckedChange={checked => setReset(Boolean(checked))} className='cursor-pointer' />
							<Label htmlFor={resetId} className='cursor-pointer'>
								Reset state periodically
							</Label>
						</div>
					)}

					{children}

					{/* Play/Pause */}
					<Button
						variant={play ? 'pause' : 'play'}
						size='lg'
						className='w-full cursor-pointer mt-auto'
						onClick={() => setPlay(!play)}
						disabled={status !== CAStatus.READY}
					>
						{play ? (
							<>
								<Pause className='w-4 h-4 mr-2' />
								Pause
							</>
						) : (
							<>
								<Play className='w-4 h-4 mr-2' />
								Start
							</>
						)}
					</Button>
				</Card>

				{/* Canvas */}
				<Card className='p-6 bg-card/50 backdrop-blur-sm border-primary/20 flex items-center justify-center'>
					<div className='relative w-full max-w-xl aspect-square'>
						{error && (
							<Alert variant='destructive' className='absolute top-0 left-0 right-0 z-10 m-4'>
								<AlertCircle className='h-4 w-4' />
								<AlertDescription>{error}</AlertDescription>
							</Alert>
						)}
						<canvas
							height={size}
							width={size}
							className={clsx('w-full h-full rounded-lg', error && 'opacity-50', className)}
							ref={canvasRef}
							style={{ imageRendering: 'pixelated' }}
						/>
					</div>
				</Card>
			</div>
		</div>
	);
}
