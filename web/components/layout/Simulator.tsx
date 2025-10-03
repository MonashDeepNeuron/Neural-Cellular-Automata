'use client';
import clsx from 'clsx';
import { useId } from 'react';
import { CAStatus, type NCAControls } from '@/hooks/useNCA';
import { Button } from '../ui/button';
import { Card } from '../ui/card';
import { Slider } from '../ui/slider';
import { Checkbox } from '../ui/checkbox';
import { Label } from '../ui/label';
import { Play, Pause, AlertCircle } from 'lucide-react';
import { Alert, AlertDescription } from '../ui/alert';

interface SimulatorProps extends NCAControls {
	name: string;
	className?: string;
	size: number;
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
	className
}: SimulatorProps) {
	const checkboxId = useId();

	return (
		<div className='container mx-auto px-6 py-24'>
			<div className='grid lg:grid-cols-[20rem_1fr] gap-6 max-w-7xl mx-auto'>
				{/* Controls Panel */}
				<Card className='p-6 bg-card/50 backdrop-blur-sm border-primary/20 h-fit'>
					<h1 className='text-2xl font-bold mb-6'>{name}</h1>
					
					{/* Status */}
					<div className='space-y-2 mb-6 pb-6 border-b border-border'>
						<div className='flex items-center justify-between text-sm'>
							<span className='text-muted-foreground'>Status</span>
							<span className={clsx(
								'font-medium',
								status === CAStatus.READY && 'text-green-500',
								status === CAStatus.ALLOCATING_RESOURCES && 'text-yellow-500',
								status === CAStatus.FAILED && 'text-destructive'
							)}>
								{status}
							</span>
						</div>
						<div className='flex items-center justify-between text-sm'>
							<span className='text-muted-foreground'>Step</span>
							<span className='font-mono'>{step}</span>
						</div>
					</div>

					{/* Frame Rate */}
					<div className='space-y-3 mb-6'>
						<div className='flex items-center justify-between'>
							<Label>Frame Rate</Label>
							<span className='text-sm font-mono text-muted-foreground'>{FPS} FPS</span>
						</div>
						<Slider
							value={[FPS]}
							onValueChange={(value) => setFPS(value[0])}
							max={240}
							min={1}
							step={1}
						/>
					</div>

					{/* Skip Frame */}
					<div className='flex items-center gap-3 p-3 mb-6 rounded-lg bg-muted/50 border border-border/50'>
						<Checkbox
							id={checkboxId}
							checked={stepsPerFrame === 2}
							onCheckedChange={(checked) => setStepsPerFrame(checked ? 2 : 1)}
						/>
						<Label htmlFor={checkboxId} className='cursor-pointer'>
							Skip every second frame
						</Label>
					</div>

					{/* Play/Pause */}
					<Button
						variant={play ? 'pause' : 'play'}
						size='lg'
						className='w-full'
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