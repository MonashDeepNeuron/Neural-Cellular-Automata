'use client';
import clsx from 'clsx';
import { AlertCircle, Pause, Play } from 'lucide-react';
import { useId } from 'react';
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
		<div className='grid pt-12 gap-6 grid-rows-[auto_1fr] grid-cols-1 max-w-full lg:grid-rows-1 lg:grid-cols-[26rem_1fr] lg:h-[calc(100vh-6rem)]'>
			{/* Controls Panel */}
			<Card className='p-6 bg-card/50 backdrop-blur-sm border-primary/20 h-fit lg:h-full flex flex-col'>
				<div className='space-y-6 flex-1'>
					{/* Header */}
					<div className='pb-4 border-b border-border'>
						<h1 className='text-2xl font-bold mb-2'>{name}</h1>
						<div className='flex items-center gap-2 text-sm'>
							<span className='text-muted-foreground'>Status:</span>
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
						<div className='flex items-center gap-2 text-sm'>
							<span className='text-muted-foreground'>Step:</span>
							<span className='font-mono font-medium'>{step}</span>
						</div>
					</div>

					{/* Framerate Control */}
					<div className='space-y-3'>
						<div className='flex items-center justify-between'>
							<Label className='text-sm font-semibold'>Frame Rate</Label>
							<span className='text-sm font-mono text-muted-foreground'>{FPS} FPS</span>
						</div>
						<Slider value={[FPS]} onValueChange={value => setFPS(value[0])} max={240} min={1} step={1} className='w-full' />
					</div>

					{/* Skip Frame Option */}
					<div className='flex items-center space-x-3 p-3 rounded-lg bg-muted/50 border border-border/50'>
						<Checkbox id={checkboxId} checked={stepsPerFrame === 2} onCheckedChange={checked => setStepsPerFrame(checked ? 2 : 1)} />
						<Label
							htmlFor={checkboxId}
							className='text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 cursor-pointer'
						>
							Skip every second frame
						</Label>
					</div>

					{/* Play/Pause Button */}
					<div className='pt-4'>
						<Button
							variant={play ? 'pause' : 'play'}
							size='lg'
							className='w-full font-semibold'
							onClick={() => setPlay(!play)}
							disabled={status !== CAStatus.READY}
						>
							{play ? (
								<>
									<Pause className='w-4 h-4 mr-2' />
									Pause Simulation
								</>
							) : (
								<>
									<Play className='w-4 h-4 mr-2' />
									Start Simulation
								</>
							)}
						</Button>
					</div>
				</div>
			</Card>

			{/* Canvas Panel */}
			<Card className='p-6 bg-card/50 backdrop-blur-sm border-primary/20 flex items-center justify-center overflow-hidden'>
				<div className='relative aspect-square w-full lg:w-auto lg:h-full max-h-full max-w-full'>
					{error && (
						<Alert variant='destructive' className='absolute top-4 left-4 right-4 z-10'>
							<AlertCircle className='h-4 w-4' />
							<AlertDescription>{error}</AlertDescription>
						</Alert>
					)}
					<canvas
						height={size}
						width={size}
						className={clsx('h-full w-full rounded-lg shadow-lg', error && 'opacity-50', className)}
						ref={canvasRef}
						style={{ imageRendering: 'pixelated' }}
					/>
				</div>
			</Card>
		</div>
	);
}
