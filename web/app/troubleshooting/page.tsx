import { AlertCircle, Monitor, RefreshCw, Settings } from 'lucide-react';
import Link from 'next/link';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import createMetadata from '@/util/createMetadata';

export const metadata = createMetadata({
	title: 'Troubleshooting',
	description: 'Fix common issues with running neural cellular automata simulations.'
});

const Troubleshooting = () => {
	return (
		<div className='min-h-screen bg-background'>
			<main className='pt-24 pb-16'>
				<div className='container mx-auto px-6'>
					{/* Header */}
					<div className='max-w-4xl mx-auto mb-12 text-center'>
						<h1 className='text-4xl md:text-5xl font-bold mb-4 bg-clip-text text-primary'>Simulation Troubleshooting</h1>
						<p className='text-lg text-muted-foreground'>Common issues and solutions for running NCA simulations</p>
					</div>

					{/* Troubleshooting Sections */}
					<div className='max-w-4xl mx-auto space-y-6'>
						{/* White Box Issue */}
						<Card>
							<CardHeader>
								<CardTitle className='flex items-center gap-2'>
									<AlertCircle className='w-5 h-5 text-primary' />
									All I See Is a White Box / No Simulation
								</CardTitle>
							</CardHeader>
							<CardContent className='space-y-6'>
								<div>
									<h3 className='font-semibold text-lg mb-2'>1️⃣ Browser Does Not Support WebGPU</h3>
									<p className='text-muted-foreground mb-3'>
										Our project relies on WebGPU, a cutting-edge graphics computing tool for web development. Unfortunately, not all
										browsers support WebGPU yet.
									</p>
									<ul className='list-disc list-inside space-y-1 text-muted-foreground ml-4'>
										<li>
											<strong className='text-foreground'>Unsupported Browsers:</strong> Firefox
										</li>
										<li>
											<strong className='text-foreground'>Supported Browsers:</strong> Chrome for Windows, Microsoft Edge, and Chrome for
											Android
										</li>
										<li>
											<strong className='text-foreground'>Experimentally Supported Browsers:</strong> If you are on an iOS or iPadOS device,
											make sure you are using Safari, and follow this short tutorial:{' '}
											<Link href='/ios-sucks' className='text-primary hover:underline'>
												iOS Tutorial
											</Link>
										</li>
									</ul>
									<p className='mt-3 text-muted-foreground'>
										To check if your browser supports WebGPU, visit{' '}
										<a href='https://caniuse.com/webgpu' className='text-primary hover:underline' target='_blank' rel='noopener noreferrer'>
											Can I Use WebGPU?
										</a>
									</p>
								</div>

								<div>
									<h3 className='font-semibold text-lg mb-2'>2️⃣ JavaScript Is Turned Off</h3>
									<p className='text-muted-foreground'>
										Our website requires JavaScript to function. Please enable JavaScript in your browser settings and refresh the page.
									</p>
								</div>
							</CardContent>
						</Card>

						{/* Settings Issue */}
						<Card>
							<CardHeader>
								<CardTitle className='flex items-center gap-2'>
									<Settings className='w-5 h-5 text-primary' />I Changed Settings and Now I See Nothing
								</CardTitle>
							</CardHeader>
							<CardContent className='space-y-6'>
								<div>
									<h3 className='font-semibold text-lg mb-2'>1️⃣ Rule String/Kernel Issue</h3>
									<p className='text-muted-foreground mb-3'>
										Patterns depend on a delicate balance of life and death, and small changes can destabilize them.
									</p>
									<p className='text-muted-foreground'>
										<strong className='text-foreground'>Solution:</strong> Try different combinations and use the 'Randomize' button to
										reset the grid with varied values.
									</p>
								</div>

								<div>
									<h3 className='font-semibold text-lg mb-2'>2️⃣ Activation Function Changes</h3>
									<p className='text-muted-foreground mb-3'>
										If you modified the activation function, there might be a bug in the code. Ensure the activation function follows WGSL
										syntax and returns a single <code className='bg-muted px-1 rounded'>float32</code> value.
									</p>
									<p className='text-muted-foreground'>
										<strong className='text-foreground'>Solution:</strong> Use your browser's developer tools (Inspect Element) to identify
										and fix issues.
									</p>
								</div>
							</CardContent>
						</Card>

						{/* Freezing Issue */}
						<Card>
							<CardHeader>
								<CardTitle className='flex items-center gap-2'>
									<Monitor className='w-5 h-5 text-primary' />
									My Screen Freezes When Running Continuous CA
								</CardTitle>
							</CardHeader>
							<CardContent>
								<h3 className='font-semibold text-lg mb-2'>Device Performance Issue</h3>
								<p className='text-muted-foreground mb-3'>
									Devices have varying capacities to handle high frame rates. If you experience lag, reduce the speed setting.
								</p>
								<p className='font-semibold mb-2'>Device Frame Rate Recommendations:</p>
								<ul className='list-disc list-inside space-y-1 text-muted-foreground ml-4'>
									<li>400 fps: New laptops with i7 Iris Xe graphics</li>
									<li>200 fps: Samsung Flip 4</li>
									<li>40 fps: Motorola Edge 30 (browser limitations may apply)</li>
									<li>0 fps: Apple devices (no WebGPU support)</li>
								</ul>
							</CardContent>
						</Card>

						{/* Flickering Issue */}
						<Card>
							<CardHeader>
								<CardTitle className='flex items-center gap-2'>
									<RefreshCw className='w-5 h-5 text-primary' />
									Continuous CA Is Flickering at Certain Frame Rates
								</CardTitle>
							</CardHeader>
							<CardContent className='space-y-6'>
								<div>
									<h3 className='font-semibold text-lg mb-2'>1️⃣ Try Selecting 'Skip Every Second Frame'</h3>
									<p className='text-muted-foreground mb-3'>
										Patterns like 'worms' and 'mitosis' often alternate between two states, causing flickering.
									</p>
									<p className='text-muted-foreground'>
										<strong className='text-foreground'>Solution:</strong> Skipping every second frame reduces this effect and helps you
										observe long-term changes.
									</p>
								</div>

								<div>
									<h3 className='font-semibold text-lg mb-2'>2️⃣ Why Does It Run Smoothly at Certain Frame Rates?</h3>
									<p className='text-muted-foreground mb-3'>To prevent freezes, the display update rate is capped at 50 fps.</p>
									<ul className='list-disc list-inside space-y-1 text-muted-foreground ml-4'>
										<li>At 1-50, 101-150 fps, etc., you see odd frames.</li>
										<li>At 51-100, 151-200 fps, etc., you see even frames.</li>
									</ul>
									<p className='mt-3 text-muted-foreground'>
										<strong className='text-foreground'>Tip:</strong> Using 'Skip Every Second Frame' ensures only even frames are
										displayed, providing smoother visuals.
									</p>
								</div>
							</CardContent>
						</Card>
					</div>
				</div>
			</main>
		</div>
	);
};

export default Troubleshooting;
