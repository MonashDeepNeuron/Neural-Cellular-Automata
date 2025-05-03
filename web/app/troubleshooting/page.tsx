import createMetadata from '@/util/createMetadata';
import Link from 'next/link';

export const metadata = createMetadata({
	title: 'Troubleshooting',
	description: 'Fix common issues with running neural cellular automata simulations.'
});

export default function Troubleshooting() {
	return (
		<div className='max-w-4xl mx-auto px-6 py-10 text-gray-800'>
			{/* Page Title */}
			<h1 className='text-4xl font-bold mb-6 text-center'>Simulation Troubleshooting</h1>

			{/* Issue 1 */}
			<section className='mb-8'>
				<h2 className='text-2xl font-semibold text-purple-mdn mb-2'>🛠️ All I See Is a White Box / No Simulation</h2>

				<h3 className='text-lg font-semibold text-gray-600 italic'>1️⃣ Browser Does Not Support WebGPU</h3>
				<p className='leading-7 text-lg mt-2'>
					Our project relies on WebGPU, a cutting-edge graphics computing tool for web development. Unfortunately, not all browsers support
					WebGPU yet.
				</p>
				<p className='mt-2'>
					<strong>Unsupported Browsers:</strong> Firefox.
				</p>
				<p className='mt-2'>
					<strong>Supported Browsers:</strong> Chrome for Windows, Microsoft Edge, and Chrome for Android.
				</p>
				<p className='mt-2'>
				<strong>Experimentally Supported Browsers:</strong> If you are on an IOS or IpadOS device, make sure you are using Safari, and follow this short tutorial
					{' '}
					<Link href='iosSucks' className='text-purple-mdn font-semibold hover:underline'>
						IOS Tutorial
					</Link>
				</p>
				<p className='mt-2'>
					To check if your browser supports WebGPU, visit{' '}
					<Link href='https://caniuse.com/webgpu' className='text-purple-mdn font-semibold hover:underline'>
						Can I Use WebGPU?
					</Link>
				</p>
				

				<h3 className='text-lg font-semibold text-gray-600 italic mt-4'>2️⃣ JavaScript Is Turned Off</h3>
				<p className='leading-7 text-lg mt-2'>
					Our website requires JavaScript to function. Please enable JavaScript in your browser settings and refresh the page.
				</p>
			</section>

			{/* Issue 2 */}
			<section className='mb-8'>
				<h2 className='text-2xl font-semibold text-purple-mdn mb-2'>⚙️ I Changed Settings and Now I See Nothing</h2>

				<h3 className='text-lg font-semibold text-gray-600 italic'>1️⃣ Rule String/Kernel Issue</h3>
				<p className='leading-7 text-lg mt-2'>
					Patterns depend on a delicate balance of life and death, and small changes can destabilize them.
				</p>
				<p>
					<strong>Solution:</strong> Try different combinations and use the 'Randomize' button to reset the grid with varied values.
				</p>

				<h3 className='text-lg font-semibold text-gray-600 italic mt-4'>2️⃣ Activation Function Changes</h3>
				<p className='leading-7 text-lg mt-2'>
					If you modified the activation function, there might be a bug in the code. Ensure the activation function follows WGSL syntax and
					returns a single `float32` value.
				</p>
				<p>
					<strong>Solution:</strong> Use your browser’s developer tools (Inspect Element) to identify and fix issues.
				</p>
			</section>

			{/* Issue 3 */}
			<section className='mb-8'>
				<h2 className='text-2xl font-semibold text-purple-mdn mb-2'>🖥️ My Screen Freezes When Running Continuous CA</h2>

				<h3 className='text-lg font-semibold text-gray-600 italic'>Device Performance Issue</h3>
				<p className='leading-7 text-lg mt-2'>
					Devices have varying capacities to handle high frame rates. If you experience lag, reduce the speed setting.
				</p>
				<p>
					<strong>Device Frame Rate Recommendations:</strong>
				</p>
				<ul className='list-disc list-inside ml-4 space-y-1 mt-2 text-lg'>
					<li>400 fps: New laptops with i7 Iris Xe graphics</li>
					<li>200 fps: Samsung Flip 4</li>
					<li>40 fps: Motorola Edge 30 (browser limitations may apply)</li>
					<li>0 fps: Apple devices (no WebGPU support)</li>
				</ul>
			</section>

			{/* Issue 4 */}
			<section className='mb-8'>
				<h2 className='text-2xl font-semibold text-purple-mdn mb-2'>🔄 Continuous CA Is Flickering at Certain Frame Rates</h2>

				<h3 className='text-lg font-semibold text-gray-600 italic'>1️⃣ Try Selecting 'Skip Every Second Frame'</h3>
				<p className='leading-7 text-lg mt-2'>
					Patterns like 'worms' and 'mitosis' often alternate between two states, causing flickering.
				</p>
				<p>
					<strong>Solution:</strong> Skipping every second frame reduces this effect and helps you observe long-term changes.
				</p>

				<h3 className='text-lg font-semibold text-gray-600 italic mt-4'>2️⃣ Why Does It Run Smoothly at Certain Frame Rates?</h3>
				<p className='leading-7 text-lg mt-2'>To prevent freezes, the display update rate is capped at 50 fps.</p>
				<ul className='list-disc list-inside ml-4 space-y-1 mt-2 text-lg'>
					<li>At 1-50, 101-150 fps, etc., you see odd frames.</li>
					<li>At 51-100, 151-200 fps, etc., you see even frames.</li>
				</ul>
				<p className='mt-2'>
					<strong>Tip:</strong> Using 'Skip Every Second Frame' ensures only even frames are displayed, providing smoother visuals.
				</p>
			</section>

			{/* Contact Us Link */}
			<div className='text-center mt-8'>
				<Link href='/contact' className='text-purple-mdn font-bold hover:underline hover:text-purple-mdn transition duration-300'>
					📩 See Next: Contact Us
				</Link>
			</div>
		</div>
	);
}
