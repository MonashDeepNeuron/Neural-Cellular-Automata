import Link from 'next/link';

export default function Troubleshooting() {
	return (
		<>
			{/* Title */}
			<div className='text-column'>
				<h1>Potential Issues</h1>
			</div>

			<div className='centre'>
				<h1>Simulation Troubleshooting</h1>

				{/* Issue 1 */}
				<h2>All I See Is a White Box / I Don’t See the Simulation</h2>
				<h3>
					<i>Browser Does Not Support WebGPU</i>
				</h3>
				<p>
					Our project relies on WebGPU, a cutting-edge graphics computing tool for web development. Unfortunately, not all browsers support
					WebGPU yet.
				</p>
				<p>
					<b>Unsupported Browsers:</b> Safari (including all browsers on iPhone/iPad) and Firefox.
				</p>
				<p>
					<b>Supported Browsers:</b> Chrome for Windows, Microsoft Edge, and Chrome for Android.
				</p>
				<p>
					to check if your browser supports WebGPU, visit
					<a href='https://caniuse.com/webgpu' target='_blank' rel='noopener noreferrer'>
						{' '}
						Can I Use WebGPU?
					</a>
					.
				</p>

				<h3>
					<i>JavaScript Is Turned Off</i>
				</h3>
				<p>Our website requires JavaScript to function. Please enable JavaScript in your browser settings and refresh the page.</p>

				{/* Issue 2 */}
				<h2>I Changed the Settings and Now I See Nothing</h2>
				<h3>
					<i>Rule String/Kernel Is Not Producing Interesting Behavior</i>
				</h3>
				<p>
					Patterns depend on a delicate balance of life and death, and small changes can destabilize them. They also require variation in
					grid values to work effectively.
				</p>
				<p>
					<b>Solution:</b> Try different combinations and use the "Randomize" button to reset the grid with varied values.
				</p>

				<h3>
					<i>Activation Function Changes</i>
				</h3>
				<p>
					If you modified the activation function, there might be a bug in the code. The activation function must follow WGSL syntax (WebGPU
					code) and return a single `float32` value.
				</p>
				<p>
					<b>Solution:</b> Use your browser's developer tools (inspect element) to identify and fix the issue.
				</p>

				{/* Issue 3 */}
				<h2>My Screen Freezes When Running Continuous CA</h2>
				<h3>
					<i>Speed Setting Is Too High for Your Device</i>
				</h3>
				<p>Devices have varying capacities to handle high frame rates. If you experience lag, reduce the speed setting.</p>
				<p>
					<b>Device Frame Rate Recommendations:</b>
				</p>
				<ul>
					<li>400 fps: New laptops with i7 Iris Xe graphics</li>
					<li>200 fps: Samsung Flip 4</li>
					<li>40 fps: Motorola Edge 30 (browser limitations may apply)</li>
					<li>0 fps: Apple devices (no WebGPU support)</li>
				</ul>

				{/* Issue 4 */}
				<h2>Continuous CA Is Really Flickery for Some Frame Rates</h2>
				<h3>
					<i>Try Selecting 'Skip Every Second Frame'</i>
				</h3>
				<p>Patterns like "worms" and "mitosis" often alternate between two states, which can cause flickering.</p>
				<p>
					<b>Solution:</b> Skipping every second frame reduces this effect and helps you observe long-term changes.
				</p>

				<h3>
					<i>Why Does It Run Smoothly at Certain Frame Rates?</i>
				</h3>
				<p>to prevent freezes, the display update rate is capped at 50fps.</p>
				<p>
					<b>Behavior:</b>
				</p>
				<ul>
					<li>At 1-50, 101-150 fps, etc., you see odd frames.</li>
					<li>At 51-100, 151-200 fps, etc., you see even frames.</li>
				</ul>
				<p>
					<b>Using 'Skip Every Second Frame':</b> This ensures only even frames are displayed, doubling updates between frames for a
					smoother experience.
				</p>

				{/* Contact Us Link */}
				<p style={{ textAlign: 'center', marginTop: '20px' }}>
					<Link href='/contact' style={{ fontWeight: 'bold' }}>
						See Next: Contact Us
					</Link>
				</p>
			</div>
		</>
	);
}
