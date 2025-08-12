import Image from 'next/image';
import Link from 'next/link';

export default function Home() {
	return (
		<div className='min-h-screen flex flex-col items-center justify-center bg-gray-50 p-6'>
			{/* Side-by-Side Layout */}
			<div className='flex flex-col-2 gap-6 max-w-7xl w-full text-center'>
				{/* Simulation Section */}
				<div className='flex flex-col w-full md:w-1/2'>
					<h1 className='text-3xl font-bold text-gray-800 mb-4'>🧪 Neural Cellular Automata Simulator</h1>
					<Link href='simulator'>
						<Image
							src='images/semitrained-cat.png'
							alt='Target Knitted Texture'
							height={60}
							width={50}
							className='w-4/5 rounded-md items-center shadow mx-10'
						/>
					</Link>
					<p className='text-gray-600 mt-2'>Experience the dynamics of Neural Cellular Automata in real-time.</p>
				</div>

				{/* Main Content */}
				<div className='w-full md:w-1/2 text-center md:text-left space-y-4'>
					<h2 className='text-3xl font-bold text-gray-800'>Welcome to Neural Cellular Automata</h2>
					<p className='text-gray-600 leading-relaxed'>
						We are a research project team under <strong>Monash DeepNeuron</strong>, exploring the potential of Neural Cellular Automata
						(NCA) for various applications. Our goal is to understand, simulate, and improve NCA models.
					</p>

					{/* Explore Section */}
					<div>
						<h2 className='text-2xl font-semibold text-gray-700'>🌐 Explore More:</h2>
						<ul className='space-y-2 mt-2'>
							<li>
								<Link href='/learn' className='text-purple-mdn font-semibold hover:underline hover:text-purple-mdn-dark transition'>
									📘 What is Neural Cellular Automata?
								</Link>
							</li>
							<li>
								<Link
									href='/learn/research'
									className='text-purple-mdn font-semibold hover:underline hover:text-purple-mdn-dark transition'
								>
									🔬 Our Research & Latest Findings
								</Link>
							</li>
							<li>
								<Link href='/simulator' className='text-purple-mdn font-semibold hover:underline hover:text-purple-mdn-dark transition'>
									🧪 Try the NCA Simulator
								</Link>
							</li>

							{/* TODO: re-write keeping up */}
							{/* <li>
								<Link href="/keeping-up" className="text-blue-500 hover:underline hover:text-blue-700 transition">
									📰 Project Updates
								</Link>
							</li> */}
						</ul>
					</div>

					{/* Join Us Section */}
					<div className='mt-4'>
						<h2 className='text-2xl font-semibold text-gray-700'>Join Us!</h2>
						<p className='text-gray-600 mt-1'>
							Are you a Monash Engineering or IT student interested in working on this project? Reach out to be informed when new positions
							open up. First-year or Master's students — all are welcome!
						</p>
						<Link
							href='https://docs.google.com/forms/d/e/1FAIpQLSckOGpNS-nFOxB4cGHmXC2z04D6_m8j26qKLZee3bZ298vNWg/viewform?usp=sharing'
							className='inline-block bg-purple-mdn text-white px-6 py-3 rounded-md shadow-md hover:bg-purple-mdn-dark transition-transform duration-300 transform hover:scale-105'
						>
							Join the Team
						</Link>
					</div>

					{/* Get In Touch Section */}
					<div className='mt-4'>
						<h2 className='text-2xl font-semibold text-gray-700'>Get in Touch</h2>
						<p className='text-gray-600 mt-1'>Want to learn more about projects like this, or interested in a collaboration?</p>
						<Link
							href='https://www.deepneuron.org/contact-us'
							className='inline-block bg-purple-mdn text-white px-6 py-3 rounded-md shadow-md hover:bg-purple-mdn-dark transition-transform duration-300 transform hover:scale-105'
						>
							Get in Touch
						</Link>
					</div>
				</div>
			</div>
		</div>
	);
}
