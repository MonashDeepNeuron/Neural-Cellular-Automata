'use client';

import Image from 'next/image';
import Link from 'next/link';

export default function Home() {
	return (
		<div className='min-h-screen flex flex-col items-center justify-center bg-gray-50 p-6'>
			{/* Side-by-Side Layout */}
			<div className='flex flex-col-2 gap-12 max-w-6xl w-full text-center'>
				{/* Simulation Section */}
				<div className='flex flex-col w-full md:w-1/2 justify-center'>
					<h1 className='text-3xl font-bold text-gray-800 mb-4'>🧪 Neural Cellular Automata Simulator</h1>
					<Link href='cellular-automata'>
						<Image
							src='images/semitrained-cat.png'
							alt='Target Knitted Texture'
							height={60}
							width={50}
							className='w-full rounded-md shadow'
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
								<Link
									href='/cellular-automata'
									className='text-purple-mdn font-semibold hover:underline hover:text-purple-mdn-dark transition'
								>
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
						<h2 className='text-2xl font-semibold text-gray-700'>🤝 Join Us!</h2>
						<p className='text-gray-600 mt-1'>Interested in working on this project? We'd love to hear from you!</p>
						<Link
							href='https://www.deepneuron.org/contact-us'
							className='inline-block bg-purple-mdn text-white px-6 py-2 rounded-md mt-3 hover:bg-purple-mdn-dark transition'
						>
							Get in Touch
						</Link>
					</div>
				</div>
			</div>
		</div>
	);
}
