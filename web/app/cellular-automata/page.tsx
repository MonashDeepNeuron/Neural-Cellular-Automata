'use client'; // 👈 This tells Next.js to treat this as a Client Component

import Card from '@/components/Card';
import Link from 'next/link';
import { useState } from 'react';

import Image from 'next/image';

export default function CellularAutomata() {
	const [showWarning, setShowWarning] = useState(true);
	return (
		<div className='max-w-4xl mx-auto px-6 py-10 text-gray-800'>
			{/* Title */}
			<div className='text-center mb-8'>
				<h1 className='text-4xl font-extrabold text-gray-800'>Neural Cellular Automata Simulator</h1>
				<p className='text-md text-gray-600 mt-2'>Explore dynamic models in real-time!</p>
			</div>

			{/* Warning Message */}
			{showWarning && (
				<div className='bg-yellow-100 border-l-4 border-yellow-500 text-yellow-800 p-4 rounded-md shadow-md mb-6 transition-opacity duration-500'>
					<p className='mb-3'>
						<strong>⚠️ Warning:</strong> This website contains content that may <u>flash at high frequencies</u>. Please use discretion when
						selecting frame rates if sensitive to flashing visuals.
					</p>
					<button
						type='button'
						onClick={() => setShowWarning(false)}
						className='bg-yellow-500 text-white px-4 py-2 rounded-md hover:bg-yellow-600 transition-transform duration-300 transform hover:scale-105'
					>
						Dismiss
					</button>
				</div>
			)}

			{/* Model Selection Panel */}
			<section className='mb-16 mt-5'>
				<Card className='bg-gray-100'>
					<h3 className='text-2xl font-semibold text-gray-800 mb-4'>Select a Model:</h3>

					<div className='grid grid-cols-1 sm:grid-cols-2 gap-4'>
						{[
							{ name: 'Classic Conway', link: '/CAs/ConwaysLife/life.html' },
							{ name: 'Life Like', link: '/CAs/LifeLike/life.html' },
							{ name: 'Larger', link: '/CAs/Larger/life.html' },
							{ name: 'Continuous', link: '/CAs/Continuous/life.html' },
							{ name: 'Growing NCA', link: 'cellular-automata/persisting-gca' },
							{ name: 'Texture NCA', link: 'cellular-automata/texture' }
						].map(model => (
							<Link
								key={model.name}
								href={model.link}
								className='bg-purple-mdn text-white text-center py-3 rounded-md shadow hover:bg-purple-mdn-dark hover:scale-105 transition-transform duration-300'
							>
								{model.name}
							</Link>
						))}
					</div>
				</Card>

				{/* Navigation to Troubleshooting */}
				<div className='text-center mt-8'>
					<Link
						href='/troubleshooting'
						className='inline-block bg-purple-mdn text-white px-6 py-3 rounded-md shadow-md hover:bg-purple-mdn-dark transition-transform duration-300 transform hover:scale-105'
					>
						🛠️ Go to Troubleshooting
					</Link>
				</div>
			</section>

			{/* Title */}
			<h1 className='text-4xl font-bold mb-6 text-center'>Cellular Automata Models</h1>

			{/* Neural Cellular Automata Section */}
			<section className='mb-10'>
				<h2 className='text-2xl font-semibold text-purple-mdn mb-3'>Neural Cellular Automata (NCA)</h2>

				<p className='leading-7 text-lg mt-4'>
					Neural Cellular Automata (NCA) are a category of cellular automata that involve using a neural network as the cell’s update rule.
					The neural network can be trained to determine how to update the cell’s value in coordination with other cells, operating on the
					same rule to produce a target behavior.
				</p>

				<p className='leading-7 text-lg mt-4'>
					From a deep learning perspective, NCA can be characterized as a Recurrent Convolutional Neural Network. Learn more about this{' '}
					<Link href='/intro' className='text-purple-mdn font-semibold hover:underline hover:text-purple-mdn-dark transition'>
						here
					</Link>
					.
				</p>
			</section>

			{/* Growing Neural Cellular Automata Section */}
			<section className='mb-10'>
				<h2 className='text-2xl font-semibold text-purple-mdn mb-3'>Growing NCA</h2>

				<Link
					href='cellular-automata/persisting-gca'
					className='inline-block bg-purple-mdn text-white px-6 py-2 rounded-md shadow-md hover:bg-purple-mdn-dark transition duration-300 mb-4'
				>
					Explore G-NCA
				</Link>

				<p className='leading-7 text-lg mt-4'>
					One of the best examples of NCA is{' '}
					<Link
						href='https://distill.pub/2020/growing-ca/'
						className='text-purple-mdn font-semibold hover:underline hover:text-purple-mdn-dark transition'
					>
						Growing Neural Cellular Automata
					</Link>{' '}
					(A. Mordvintsev et al., 2020), where they trained NCA to ‘grow’ target images from a single seed cell.
				</p>

				<p className='leading-7 text-lg mt-4'>
					The Growing-NCA model emphasises that the perception of only neighbouring cells bears parallels with how natural cells communicate
					within living organisms. Interestingly, this results in Growing Neural Cellular Automata (and other NCA) also showing natural
					regenerative properties when the image is disturbed during generation.
				</p>

				<p className='leading-7 text-lg mt-4'>
					Designed to cut the model back to as simple and small a system as possible, Growing NCA is designed as a demonstration of how
					exceedingly simple systems can robustly self-organise into a very complex system without a large amount of information and very
					simple instructions.
				</p>

				<p className='leading-7 text-lg mt-4'>
					Understanding this model is a good foundation for Neural Cellular Automata, as a large portion of NCA use a very similar base
					structure, with varying objective functions. They incorporate relevant optimisations and recombination.
				</p>
			</section>

			{/* Texture NCA */}
			<section className='mb-10'>
				<h2 className='text-2xl font-semibold text-purple-mdn mb-3'>Texture NCA</h2>

				<Link
					href='cellular-automata/texture'
					className='inline-block bg-purple-mdn text-white px-6 py-2 rounded-md shadow-md hover:bg-purple-mdn-dark transition duration-300 mb-4'
				>
					Explore Textures
				</Link>

				<p className='leading-7 text-lg mt-4'>
					Texture NCA is based off the paper{' '}
					<Link
						href='https://arxiv.org/pdf/2105.07299'
						className='text-purple-mdn font-semibold hover:underline hover:text-purple-mdn-dark transition'
					>
						Self Organising Textures
					</Link>{' '}
					(A. Mordvintsev et al., 2021), where they trained NCA to ‘grow’ target images from a single seed cell.
				</p>

				<p className='leading-7 text-lg mt-4'>
					The main difference between Texture NCA and Growing NCA is that it aims to replicate image features on a small scale - an
					application that takes advantage of the short-range communication and organisational capabilities of NCA.
				</p>

				<p className='leading-7 text-lg mt-4'>
					It shares properties of regeneration, as well as independence of grid location, resulting in textures that can be smoothly and
					cohesively replicated over grids of any size and shape - or even onto 3D graphs, as the paper{' '}
					<Link
						href='https://doi.org/10.1145/3658127'
						className='text-purple-mdn font-semibold hover:underline hover:text-purple-mdn-dark transition'
					>
						Mesh Neural Cellular Automata
					</Link>{' '}
					(E. Pajouheshgar et al., 2024) expands.
				</p>
			</section>

			{/* Game of Life Section */}
			<section className='mb-10'>
				<div className='grid grid-cols-2 overflow-y-auto gap-4'>
					<div>
						<h2 className='text-2xl font-semibold text-purple-mdn mb-3'>John Conway’s Game of Life</h2>

						{/* TODO:Uncomment when the page for this is made. */}
						{/* <Link
							href='/CAs/ConwaysLife/life.html'
							className='inline-block bg-purple-mdn text-white px-6 py-2 rounded-md shadow-md hover:bg-purple-mdn-dark transition duration-300 mb-4'
						>
							Play Classic Conway
						</Link> */}

						<p className='leading-7 text-lg mt-4'>
							This is probably the most famous example of cellular automata. The Game of Life operates on these simple rules:
						</p>

						<ul className='list-disc list-inside space-y-2 text-lg mt-4'>
							<li>All cells are either alive or dead (1 or 0).</li>
							<li>A living cell with 2 or 3 neighbors survives.</li>
							<li>A dead cell with exactly 3 neighbors becomes alive.</li>
							<li>In all other cases, the cell dies or remains dead.</li>
						</ul>

						<p className='leading-7 text-lg mt-4'>
							Even with such simple rules, complex behaviors can emerge. Many self-sustaining patterns have been discovered. A more
							sophisticated version can be found{' '}
							<Link
								href='https://playgameoflife.com/'
								className='text-purple-mdn font-semibold hover:underline hover:text-purple-mdn-dark transition'
							>
								here
							</Link>
							.
						</p>
					</div>

					<div>
						<Image
							src='/images/GoL neighbourhood.png'
							alt='The neighbourhood of each cell consists of the cells in contact with it.'
							height={60}
							width={50}
							className='w-full rounded-md shadow'
						/>
						<p className='text-sm text-center italic'>The neighbourhood of each cell consists of the cells in contact with it.</p>
					</div>
				</div>
			</section>

			{/* Life-Like Section */}
			<section className='mb-10'>
				<h2 className='text-2xl font-semibold text-purple-mdn mb-3'>Life Like Cellular Automata</h2>

				{/* TODO: Uncomment when the page for this is made */}
				{/* <Link 
					href='/CAs/LifeLike/life.html' 
					className='inline-block bg-purple-mdn text-white px-6 py-2 rounded-md shadow-md hover:bg-purple-mdn-dark transition duration-300 mb-4'
				>
					Life Like
				</Link> */}

				<p className='leading-7 text-lg mt-4'>
					Life like CA operate very similarly to the Game of Life in that all cells are either alive or dead. However, Life Like gives you
					the freedom to choose how many cells must be alive in the neighbourhood to either survive or be born. This is specified by a{' '}
					<i>rule string</i>.
				</p>

				<p className='leading-7 text-lg mt-4'>
					The rule string format we have used is <i>survival/birth notation.</i> In this notation, the original Game of Life would be
					expressed as <b>23/3</b> (we don’t worry about spacing between numbers because they can only be the numbers 0-8), where 2 or 3
					cells in the neighbourhood are required for survival of a living cell, and 3 cells in the neighbourhood are required for a dead
					cell to come to life/be born.
				</p>
			</section>

			{/* Larger than Life Section */}
			<section className='mb-10'>
				<div className='grid grid-cols-2 overflow-y-auto gap-4'>
					<div>
						<h2 className='text-2xl font-semibold text-purple-mdn mb-3'>Larger than Life</h2>

						{/* TODO: Uncomment when the page for this is made */}
						{/* <Link 
							href='/CAs/Larger/life.html' 
							className='inline-block bg-purple-mdn text-white px-6 py-2 rounded-md shadow-md hover:bg-purple-mdn-dark transition duration-300 mb-4'
						>
							Larger
						</Link> */}

						<p className='leading-7 text-lg mt-4'>
							Larger than Life builds on Life Like CA by introducing even more flexibility. This means the following are now specifiable in
							the rule string:
						</p>

						<ul className='list-disc list-inside space-y-2 text-lg mt-4'>
							<li>The neighbourhood radius to encompass cells that are further than one cell away.</li>
							<li>Neighbourhood shape.</li>
							<li>
								Minimum lifespan of living cells (basically living cells have HP that is deducted from every time it’s exposed to death
								conditions)
							</li>
						</ul>
					</div>

					<div className='grid grid-cols-1 overflow-y-auto gap-4'>
						<Image
							src='/images/LargerRadius2.png'
							alt='The neighbourhood in larger than life also contains all the 
							cells within a specified distance from the target cell'
							height={60}
							width={50}
							className='w-full rounded-md shadow'
						/>
						<p className='text-sm text-center italic'>
							Neighbourhood of a target cell in Larger than Life when the neighbourhood radius is set to 2 cells
						</p>

						<Image
							src='/images/LargerNeighbourhoodTypes.png'
							alt='The neighbourhood of each cell consists of the cells in contact with it.'
							height={60}
							width={50}
							className='w-full rounded-md shadow'
						/>
						<p className='text-sm text-center italic'>
							Different neighbourhood shapes/different ways of determining a cell's distance from the target cell
						</p>
					</div>
				</div>
			</section>

			{/* Continuous Cellular Automata Section */}
			<section className='mb-10'>
				<h2 className='text-2xl font-semibold text-purple-mdn mb-3'>Continuous Cellular Automata</h2>

				{/* TODO: Uncomment when the page for this is made */}
				{/* <Link
					href='/CAs/Continuous/life.html'
					className='inline-block bg-purple-mdn text-white px-6 py-2 rounded-md shadow-md hover:bg-purple-mdn-dark transition duration-300 mb-4'
				>
					Explore Continuous CA
				</Link> */}

				<p className='leading-7 text-lg mt-4'>
					Continuous CA also builds on Life Like CA. The main difference is that instead of using the binary dead or alive as states, we use
					a continuous range of values. The new cell state value is calculated in by multiplying each neighbour by a weight, adding this
					together and applying a basic mathematical function to it. Continuous CA can display behaviours similar to basic organisms and
					population level behaviours of bacteria etc. that have simple behavioural patterns.
				</p>
			</section>
		</div>
	);
}
