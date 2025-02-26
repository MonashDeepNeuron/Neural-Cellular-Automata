export default function CellularAutomata() {
	return (
		<div className='max-w-4xl mx-auto px-6 py-10 text-gray-800'>
			{/* Title */}
			<h1 className='text-4xl font-bold text-purple-700 mb-6 text-center'>Cellular Automata Models</h1>

			{/* Neural Cellular Automata Section */}
			<section className='mb-10'>
				<h2 className='text-2xl font-semibold text-purple-600 mb-3'>Neural Cellular Automata (NCA)</h2>
				<a
					href='/CAs/GCA/life.html'
					className='inline-block bg-purple-600 text-white px-6 py-2 rounded-md shadow-md hover:bg-purple-700 transition duration-300 mb-4'
				>
					Explore G-NCA
				</a>
				<p className='leading-7 text-lg mt-4'>
					Neural Cellular Automata (NCA) are a category of cellular automata that involve using a neural network as the cell’s update rule.
					The neural network can be trained to determine how to update the cell’s value in coordination with other cells, operating on the
					same rule to produce a target behavior.
				</p>
				<p className='leading-7 text-lg mt-4'>
					One of the best examples of NCA is{' '}
					<a
						href='https://distill.pub/2020/growing-ca/'
						className='text-purple-500 font-semibold hover:underline'
						target='_blank'
						rel='noopener noreferrer'
					>
						Growing Neural Cellular Automata
					</a>{' '}
					(A. Mordvintsev et al., 2020), where they trained NCA to ‘grow’ target images from a single seed cell.
				</p>
				<p className='leading-7 text-lg mt-4'>
					From a deep learning perspective, NCA can be characterized as a Recurrent Convolutional Neural Network. Learn more about this{' '}
					<a href='/Pages/nca-ca.html' className='text-purple-500 font-semibold hover:underline'>
						here
					</a>
					.
				</p>
				<p className='leading-7 text-lg mt-4'>
					NCA models display properties similar to how cells communicate within living organisms. Interestingly, the model from Growing
					Neural Cellular Automata also showed natural regenerative properties.
				</p>
			</section>

			{/* Game of Life Section */}
			<section className='mb-10'>
				<h2 className='text-2xl font-semibold text-purple-600 mb-3'>John Conway’s Game of Life</h2>
				<a
					href='/CAs/ConwaysLife/life.html'
					className='inline-block bg-purple-600 text-white px-6 py-2 rounded-md shadow-md hover:bg-purple-700 transition duration-300 mb-4'
				>
					Play Classic Conway
				</a>
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
					<a
						href='https://playgameoflife.com/'
						className='text-purple-500 font-semibold hover:underline'
						target='_blank'
						rel='noopener noreferrer'
					>
						here
					</a>
					.
				</p>
			</section>

			{/* Continuous Cellular Automata Section */}
			<section>
				<h2 className='text-2xl font-semibold text-purple-600 mb-3'>Continuous Cellular Automata</h2>
				<a
					href='/CAs/Continuous/life.html'
					className='inline-block bg-purple-600 text-white px-6 py-2 rounded-md shadow-md hover:bg-purple-700 transition duration-300 mb-4'
				>
					Explore Continuous CA
				</a>
				<p className='leading-7 text-lg mt-4'>
					Continuous Cellular Automata builds on life-like CA. The key difference is that instead of using binary states (dead or alive), we
					use a continuous range of values, allowing for more nuanced behaviors.
				</p>
			</section>
		</div>
	);
}
