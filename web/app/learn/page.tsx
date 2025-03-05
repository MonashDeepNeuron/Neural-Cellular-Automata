import createMetadata from '@/util/createMetadata';
import Link from 'next/link';

export const metadata = createMetadata({
	title: 'Learn',
	description: 'Learn about what neural cellular automata are - the evolution from cellular automata.'
});

export default function Intro() {
	return (
		<div className='max-w-4xl mx-auto px-6 py-10 text-gray-800'>
			{/* Main Title */}
			<h1 className='text-4xl font-bold text-gray-800 mb-6 text-center'>Neural Cellular Automata</h1>
			{/* TODO: Re-write or update. This article content AI generated */}
			{/* Introduction Section */}
			<section className='mb-8'>
				<h2 className='text-2xl font-semibold text-purple-mdn mb-3'>Introduction</h2>
				<p className='text-lg leading-7'>
					Neural Cellular Automata (NCA) represent an innovative intersection between traditional cellular automata and neural networks.
					They are capable of learning complex behaviors through simple, local interactions between cells, mimicking biological growth and
					regeneration processes.
				</p>
			</section>

			{/* Cellular Automata Perspective */}
			<section className='mb-8'>
				<h2 className='text-2xl font-semibold text-purple-mdn mb-3'>The Cellular Automata Perspective</h2>
				<p className='text-lg leading-7'>
					From the perspective of cellular automata, NCAs build on the concept of simple, rule-based systems where each cell updates its
					state based on the states of its neighbors. This approach enables the emergence of complex patterns from basic rules.
				</p>
			</section>

			{/* Neural Network Perspective */}
			<section>
				<h2 className='text-2xl font-semibold text-purple-mdn mb-3'>The Neural Network Perspective</h2>
				<p className='text-lg leading-7'>
					When viewed through the lens of neural networks, NCAs leverage the power of deep learning to optimize the update rules. This
					allows the system to adapt, learn, and generalize behaviors across various environments, making NCAs versatile tools for modeling
					dynamic systems.
				</p>
			</section>

			{/* Research */}
			<section>
				<h2 className='text-2xl font-semibold text-purple-mdn mb-3'>The Research</h2>
				<p className='text-lg leading-7'>
					... Read about our favourite research articles{' '}
					<Link href='/learn/research' className='text-purple-mdn font-semibold hover:underline hover:text-purple-mdn-dark transition'>
						here
					</Link>
				</p>
			</section>
		</div>
	);
}
