import Link from 'next/link';

export default function About() {
	return (
		<div className='max-w-3xl mx-auto px-6 py-10 text-gray-800'>
			{/* Title */}
			<h1 className='text-4xl font-bold mb-6 text-center'>About Us</h1>

			{/* Who Are We Section */}
			<section className='mb-8'>
				<h2 className='text-2xl font-semibold text-purple-mdn mb-2'>Who Are We?</h2>
				<p className='leading-7 text-lg'>
					We are a project team under{' '}
					<a
						href='https://www.deepneuron.org/'
						target='_blank'
						rel='noopener noreferrer'
						className='text-purple-mdn font-semibold hover:underline hover:text-purple-mdn-dark transition'
					>
						Monash DeepNeuron
					</a>
					, an Engineering/IT student team run by Monash University students. 
					NCA is one of many research projects, which you can read more about{' '}
					<a
						href='https://www.deepneuron.org/'
						target='_blank'
						rel='noopener noreferrer'
						className='text-purple-mdn font-semibold hover:underline hover:text-purple-mdn-dark transition'
					>
						here
					</a>
					!
				</p>
			</section>

			{/* Project Objectives */}
			<section className='mb-8'>
				<h2 className='text-2xl font-semibold text-purple-mdn mb-2'>Project Objectives</h2>
				<ol className='list-decimal list-inside space-y-2 text-lg'>
					<li>What are NCA? How is NCA different from other CA and Neural Networks?</li>
					<li>What can NCA be used for? Does NCA provide an advantage over other similar architectures?</li>
					<li>How can NCA be improved?</li>
				</ol>
				<p className='mt-4 text-lg'>As a result of answering these questions, we aim to produce a comprehensive research paper.</p>
			</section>

			{/* Project Updates */}
			<section className='mb-8'>
				<h2 className='text-2xl font-semibold text-purple-mdn mb-2'>Project Updates</h2>
				<p className='text-lg'>
					Stay up-to-date with our latest progress by visiting the{' '}
					<Link href='/keeping-up' className='text-purple-mdn font-semibold hover:underline hover:text-purple-mdn-dark transition'>
						Project Updates
					</Link>{' '}
					page!
				</p>
			</section>

			{/* Join Us Section */}
			<section>
				<h2 className='text-2xl font-semibold text-purple-mdn mb-2'>Join Us!</h2>
				<p className='text-lg'>
					Are you a Monash Engineering or IT student interested in working on this project? Reach out to be informed when new
					positions open up. First-year or Master's students — all are welcome!
				</p>
				<a
					href='https://www.deepneuron.org/join-us'
					target='_blank'
					rel='noopener noreferrer'
					className='inline-block mt-4 px-6 py-2 bg-purple-mdn text-white font-semibold rounded-md shadow-md hover:bg-purple-mdn-dark transition duration-300'
				>
					Join the Team
				</a>
			</section>
		</div>
	);
}
