import Link from 'next/link';
import ProfileCard from '@/components/Profile';
import createMetadata from '@/util/createMetadata';

export const metadata = createMetadata({
	title: 'About',
	description: 'We are a project team under Monash DeepNeuron, an Engineering/IT student team run by Monash University students.'
});

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
					<Link
						href='https://www.deepneuron.org/'
						className='text-purple-mdn font-semibold hover:underline hover:text-purple-mdn-dark transition'
					>
						Monash DeepNeuron
					</Link>
					, an Engineering/IT student team run by Monash University students. Started in November 2023, NCA is one of many research
					projects, which you can read more about{' '}
					<Link
						href='https://www.deepneuron.org/'
						className='text-purple-mdn font-semibold hover:underline hover:text-purple-mdn-dark transition'
					>
						here
					</Link>
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

			{/* Join Us Section */}
			<section className='mb-8'>
				<h2 className='text-2xl font-semibold text-purple-mdn mb-2'>Join Us!</h2>
				<p className='text-lg'>
					Are you a Monash Engineering or IT student interested in working on this project? Reach out to be informed when new positions open
					up. First-year or Master's students — all are welcome!
				</p>
				<Link
					href='https://docs.google.com/forms/d/e/1FAIpQLSckOGpNS-nFOxB4cGHmXC2z04D6_m8j26qKLZee3bZ298vNWg/viewform?usp=sharing'
					className='inline-block bg-purple-mdn text-white px-6 py-3 rounded-md shadow-md hover:bg-purple-mdn-dark transition-transform duration-300 transform hover:scale-105'
				>
					Join the Team
				</Link>
			</section>

			{/* Meet the Team */}
			<section className='mb-8'>
				<h2 className='text-2xl font-semibold text-purple-mdn mb-2'>Meet the Team!</h2>
				<p>Super proud of our amazing team!</p>
				<div className='grid grid-cols-3 gap-4 mb-8'>
					<ProfileCard
						name='Afraz Gul'
						imageLink='/images/profile/Afraz.jpg'
						subtitle='Project Lead'
						description='Bachelor of Science and Computer Science'
					/>
					<ProfileCard
						name='Chloe Koe'
						imageLink='/images/profile/Chloe.jpg'
						subtitle='Deep Learning & Graphics Engineer'
						description='Bachelor of Computer Science'
					/>
					<ProfileCard
						name='Nathan Culshaw'
						imageLink='/images/profile/Nathan.jpg'
						subtitle='Deep Learning Engineer'
						description='Bachelor of Computer Science Advanced (Honours)'
					/>
					<ProfileCard
						name='Angus Bosmans'
						imageLink='/images/profile/Angus.jpg'
						subtitle='High Performance Computing Engineer'
						description='Bachelor of Mechatronics Engineering (AI) & Arts'
					/>
					<ProfileCard
						name='Luca Lowndes'
						imageLink='/images/profile/Luca.jpg'
						subtitle='Deep Learning Engineer'
						description='Bachelor of Computer Science and Engineering '
					/>
				</div>
				<h2 className='text-2xl font-semibold text-purple-mdn mb-2'>Our advisors and alumni</h2>
				<p>Their help has been just as invaluable to the project!</p>
				<div className='grid grid-cols-3 gap-4'>
					<ProfileCard
						name='Keren Collins'
						imageLink='/images/profile/Keren.jpg'
						subtitle='Deep learning advisor'
						description='Bachelor of Biomedical Engineering'
					/>
					<ProfileCard
						name='Joshua Riantoputra'
						imageLink='/images/profile/Josh.jpg'
						subtitle='Deep Learning Advisor and founder'
						description='Bachelor of Mathematics and Computational Science'
					/>
					<ProfileCard name='Nyan Knaw' imageLink='/images/profile/Nyan.jpg' subtitle='Deep Learning Advisor' description='' />
					<ProfileCard name='Alex Mai' imageLink='/images/profile/Alex.jpg' subtitle='Alumni' description='Bachelor of Computer Science' />
				</div>
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
		</div>
	);
}
