import Card from '@/components/Card';
import createMetadata from '@/util/createMetadata';

export const metadata = createMetadata({
	title: 'Keeping Up',
	description: 'Follow updates and see how this project has evolved over time.'
});

export default function KeepingUp() {
	return (
		<div className='max-w-4xl mx-auto px-6 py-10 text-gray-800'>
			{/* TODO: Current content is accurate, however needs to be structured and written nicely */}
			{/* Page Title */}
			<h1 className='text-4xl font-bold mb-8 text-center'>Project Updates - 2024</h1>

			{/* February Update */}
			<section className='mb-8'>
				<h2 className='text-2xl font-semibold text-purple-mdn mb-2'>February Update 2025</h2>

				<ul className='list-disc list-inside space-y-2 text-lg mt-4'>
					<li>Implemented and trained later stages of the Growing NCA model, including our version of the persisting model.</li>
					<li>
						Implemented and trained image recognition models in the methodology specified by Image Segmentation NCA and then Med-NCA.{' '}
					</li>
					<li>Implemented and trained Self-Organising Textures. </li>
					<li>Continued background research for competitive methods for proposed usages of NCA.</li>
					<li>Re-vamped the website to use Next.js, making the website cleaner more maintainable.</li>
					<li>Added completed models for Textures to the website and added final randomising function for Growing NCA. </li>
					<li>Added educational pages to the website.</li>
				</ul>
			</section>

			{/* October Update */}
			<section className='mb-8'>
				<h2 className='text-2xl font-semibold text-purple-mdn mb-2'>October Update 2024</h2>
				<ul className='list-disc list-inside space-y-2 text-lg mt-4'>
					<li>Re-wrote and fixed our growing cat model, including training and model code.</li>
					<li>Almost finished translation of growing cat model into WebGPU, except components requiring randomisation.</li>
					<li>Did substantial background research onto current state of the art in Neural Cellular Automata & related</li>
					<li>Finalised summer team and delivered training sessions </li>
					<li>Created automatic learning rate adjustment for training NCA</li>
				</ul>
			</section>

			{/* July Update */}
			<section className='mb-8'>
				<h2 className='text-2xl font-semibold text-purple-mdn mb-2'>July Update 2024</h2>
				<p className='text-lg leading-7'>
					Website Launch! The first iteration of the website is hosted Featuring: Conway's life, life-like, larger than life and continuous
					ca Literature searching conducted for NCA, getting an understanding of the major papers regarding NCA
				</p>
			</section>

			{/* Website Launch */}
			<Card className='bg-gray-100 text-center'>
				<h2 className='text-2xl font-semibold text-purple-mdn mb-2'>🎉 Website Launched!</h2>
				<p className='text-lg leading-7'>
					Our official website is live, making it easier to explore our work, access resources, and follow project updates. Thank you for
					being part of this journey!
				</p>
			</Card>
		</div>
	);
}
