import Card from '@/components/Card';

export default function KeepingUp() {
	return (
		<div className='max-w-3xl mx-auto px-6 py-10 text-gray-800'>
			{/* INACCURATE, CONTENT RE-WRITE*/}
			{/* Page Title */}
			<h1 className='text-4xl font-bold text-purple-mdn mb-8 text-center'>Project Updates - 2024</h1>

			{/* EOY Update */}
			<section className='mb-8'>
				<h2 className='text-2xl font-semibold text-purple-mdn mb-2'>End of Year (EOY) Update - December 2024</h2>
				<p className='text-lg leading-7'>...</p>
			</section>

			{/* October Update */}
			<section className='mb-8'>
				<h2 className='text-2xl font-semibold text-purple-mdn mb-2'>October Update 2024</h2>
				<p className='text-lg leading-7'>...</p>
			</section>

			{/* July Update */}
			<section className='mb-8'>
				<h2 className='text-2xl font-semibold text-purple-mdn mb-2'>July Update 2024</h2>
				<p className='text-lg leading-7'>...</p>
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
