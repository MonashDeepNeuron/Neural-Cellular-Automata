import Card from '@/components/Card';
import createMetadata from '@/util/createMetadata';
import Link from 'next/link';

export const metadata = createMetadata({
	title: 'Contact Us',
	description:
		'We would love to hear from you! If you have any questions, feedback, or inquiries, please don’t hesitate to reach out to us.'
});

export default function Contact() {
	return (
		<div className='max-w-3xl mx-auto px-6 py-10 text-gray-800'>
			{/* Title */}
			<h1 className='text-4xl font-bold mb-4 text-center'>Contact Us</h1>

			{/* Intro Text */}
			<p className='text-lg leading-7 text-center mb-6'>
				We would love to hear from you! If you have any questions, feedback, or inquiries, please don’t hesitate to reach out to us.
			</p>

			{/* Contact Information */}
			<Card className='bg-gray-100 text-center'>
				<p className='text-xl font-semibold mb-2'>Get in Touch:</p>
				<Link
					href='https://www.deepneuron.org/contact-us'
					className='inline-block bg-purple-mdn text-white px-6 py-3 rounded-md shadow-md hover:bg-purple-mdn-dark transition-transform duration-300 transform hover:scale-105'
				>
					Visit the Deep Neuron Contact Page
				</Link>
			</Card>

			{/* Additional Note */}
			<p className='text-md mt-6 text-center'>
				We typically respond within a few business days. Thank you for your interest in our project!
			</p>
		</div>
	);
}
