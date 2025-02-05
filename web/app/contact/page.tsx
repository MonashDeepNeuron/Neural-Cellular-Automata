export default function Contact() {
	return (
		<div className="max-w-3xl mx-auto px-6 py-10 text-gray-800">
			{/* Title */}
			<h1 className="text-4xl font-bold text-purple-700 mb-4 text-center">Contact Us</h1>

			{/* Intro Text */}
			<p className="text-lg leading-7 text-center mb-6">
				We would love to hear from you! If you have any questions, feedback, or inquiries, please don’t hesitate to reach out to us.
			</p>

			{/* Contact Information */}
			<div className="bg-purple-50 p-6 rounded-md shadow-md text-center">
				<p className="text-xl font-semibold mb-2">Get in Touch:</p>
				<a
					href="https://www.deepneuron.org/contact-us"
					target="_blank"
					rel="noopener noreferrer"
					className="inline-block bg-purple-600 text-white px-6 py-3 rounded-md shadow-md hover:bg-purple-700 transition duration-300"
				>
					Visit the Deep Neuron Contact Page
				</a>
			</div>

			{/* Additional Note */}
			<p className="text-md mt-6 text-center">
				We typically respond within a few business days. Thank you for your interest in our project!
			</p>
		</div>
	);
}
