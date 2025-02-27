'use client'; // 👈 This tells Next.js to treat this as a Client Component

import Link from 'next/link';
import { useState } from 'react';

export default function Simulator() {
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
			<div className='bg-gray-100 border-gray-500 p-6 rounded-md shadow-md mb-8'>
				<h3 className='text-2xl font-semibold text-gray-800 mb-4'>Select a Model:</h3>

				<div className='grid grid-cols-1 sm:grid-cols-2 gap-4'>
					{[
						{ name: 'Classic Conway', link: '/CAs/ConwaysLife/life.html' },
						{ name: 'Life Like', link: '/CAs/LifeLike/life.html' },
						{ name: 'Larger', link: '/CAs/Larger/life.html' },
						{ name: 'Continuous', link: '/CAs/Continuous/life.html' },
						{ name: 'G-NCA', link: '/CAs/GCA/life.html' }
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
			</div>

			{/* Navigation to Troubleshooting */}
			<div className='text-center mt-8'>
				<Link
					href='/troubleshooting'
					className='inline-block bg-purple-mdn text-white px-6 py-3 rounded-md shadow-md hover:bg-purple-mdn-dark transition-transform duration-300 transform hover:scale-105'
				>
					🛠️ Go to Troubleshooting
				</Link>
			</div>
		</div>
	);
}
