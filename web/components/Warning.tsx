'use client';
import { useState } from 'react';

export default function Warning() {
	const [show, setShow] = useState(true);
	if (!show) return null;

	return (
		<div className='bg-yellow-100 border-l-4 border-yellow-500 text-yellow-800 p-4 rounded-md shadow-md mb-6 transition-opacity duration-500'>
			<p className='mb-3'>
				<strong>⚠️ Warning:</strong> This website contains content that may <u>flash at high frequencies</u>. Please use discretion when
				selecting frame rates if sensitive to flashing visuals.
			</p>
			<button
				type='button'
				onClick={() => setShow(false)}
				className='bg-yellow-500 text-white px-4 py-2 rounded-md hover:bg-yellow-600 transition-transform duration-300 transform hover:scale-105'
			>
				Dismiss
			</button>
		</div>
	);
}
