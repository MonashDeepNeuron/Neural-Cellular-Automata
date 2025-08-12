'use client';
import { useEffect, useState } from 'react';

export function Warning() {
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

export function CompatibilityWarning() {
	const [show, setShow] = useState(false);

	useEffect(() => {
		const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
		console.log(navigator.platform);
		if (isIOS) {
			setShow(true);
		}
	}, []);

	if (!show) return null;

	return (
		<div className='bg-yellow-100 border-l-4 border-yellow-500 text-yellow-800 p-4 rounded-md shadow-md mb-6 transition-opacity duration-500'>
			<p className='mb-3'>
				<strong>⚠️ Notice:</strong> WebGPU is <u>not natively supported</u> on iOS devices. Follow our guide at{' '}
				<a href='/iosSucks' className='text-purple-mdn hover:underline'>
					/iosSucks
				</a>{' '}
				to set it up.
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
