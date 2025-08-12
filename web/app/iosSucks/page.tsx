import Image from 'next/image';
import Link from 'next/link';
import createMetadata from '@/util/createMetadata';

export const metadata = createMetadata({
	title: 'WebGPU Setup',
	description: 'A tutorial for setting up webGPU on ios'
});

export default function WebGPUSetup() {
	return (
		<>
			<h1 className='text-3xl font-bold mb-6'>Setting Up WebGPU on iOS</h1>

			<p className='mb-4'>
				WebGPU is not yet natively supported on iOS devices like iPhone and iPad. However, you can enable experimental support by following
				these steps:
			</p>

			<h2 className='text-2xl font-semibold mt-8 mb-4'>How to Enable WebGPU</h2>
			<ol className='list-decimal list-inside space-y-8 mb-10'>
				<li>
					Open the <strong>Settings</strong> app on your device.
				</li>
				<li>
					Go to <strong>Safari</strong>.
					<div className='mt-2'>
						<Image
							src='/webGPUSetup/Step1.jpg'
							alt='Navigate to Experimental Features'
							width={150}
							height={100}
							className='rounded-md shadow-sm'
						/>
					</div>
				</li>
				<li>
					Go to <strong>Advanced</strong>.
					<div className='mt-2'>
						<Image
							src='/webGPUSetup/Step2.jpg'
							alt='Navigate to Experimental Features'
							width={150}
							height={100}
							className='rounded-md shadow-sm'
						/>
					</div>
				</li>
				<li>
					Go to <strong>Experimental Features</strong>.
					<div className='mt-2'>
						<Image
							src='/webGPUSetup/Step3.jpg'
							alt='Navigate to Experimental Features'
							width={150}
							height={100}
							className='rounded-md shadow-sm'
						/>
					</div>
				</li>
				<li>
					Find <strong>WebGPU</strong> in the list and <strong>turn it on</strong>.
					<div className='mt-2'>
						<Image src='/webGPUSetup/Step4.jpg' alt='Enable WebGPU' width={150} height={100} className='rounded-md shadow-sm' />
					</div>
				</li>
				<li>
					Completely <strong>close and reopen Safari</strong> (swipe it away in the app switcher).
				</li>
				<li>
					Return to this website and <strong>reload the page</strong>.
				</li>
			</ol>

			<div className='bg-yellow-100 border-l-4 border-yellow-500 text-yellow-800 p-4 rounded-md shadow-sm mb-10'>
				<p className='font-semibold'>⚠️ Important:</p>
				<p>WebGPU support on iOS is experimental. Some features may be unstable or perform differently compared to desktop browsers.</p>
			</div>

			<h2 className='text-2xl font-semibold mb-4'>Troubleshooting</h2>
			<ul className='list-disc list-inside space-y-2 mb-6'>
				<li>
					Make sure your device is running <strong>iOS 17 or later</strong>.
				</li>
				<li>
					Use the <strong>Safari browser</strong> (This is only tested for Safari).
				</li>
			</ul>

			<p>If you're still having issues, please contact us or check back for future updates!</p>
			{/* Contact Us Link */}
			<div className='text-left mt-2 mb-2'>
				<Link href='/contact' className='text-purple-mdn font-bold hover:underline hover:text-purple-mdn transition duration-300'>
					📩 Contact Us
				</Link>
			</div>
		</>
	);
}
