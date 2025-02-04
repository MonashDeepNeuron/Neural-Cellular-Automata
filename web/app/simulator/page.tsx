import Link from 'next/link';
import { useState } from 'react';

export default function Simulator() {
	const [showWarning, setShowWarning] = useState(true);

	return (
		<div className='container'>
			{/* Title */}
			<div className='text-column'>
				<h1>Simulator</h1>
			</div>

			{/* Warning Message */}
			{showWarning && (
				<div className='warning centre'>
					<p>
						<b>Warning:</b> This website contains content that may
						<u> flash at high frequencies</u>. If you or someone viewing this is sensitive to this type of content, please use discretion
						when choosing frame-rates.
					</p>
					<button type='button' onClick={() => setShowWarning(false)}>
						Dismiss
					</button>
				</div>
			)}

			{/* Model Selection Panel */}
			<div className='controlPanel'>
				<h3>Select a model:</h3>
				<div className='panelRow controlGroup'>
					<a href='/CAs/ConwaysLife/life.html' className='button' id='classic_conway'>
						Classic Conway
					</a>
					<a href='/CAs/LifeLike/life.html' className='button' id='life_like'>
						Life Like
					</a>
					<a href='/CAs/Larger/life.html' className='button' id='larger'>
						Larger
					</a>
				</div>
				<div className='panelRow controlGroup'>
					<a href='/CAs/Continuous/life.html' className='button' id='continuous'>
						Continuous
					</a>
					<a href='/CAs/GCA/life.html' className='button' id='growing_nca'>
						G-NCA
					</a>
				</div>
			</div>

			{/* Navigation to Troubleshooting */}
			<p style={{ textAlign: 'center', marginTop: '20px' }}>
				<Link href='/troubleshooting' className='font-bold'>
					See Next: Trouble Shooting
				</Link>
			</p>
		</div>
	);
}
