import WebGPUCanvas from '@/components/Canvas';
import { Link } from 'next/link'

export default function Home() {
	return (
		<>
			{/* Main Content */}
			<div className='centre'>
				<h1>Welcome to Neural Cellular Automata</h1>
				<p>
					We are a research project team under Monash DeepNeuron, exploring the potential of Neural Cellular Automata (NCA) for various
					applications. Our goal is to understand, simulate, and improve NCA models.
				</p>

				<h2>Explore More:</h2>
				<ul>
					<li>
						<Link to='/nca-intro'>What is Neural Cellular Automata?</Link>
					</li>
					<li>
						<Link to='/nca-research'>Our Research & Latest Findings</Link>
					</li>
					<li>
						<Link to='/simulator-home'>Try the NCA Simulator</Link>
					</li>
					<li>
						<Link to='/keeping-up'>Project Updates</Link>
					</li>
				</ul>

				<h2>Join Us!</h2>
				<p>
					Interested in working on this project? <br />
					<a href='https://www.deepneuron.org/contact-us' target='_blank' rel='noopener noreferrer'>
						Get in Touch!
					</a>
				</p>
			</div>
			<WebGPUCanvas />
		</>
	);
}
