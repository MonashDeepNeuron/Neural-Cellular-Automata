import Image from 'next/image';
import Link from 'next/link';

export default function Navbar() {
	return (
		<nav className='bg-purple overflow-hidden px-2 py-4 flex items-center justify-between'>
			<a href='https://www.deepneuron.org/'>
				<Image src='/images/mdn-logo.png' alt='Deep Neuron Logo' height={20} width={47} />
			</a>
			<Link className='no-underline text-white text-lg transition-all' href='/'>
				Home
			</Link>
			<Link className='no-underline text-white text-lg transition-all' href='/cellular-automata'>
				Cellular Automata
			</Link>
			<Link className='no-underline text-white text-lg transition-all' href='/simulator'>
				Simulator
			</Link>
			<Link className='no-underline text-white text-lg transition-all' href='/troubleshooting'>
				Troubleshooting
			</Link>
			<Link className='no-underline text-white text-lg transition-all' href='/contact'>
				Contact Us
			</Link>
		</nav>
	);
}
