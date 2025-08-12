'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useState } from 'react';

export default function Navbar() {
	const [isOpen, setIsOpen] = useState(false);
	const pathname = usePathname();

	const navLinks = [
		{ name: 'Home', path: '/' },
		{ name: 'About', path: '/about' },
		{ name: 'Simulator', path: '/simulator' },
		{ name: 'Learn', path: '/learn' },
		{ name: 'Troubleshooting', path: '/troubleshooting' },
		{ name: 'Contact Us', path: '/contact' }
	];

	return (
		<nav className='bg-purple-mdn shadow-md fixed top-0 left-0 right-0 z-50 h-16'>
			{/* Purple Background */}
			<div className='max-w-full mx-auto px-4 py-3 flex items-center justify-between'>
				{/* Logo */}
				<div className='flex items-center space-x-4'>
					<a href='https://www.deepneuron.org/' target='_blank' rel='noopener noreferrer'>
						<img src='/images/mdn-logo.png' alt='Deep Neuron Logo' height={40} width={80} />
					</a>

					{/* Navigation Links */}
					<div className='hidden md:flex items-center space-x-6'>
						{navLinks.map(link => (
							<Link
								key={link.path}
								href={link.path}
								className={`px-3 py-2 rounded-md text-lg font-medium text-white transition-transform duration-200 ${
									pathname === link.path
										? 'font-semibold' // Active link is bold
										: 'hover:scale-105 hover:text-purple-mdn-light' // Hover effect
								}`}
							>
								{link.name}
							</Link>
						))}
					</div>
				</div>

				{/* Hamburger Icon for Mobile */}
				<button className='md:hidden text-white focus:outline-hidden' onClick={() => setIsOpen(!isOpen)} type='button'>
					☰
				</button>
			</div>
			{/* Mobile Menu */}
			{isOpen && (
				<div className='md:hidden bg-purple-mdn shadow-md'>
					{' '}
					{/* Purple Background for Mobile Menu */}
					{navLinks.map(link => (
						<Link
							key={link.path}
							href={link.path}
							className='block px-4 py-2 text-white hover:text-gray-200 transition-transform duration-200 hover:scale-105'
							onClick={() => setIsOpen(false)}
						>
							{link.name}
						</Link>
					))}
				</div>
			)}
		</nav>
	);
}
