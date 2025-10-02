'use client';

import { Menu, X } from 'lucide-react';
import Image from 'next/image';
import Link from 'next/link';
import { useState } from 'react';
import logoImage from '@/public/images/mdn-logo.png';

const navLinks = [
	{ label: 'Home', href: '/' },
	{ label: 'About', href: '/about' },
	{ label: 'Simulator', href: '/simulator' },
	{ label: 'Learn', href: '/learn' },
	{ label: 'Troubleshooting', href: '/troubleshooting' },
	{ label: 'Contact', href: '/contact' }
];

export const Navigation = () => {
	const [isOpen, setIsOpen] = useState(false);

	return (
		<nav className='fixed top-0 left-0 right-0 z-50 border-b border-primary/10 bg-primary/95 backdrop-blur-lg'>
			<div className='container mx-auto px-6'>
				<div className='flex items-center justify-between h-16'>
					{/* Logo */}
					<div className='flex items-center gap-2'>
						<Link href='https://www.deepneuron.org/' target='_blank' rel='noopener noreferrer'>
							<Image src={logoImage} alt='Monash DeepNeuron Logo' className='h-16 w-16 object-contain' />
						</Link>
					</div>

					{/* Desktop Navigation */}
					<div className='hidden md:flex items-center gap-8'>
						{navLinks.map(link => (
							<Link key={link.href} href={link.href} className='text-sm text-white/80 hover:text-white transition-colors duration-200'>
								{link.label}
							</Link>
						))}
					</div>

					{/* Mobile Menu Button */}
					<button onClick={() => setIsOpen(!isOpen)} className='md:hidden p-2 rounded-lg hover:bg-white/10 transition-colors'>
						{isOpen ? <X className='w-5 h-5' /> : <Menu className='w-5 h-5' />}
					</button>
				</div>

				{/* Mobile Navigation */}
				{isOpen && (
					<div className='md:hidden py-4 border-t border-primary/10'>
						<div className='flex flex-col gap-4'>
							{navLinks.map(link => (
								<Link
									key={link.href}
									href={link.href}
									className='text-sm text-white/80 hover:text-white transition-colors duration-200 py-2'
									onClick={() => setIsOpen(false)}
								>
									{link.label}
								</Link>
							))}
						</div>
					</div>
				)}
			</div>
		</nav>
	);
};
