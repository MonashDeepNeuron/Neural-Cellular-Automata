import { Poppins } from 'next/font/google';
import './globals.css';
import clsx from 'clsx';
import type { Viewport } from 'next';
import { Navigation } from '@/components/layout/Navigation';
import createMetadata from '@/util/createMetadata';
import { Footer } from '@/components/layout/Footer';

const poppins = Poppins({
	weight: ['500', '400'],
	subsets: ['latin']
});

export const metadata = createMetadata();

export const viewport: Viewport = {
	themeColor: '#7065f9'
};

export default function RootLayout({
	children
}: Readonly<{
	children: React.ReactNode;
}>) {
	return (
		<html lang='en'>
			<body className={clsx(poppins.className, 'antialiased text-black overflow-x-hidden max-w-screen')}>
				<Navigation />
				<main className='min-h-screen pt-4 p-4 max-w-full'>
					{children}
				</main>
				<Footer />
			</body>
		</html>
	);
}
