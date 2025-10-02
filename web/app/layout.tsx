import { Poppins } from 'next/font/google';
import './globals.css';
import clsx from 'clsx';
import type { Viewport } from 'next';
import createMetadata from '@/util/createMetadata';
import { Navigation } from '@/components/layout/Navigation';

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
				<main className='min-h-screen p-4 pt-20 max-w-full'>
					{/* Added 'pt-20' for top padding */}
					{children}
				</main>
			</body>
		</html>
	);
}
