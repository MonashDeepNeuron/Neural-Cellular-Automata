import { Poppins } from 'next/font/google';
import './globals.css';
import Providers from '@/components/Providers';
import Navbar from '@/components/layout/Nav';
import createMetadata from '@/util/createMetadata';
import clsx from 'clsx';
import type { Viewport } from 'next';

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
				<Navbar />
				<main className='min-h-screen p-4 pt-20 max-w-full'>
					{/* Added 'pt-20' for top padding */}
					<Providers>{children}</Providers>
				</main>
			</body>
		</html>
	);
}
