import { Geist, Geist_Mono } from 'next/font/google';
import './globals.css';
import Providers from '@/components/Providers';
import Navbar from '@/components/layout/Nav';
import createMetadata from '@/util/createMetadata';
import clsx from 'clsx';

const geistSans = Geist({
	variable: '--font-geist-sans',
	subsets: ['latin']
});

const geistMono = Geist_Mono({
	variable: '--font-geist-mono',
	subsets: ['latin']
});

export const metadata = createMetadata();

export default function RootLayout({
	children
}: Readonly<{
	children: React.ReactNode;
}>) {
	return (
		<html lang='en'>
			<body className={clsx(geistSans.variable, geistMono.variable, 'antialiased text-black overflow-x-hidden max-w-screen')}>
				<Navbar />
				<main className='min-h-screen p-4 pt-20 max-w-full'>
					{/* Added 'pt-20' for top padding */}
					<Providers>{children}</Providers>
				</main>
			</body>
		</html>
	);
}
