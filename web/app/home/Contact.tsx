import { Mail, Users } from 'lucide-react';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';

export const Contact = () => {
	return (
		<section className='relative py-24 overflow-hidden'>
			<div className='container relative z-10 mx-auto px-6'>
				<div className='grid md:grid-cols-2 gap-8 max-w-6xl mx-auto'>
					{/* Join Us Card */}
					<Card className='relative p-10 bg-card/50 backdrop-blur-sm border-primary/20 overflow-hidden group hover:border-primary/40 transition-all duration-300 flex flex-col'>
						<div className='absolute inset-0 bg-gradient-to-br from-primary/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300' />

						<div className='relative space-y-6 flex-1 flex flex-col'>
							<div className='w-14 h-14 rounded-xl bg-primary/10 flex items-center justify-center'>
								<Users className='w-7 h-7 text-primary' />
							</div>

							<div className='flex-1'>
								<h3 className='text-2xl font-bold mb-3'>Join Us!</h3>
								<p className='text-muted-foreground'>
									Are you a Monash Engineering or IT student interested in working on this project? Reach out to be informed when new
									positions open up. First-year or Master's students — all are welcome!
								</p>
							</div>

							<Link
								href='https://docs.google.com/forms/d/e/1FAIpQLSckOGpNS-nFOxB4cGHmXC2z04D6_m8j26qKLZee3bZ298vNWg/viewform?usp=sharing'
								target='_blank'
								rel='noopener noreferrer'
								className='mt-auto'
							>
								<Button variant='glow' size='lg' className='w-full cursor-pointer'>
									Join the Team
								</Button>
							</Link>
						</div>
					</Card>

					{/* Contact Us Card */}
					<Card className='relative p-10 bg-card/50 backdrop-blur-sm border-primary/20 overflow-hidden group hover:border-primary/40 transition-all duration-300 flex flex-col'>
						<div className='absolute inset-0 bg-gradient-to-br from-primary/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300' />

						<div className='relative space-y-6 flex-1 flex flex-col'>
							<div className='w-14 h-14 rounded-xl bg-primary/10 flex items-center justify-center'>
								<Mail className='w-7 h-7 text-secondary' />
							</div>

							<div className='flex-1'>
								<h3 className='text-2xl font-bold mb-3'>Contact Us!</h3>
								<p className='text-muted-foreground'>
									Want to learn more about projects like this, or interested in a collaboration? We'd love to hear from you.
								</p>
							</div>

							<Link href='https://deepneuron.org/contact-us' target='_blank' rel='noopener noreferrer' className='mt-auto'>
								<Button variant='glow' size='lg' className='w-full cursor-pointer'>
									Contact Us
								</Button>
							</Link>
						</div>
					</Card>
				</div>
			</div>
		</section>
	);
};
