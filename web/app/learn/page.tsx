import { BookOpen, ExternalLink } from 'lucide-react';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import createMetadata from '@/util/createMetadata';

export const metadata = createMetadata({
	title: 'Learn',
	description: 'Learn about what neural cellular automata are - the evolution from cellular automata.'
});

export default function Learn() {
	return (
		<div className='min-h-screen'>
			<main className='pt-24 pb-16'>
				<div className='container mx-auto px-6'>
					{/* Header */}
					<div className='max-w-4xl mx-auto text-center mb-12'>
						<h1 className='text-4xl lg:text-5xl font-bold mb-6'>
							Understanding <span className='text-primary'>Neural Cellular Automata</span>
						</h1>
						<p className='text-xl text-muted-foreground mb-8'>
							Discover the fascinating intersection of cellular automata and neural networks
						</p>
					</div>

					{/* Introduction */}
					<div className='max-w-4xl mx-auto mb-16'>
						<Card className='p-8 bg-card/50 backdrop-blur-sm border-primary/20'>
							<h2 className='text-3xl font-bold mb-6'>Introduction</h2>
							<p className='text-lg text-muted-foreground leading-relaxed mb-6'>
								Neural Cellular Automata (NCA) represent an innovative intersection between traditional cellular automata and neural
								networks. They are capable of learning complex behaviors through simple, local interactions between cells, mimicking
								biological growth and regeneration processes.
							</p>
						</Card>
					</div>

					{/* Perspectives Section */}
					<div className='max-w-4xl mx-auto mb-16 space-y-8'>
						{/* Cellular Automata Perspective */}
						<Card className='p-8 bg-gradient-to-br from-primary/5 to-secondary/5 border-primary/20'>
							<h3 className='text-2xl font-bold mb-4'>The Cellular Automata Perspective</h3>
							<p className='text-lg text-muted-foreground leading-relaxed'>
								From the perspective of cellular automata, NCAs build on the concept of simple, rule-based systems where each cell updates
								its state based on the states of its neighbors. This approach enables the emergence of complex patterns from basic rules.
							</p>
						</Card>

						{/* Neural Network Perspective */}
						<Card className='p-8 bg-gradient-to-br from-secondary/5 to-primary/5 border-primary/20'>
							<h3 className='text-2xl font-bold mb-4'>The Neural Network Perspective</h3>
							<p className='text-lg text-muted-foreground leading-relaxed'>
								When viewed through the lens of neural networks, NCAs leverage the power of deep learning to optimize the update rules. This
								allows the system to adapt, learn, and generalize behaviors across various environments, making NCAs versatile tools for
								modeling dynamic systems.
							</p>
						</Card>
					</div>

					{/* Research Section */}
					<div className='max-w-4xl mx-auto text-center'>
						<Card className='p-8 bg-card/50 backdrop-blur-sm border-primary/20'>
							<h3 className='text-2xl font-bold mb-4'>Dive Deeper into Research</h3>
							<p className='text-muted-foreground mb-6 text-lg'>Ready to explore the research behind Neural Cellular Automata?</p>
							<Button asChild variant='default' size='lg'>
								<Link href='/learn/research' className='inline-flex items-center gap-2'>
									<BookOpen className='w-5 h-5' />
									Explore Research Papers
									<ExternalLink className='w-4 h-4' />
								</Link>
							</Button>
						</Card>
					</div>
				</div>
			</main>
		</div>
	);
}
