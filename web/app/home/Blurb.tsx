import { Card } from '@/components/ui/card';

export const Blurb = () => {
	return (
		<section className='relative py-24 overflow-hidden'>
			<div className='container relative z-10 mx-auto px-6'>
				<Card className='max-w-4xl mx-auto p-12 bg-card/50 backdrop-blur-sm border-primary/20'>
					<div className='space-y-6'>
						<h2 className='text-3xl lg:text-4xl font-bold'>
							Welcome to <span className='text-primary'>Neural Cellular Automata</span>
						</h2>

						<p className='text-lg text-muted-foreground leading-relaxed'>
							We are a research project team under <span className='text-foreground font-semibold'>Monash DeepNeuron</span>, exploring the
							potential of Neural Cellular Automata (NCA) for various applications. Our goal is to understand, simulate, and improve NCA
							models.
						</p>
					</div>
				</Card>
			</div>
		</section>
	);
};
