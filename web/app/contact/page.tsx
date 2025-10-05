import { ExternalLink, Mail, MessageSquare } from 'lucide-react';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import createMetadata from '@/util/createMetadata';

export const metadata = createMetadata({
	title: 'Contact Us',
	description:
		'We would love to hear from you! If you have any questions, feedback, or inquiries, please don’t hesitate to reach out to us.'
});
const Contact = () => {
	return (
		<div className='min-h-screen bg-background'>
			<main className='pt-24 pb-16'>
				<div className='container mx-auto px-6'>
					{/* Header */}
					<div className='max-w-3xl mx-auto mb-12 text-center'>
						<h1 className='text-4xl md:text-5xl font-bold mb-4 bg-clip-text text-primary'>Contact Us</h1>
						<p className='text-lg text-muted-foreground'>
							We would love to hear from you! If you have any questions, feedback, or inquiries, please don't hesitate to reach out.
						</p>
					</div>

					{/* Contact Card */}
					<div className='max-w-2xl mx-auto'>
						<Card className='border-2'>
							<CardHeader className='text-center'>
								<div className='mx-auto mb-4 w-16 h-16 rounded-full bg-primary flex items-center justify-center'>
									<Mail className='w-8 h-8 text-primary-foreground' />
								</div>
								<CardTitle className='text-2xl'>Get in Touch</CardTitle>
								<CardDescription className='text-base'>Visit the Deep Neuron contact page to send us a message</CardDescription>
							</CardHeader>
							<CardContent className='text-center space-y-6'>
								<div className='p-6 bg-muted/50 rounded-lg'>
									<MessageSquare className='w-12 h-12 text-primary mx-auto mb-3' />
									<p className='text-muted-foreground'>
										We typically respond within a few business days. Thank you for your interest in our project!
									</p>
								</div>

								<Button asChild size='lg' className='w-full sm:w-auto'>
									<Link
										href='https://www.deepneuron.org/contact-us'
										target='_blank'
										rel='noopener noreferrer'
										className='inline-flex items-center gap-2'
									>
										<span className='truncate'>Visit Deep Neuron Contact Page</span>
										<ExternalLink className='w-4 h-4' />
									</Link>
								</Button>
							</CardContent>
						</Card>
					</div>
				</div>
			</main>
		</div>
	);
};

export default Contact;
