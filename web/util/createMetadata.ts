import type { Metadata } from 'next';

interface CreateMetadataParams {
	title?: string;
	description?: string;
}

export default function createMetadata(params?: CreateMetadataParams): Metadata {
	const title = params?.title ? `${params.title} | Neural Cellular Automata` : 'Neural Cellular Automata';
	const description = params?.description ?? 'Using Neural Networks to simulate life with cellular automata.';

	const metadata: Metadata = {
		title,
		description,
		openGraph: {
			type: 'website',
			title,
			description: description,
			url: 'https://neuralca.org',
			siteName: 'Neural Cellular Automata',
			locale: 'en',
			images: 'https://neuralca.org/cover.jpg'
		},
		appleWebApp: {
			title: 'NCA'
		},
		twitter: {
			card: 'summary_large_image'
		}
	};

	return metadata;
}
