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
		},
    appleWebApp: {
      title: 'NCA'
    }
	};

	return metadata;
}
