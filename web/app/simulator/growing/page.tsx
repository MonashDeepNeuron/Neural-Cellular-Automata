import createMetadata from '@/util/createMetadata';
import Client from './client';

export const metadata = createMetadata({
	title: 'Growing Simulator',
	description: 'Simulate growing organisms using location independent cellular automata.'
});

export default function page() {
	return <Client />;
}
