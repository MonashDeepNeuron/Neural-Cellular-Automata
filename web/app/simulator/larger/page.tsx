import createMetadata from '@/util/createMetadata';
import Client from './client';

export const metadata = createMetadata({
	title: 'Larger Than Life Simulator',
	description: 'Use the Larger than Life ruleset for cellular automata.'
});

export default function page() {
	return <Client />;
}
