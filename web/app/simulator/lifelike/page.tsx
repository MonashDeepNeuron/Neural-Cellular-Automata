import createMetadata from '@/util/createMetadata';
import Client from './client';

export const metadata = createMetadata({
	title: 'Life-Like Simulator',
	description: 'Use the Like Like cellular automata ruleset.'
});

export default function page() {
	return <Client />;
}
