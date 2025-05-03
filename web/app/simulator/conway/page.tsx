import createMetadata from '@/util/createMetadata';
import Client from './client';

export const metadata = createMetadata({
	title: "Conway's Game of Life",
	description: "Use the Conway's game of life for cellular automata."
});

export default function page() {
	return <Client />;
}
