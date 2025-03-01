import createMetadata from '@/util/createMetadata';
import Client from './client';

export const metadata = createMetadata({
	title: 'Continuous Simulator',
	description: "Simulate the natural evolution of Conway's game of life into the continuous domain."
});

export default function page() {
	return <Client />;
}
