import createMetadata from '@/util/createMetadata';
import Client from './client';

export const metadata = createMetadata({
	title: 'Texture Simulator',
	description: 'Simulate growing textures using location independent cellular automata.'
});

export default function page() {
	return <Client />;
}
