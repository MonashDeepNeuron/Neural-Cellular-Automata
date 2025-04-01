import createMetadata from '@/util/createMetadata';
import Client from './client';

export const metadata = createMetadata({
	title: '3D Simulator',
	description: 'See a realtime 3D voxel render and lighting engine.'
});

export default function page() {
	return <Client />;
}
