import { Blurb } from '@/app/home/Blurb';
import { Contact } from '@/app/home/Contact';
import { Content } from '@/app/home/Content';
import { Explore } from '@/app/home/Explore';

export default function Home() {
	return (
		<div>
			<Content />
			<Blurb />
			<Explore />
			<Contact />
		</div>
	);
}
