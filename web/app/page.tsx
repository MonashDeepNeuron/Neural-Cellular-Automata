import { Content } from "@/app/home/Content";
import { Blurb } from "@/app/home/Blurb";
import { Explore } from "@/app/home/Explore";
import { Contact } from "@/app/home/Contact";


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
