import { Navigation } from "@/components/layout/Navigation";
import { Content } from "@/app/home/Content";
import { About } from "@/app/home/About";
import { Explore } from "@/app/home/Explore";
import { Contact } from "@/app/home/Contact";


export default function Home() {
	return (
		<div className="min-h-screen">
			<Navigation />
			<Content />
			<About />
			<Explore />
			<Contact />
    	</div>
	);
}
