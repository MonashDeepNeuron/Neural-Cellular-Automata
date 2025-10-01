import { Navigation } from "@/components/Navigation";
import { Content } from "@/components/Content";
import { About } from "@/components/About";
import { Explore } from "@/components/Explore";


export default function Home() {
	return (
		<div className="min-h-screen">
			<Navigation />
			<Content />
			<About />
			<Explore />
    	</div>
	);
}
