import { Navigation } from "@/components/Navigation";
import { Content } from "@/components/Content";
import { About } from "@/components/About";

export default function Home() {
	return (
		<div className="min-h-screen">
			<Navigation />
			<Content />
			<About />
    	</div>
	);
}
