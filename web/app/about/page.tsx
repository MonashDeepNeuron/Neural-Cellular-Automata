import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ExternalLink } from "lucide-react";
import { Contact } from "@/components/common/Contact";

const teamMembers = [
  {
    name: "Afraz Gul",
    role: "Project Lead",
    degree: "Bachelor of Science and Computer Science",
  },
  {
    name: "Chloe Koe",
    role: "Deep Learning & Graphics Engineer",
    degree: "Bachelor of Computer Science",
    image: "/images/profile/Chloe.jpg"
  },
  {
    name: "Nathan Culshaw",
    role: "Deep Learning Engineer",
    degree: "Bachelor of Computer Science Advanced (Honours)",
    image: "/images/profile/Nathan.jpg"
  },
  {
    name: "Angus Bosmans",
    role: "High Performance Computing Engineer",
    degree: "Bachelor of Mechatronics Engineering (AI) & Arts",
    image: "/images/profile/Angus.jpg"
  },
  {
    name: "Luca Lowndes",
    role: "Deep Learning Engineer",
    degree: "Bachelor of Computer Science and Engineering",
    image: "/images/profile/Luca.jpg"
  }
];

const advisors = [
  {
    name: "Keren Collins",
    role: "Deep Learning Advisor",
    degree: "Bachelor of Biomedical Engineering",
    image: "/images/profile/Keren.jpg"
  },
  {
    name: "Joshua Riantoputra",
    role: "Deep Learning Advisor and founder",
    degree: "Bachelor of Mathematics and Computational Science",
    image: "/images/profile/Josh.jpg"
  },
  {
    name: "Nyan Knaw",
    role: "Deep Learning Advisor",
    degree: "Bachelor of Engineering",
    image: "/images/profile/Nyan.jpg"
  },
  {
    name: "Alexander Mai",
    role: "Alumni",
    degree: "Bachelor of Computer Science",
    image: "/images/profile/Alex.jpg"
  }
];

const About = () => {
  return (
	<div>
		<section id="about" className="relative py-24 overflow-hidden">
		<div className="container relative z-10 mx-auto px-6">
			{/* Who Are We */}
			<div className="max-w-4xl mx-auto mb-16">
			<h2 className="text-4xl lg:text-5xl font-bold mb-8 text-center">
				About <span className="gradient-text">Us</span>
			</h2>
			
			<Card className="p-8 bg-card/50 backdrop-blur-sm border-primary/20">
				<h3 className="text-2xl font-bold mb-4">Who Are We?</h3>
				<p className="text-lg text-muted-foreground leading-relaxed mb-6">
				We are a project team under{" "}
				<a 
					href="https://www.deepneuron.org/" 
					target="_blank" 
					rel="noopener noreferrer"
					className="text-primary hover:underline font-semibold"
				>
					Monash DeepNeuron
				</a>
				, an Engineering/IT student team run by Monash University students. Started in November 2023, 
				NCA is one of many research projects, which you can read more about{" "}
				<a 
					href="https://www.deepneuron.org/" 
					target="_blank" 
					rel="noopener noreferrer"
					className="text-primary hover:underline"
				>
					here
				</a>
				!
				</p>

				<h3 className="text-2xl font-bold mb-4 mt-8">Project Objectives</h3>
				<ol className="list-decimal list-inside space-y-2 text-muted-foreground mb-6">
				<li>What are NCA? How is NCA different from other CA and Neural Networks?</li>
				<li>What can NCA be used for? Does NCA provide an advantage over other similar architectures?</li>
				<li>How can NCA be improved?</li>
				</ol>
				<p className="text-muted-foreground">
				As a result of answering these questions, we aim to produce a comprehensive research paper.
				</p>
			</Card>
			</div>

			{/* Team Members */}
			<div className="mb-20">
			<h3 className="text-3xl font-bold mb-4 text-center">Meet the Team!</h3>
			
			<div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8 max-w-6xl mx-auto">
				{teamMembers.map((member) => (
				<Card 
					key={member.name}
					className="p-6 bg-card/50 backdrop-blur-sm border-primary/20 hover:border-primary/40 transition-all duration-300"
				>
					<div className="aspect-square mb-4 overflow-hidden rounded-lg">
					<img 
						src={member.image} 
						alt={member.name}
						className="w-full h-full object-cover"
					/>
					</div>
					<h4 className="text-xl font-bold mb-1">{member.name}</h4>
					<p className="text-sm text-primary font-semibold mb-2">{member.role}</p>
					<p className="text-sm text-muted-foreground">{member.degree}</p>
				</Card>
				))}
			</div>
			</div>

			{/* Advisors & Alumni */}
			<div>
			<h3 className="text-3xl font-bold mb-4 text-center">Our Advisors and Alumni</h3>
			<p className="text-center text-muted-foreground mb-12">Their help has been just as invaluable to the project!</p>
			
			<div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8 max-w-6xl mx-auto">
				{advisors.map((advisor) => (
				<Card 
					key={advisor.name}
					className="p-6 bg-card/50 backdrop-blur-sm border-primary/20 hover:border-primary/40 transition-all duration-300"
				>
					<div className="aspect-square mb-4 overflow-hidden rounded-lg">
					<img 
						src={advisor.image} 
						alt={advisor.name}
						className="w-full h-full object-cover"
					/>
					</div>
					<h4 className="text-lg font-bold mb-1">{advisor.name}</h4>
					<p className="text-sm text-primary font-semibold mb-2">{advisor.role}</p>
					{advisor.degree && (
					<p className="text-sm text-muted-foreground">{advisor.degree}</p>
					)}
				</Card>
				))}
			</div>
			</div>
		</div>
		</section>
	</div>
  );
};

export default About;