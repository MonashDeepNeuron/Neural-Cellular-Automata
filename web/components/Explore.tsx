import { Card } from "@/components/ui/card";
import { FlaskConical, BookOpen, Microscope } from "lucide-react";

const features = [
  {
    icon: BookOpen,
    title: "What is Neural Cellular Automata?",
    description: "An overview of NCA.",
    link: "/learn",
  },
  {
    icon: Microscope,
    title: "Our Research & Latest Findings",
    description: "Top 5 Papers in the field, summarised by us!",
    link: "/research",
  },
  {
    icon: FlaskConical,
    title: "Try the NCA Simulator",
    description: "Explore various dynamic models in real-time!",
    link: "/simulator",
  },
];

export const Explore = () => {
  return (
    <section className="relative py-24 overflow-hidden">
      {/* Background elements */}
      <div className="absolute inset-0 bg-grid-pattern bg-grid opacity-5" />
      
      <div className="container relative z-10 mx-auto px-6">
        <div className="text-center mb-16">
          <h2 className="text-3xl lg:text-5xl font-bold mb-4">
            <span className="gradient-text">Explore More</span>
          </h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Dive deep into the world of Neural Cellular Automata
          </p>
        </div>
        
        <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
          {features.map((feature, index) => (
            <Card 
              key={index}
              className="group relative p-8 bg-card/50 backdrop-blur-sm border-primary/20 hover:border-primary/40 transition-all duration-300 hover:shadow-lg hover:shadow-primary/20 hover:-translate-y-2 cursor-pointer"
            >
              {/* Glow effect on hover */}
              <div className="absolute inset-0 bg-gradient-to-br from-primary/5 to-secondary/5 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
              
              <div className="relative space-y-4">
                <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center group-hover:bg-primary/20 transition-colors duration-300">
                  <feature.icon className="w-6 h-6 text-primary" />
                </div>
                
                <h3 className="text-xl font-semibold text-foreground group-hover:text-primary transition-colors duration-300">
                  {feature.title}
                </h3>
                
                <p className="text-muted-foreground">
                  {feature.description}
                </p>
                
                <div className="flex items-center gap-2 text-primary font-medium group-hover:gap-3 transition-all duration-300">
                  <span className="text-sm">Learn more</span>
                  <span className="text-lg">→</span>
                </div>
              </div>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
};
