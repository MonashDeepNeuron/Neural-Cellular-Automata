import { Github, Linkedin } from "lucide-react";
import Link from "next/link";

export const Footer = () => {
  return (
    <footer className="relative border-t border-primary/10 bg-background/50 backdrop-blur-sm">
      <div className="container mx-auto px-6 py-12">
        <div className="grid md:grid-cols-4 gap-8">
          {/* Brand */}
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <span className="font-bold text-lg">Neural<span className="text-primary">CA</span></span>
            </div>
            <p className="text-sm text-muted-foreground">
              Exploring Neural Cellular Automata under Monash DeepNeuron
            </p>
          </div>

          {/* Quick Links */}
          <div>
            <h4 className="font-semibold mb-4">Quick Links</h4>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li><Link href="/" className="hover:text-primary transition-colors">Home</Link></li>
              <li><Link href="/about" className="hover:text-primary transition-colors">About</Link></li>
              <li><Link href="/simulator" className="hover:text-primary transition-colors">Simulator</Link></li>
              <li><Link href="/learn" className="hover:text-primary transition-colors">Learn</Link></li>
            </ul>
          </div>

          {/* Research */}
          <div>
            <h4 className="font-semibold mb-4">Research</h4>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li><Link href="/research" className="hover:text-primary transition-colors">Research</Link></li>
              <li><Link href="/contact" className="hover:text-primary transition-colors">Contact</Link></li>
            </ul>
          </div>

          {/* Social */}
          <div>
            <h4 className="font-semibold mb-4">Connect</h4>
            <div className="flex gap-4">
              <Link 
                href="https://github.com/MonashDeepNeuron/Neural-Cellular-Automata"
                target="_blank"
                className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center hover:bg-primary/20 transition-colors"
              >
                <Github className="w-5 h-5" />
              </Link>
              <Link
                href="https://au.linkedin.com/company/monashdeepneuron" 
                target="_blank" 
                className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center hover:bg-primary/20 transition-colors"
              >
                <Linkedin className="w-5 h-5" />
              </Link>
            </div>
          </div>
        </div>

        <div className="mt-12 pt-8 border-t border-primary/10 text-center text-sm text-muted-foreground">
          <p>© 2025 Neural Cellular Automata Project. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
};
