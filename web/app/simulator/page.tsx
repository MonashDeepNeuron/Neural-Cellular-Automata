import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { AlertTriangle, ExternalLink } from "lucide-react";

const simulatorModels = [
  {
    title: "Classic Conway's Game of Life",
    link: "/simulator/conway",
    description: "The most famous example of cellular automata operating on simple rules: All cells are either alive or dead (1 or 0).",
    details: "A living cell with 2 or 3 neighbors survives. A dead cell with exactly 3 neighbors becomes alive. In all other cases, the cell dies or remains dead. Even with such simple rules, complex behaviors can emerge.",
    demoVideo: "/model_demos/conway.webm"
  },
  {
    title: "Life-Like Cellular Automata",
    link: "/simulator/lifelike",
    description: "Operates very similarly to the Game of Life in that all cells are either alive or dead. However, Life Like gives you the freedom to choose how many cells must be alive.",
    details: "The rule string format uses survival/birth notation. In this notation, the original Game of Life would be expressed as 23/3, where 2 or 3 cells in the neighbourhood are required for survival of a living cell, and 3 cells are required for a dead cell to come to life.",
    demoVideo: "/model_demos/lifelike.webm"
  },
    {
    title: "Larger than Life",
    link: "/simulator/larger",
    description: "Builds on Life Like CA by introducing even more flexibility with specifiable neighbourhood radius, shapes, and minimum lifespan of living cells.",
    details: "The neighbourhood radius can encompass cells that are further than one cell away. Different neighbourhood shapes allow for different ways of determining a cell's distance from the target cell.",
    demoVideo: "/model_demos/larger.webm"
  },
  {
    title: "Continuous Cellular Automata",
    link: "/simulator/continuous",
    description: "Instead of using binary dead or alive states, Continuous CA uses a continuous range of values.",
    details: "The new cell state value is calculated by multiplying each neighbour by a weight, adding this together and applying a basic mathematical function to it. Can display behaviours similar to basic organisms and population level behaviours of bacteria.",
    demoVideo: "/model_demos/continuous.webm"
  },
  {
    title: "Growing NCA",
    link: "/simulator/growing",
    description: "One of the best examples of NCA is Growing Neural Cellular Automata (A. Mordvintsev et al., 2020), where they trained NCA to 'grow' target images from a single seed cell.",
    details: "The Growing-NCA model emphasises that the perception of only neighbouring cells bears parallels with how natural cells communicate within living organisms. Interestingly, this results in Growing Neural Cellular Automata (and other NCA) also showing natural regenerative properties when the image is disturbed during generation.",
    demoVideo: "/model_demos/growing.webm"
  },
  {
    title: "Texture NCA",
    link: "/simulator/texture",
    description: "Based on Self Organising Textures (A. Mordvintsev et al., 2021), where they trained NCA to 'grow' target images from a single seed cell.",
    details: "The main difference between Texture NCA and Growing NCA is that it aims to replicate image features on a small scale. It shares properties of regeneration, as well as independence of grid location, resulting in textures that can be smoothly and cohesively replicated over grids of any size and shape.",
    demoVideo: "/model_demos/texture.webm"
  },

];

const Simulator = () => {
  return (
    <div className="min-h-screen">
      <main className="pt-24 pb-16">
        <div className="container mx-auto px-6">
          {/* Header */}
          <div className="max-w-4xl mx-auto text-center mb-12">
            <h1 className="text-4xl lg:text-5xl font-bold mb-6">
              Neural Cellular Automata <span className="gradient-text">Simulator</span>
            </h1>
            <p className="text-xl text-muted-foreground mb-8">
              Explore dynamic models in real-time!
            </p>
            
            {/* Warning Alert */}
            <Alert className="mb-8 border-destructive/50 bg-destructive/5 text-left">
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription className="text-left">
                <strong>Warning:</strong> This website contains content that may flash at high frequencies. 
                Please use discretion when selecting frame rates if sensitive to flashing visuals.
              </AlertDescription>
            </Alert>
          </div>

          {/* Introduction */}
          <div className="max-w-4xl mx-auto mb-16">
            <Card className="p-8 bg-card/50 backdrop-blur-sm border-primary/20">
              <h2 className="text-3xl font-bold mb-4">What are Neural Cellular Automata?</h2>
              <p className="text-lg text-muted-foreground leading-relaxed mb-4">
                Neural Cellular Automata (NCA) are a category of cellular automata that involve using a neural 
                network as the cell's update rule. The neural network can be trained to determine how to update 
                the cell's value in coordination with other cells, operating on the same rule to produce a target behavior.
              </p>
              <p className="text-muted-foreground">
                From a deep learning perspective, NCA can be characterized as a Recurrent Convolutional Neural Network.
              </p>
            </Card>
          </div>

          {/* Simulator Models Grid */}
          <div className="mb-12">
            <h2 className="text-3xl font-bold mb-8 text-center">Select a Model</h2>
            
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
              {simulatorModels.map((model) => (
                <Card 
                  key={model.title}
                  className="overflow-hidden bg-card/50 backdrop-blur-sm border-primary/20 hover:border-primary/40 transition-all duration-300 flex flex-col pt-0"
                >
                  {/* Video Demo */}
                  <div className="w-full aspect-square bg-muted relative overflow-hidden">
                    <video
                      src={model.demoVideo}
                      autoPlay
                      loop
                      muted
                      playsInline
                      className="w-full h-full object-cover"
                    />
                  </div>
                  
                  {/* Content */}
                  <div className="px-6 flex flex-col flex-grow">
                    <h3 className="text-xl font-bold mb-3">{model.title}</h3>
                    <p className="text-sm text-muted-foreground mb-4 flex-grow">
                      {model.description}
                    </p>
                    <div className="pt-4 border-t border-border">
                      <p className="text-xs text-muted-foreground mb-4">
                        {model.details}
                      </p>
                      <Button asChild variant="outline" className="w-full">
                        <a 
                          href={model.link}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="inline-flex items-center justify-center gap-2"
                        >
                          Explore {model.title}
                          <ExternalLink className="w-4 h-4" />
                        </a>
                      </Button>
                    </div>
                  </div>
                </Card>
              ))}
            </div>
          </div>

          {/* Additional Info */}
          <div className="max-w-4xl mx-auto text-center">
            <Card className="p-8 bg-gradient-to-br from-primary/5 to-secondary/5 border-primary/20">
              <h3 className="text-2xl font-bold mb-4">Need Help?</h3>
              <p className="text-muted-foreground mb-6">
                If you're experiencing issues with the simulator, check out our troubleshooting guide.
              </p>
              <Button asChild variant="default">
                <a 
                  href="/troubleshooting"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2"
                >
                  Go to Troubleshooting
                  <ExternalLink className="w-4 h-4" />
                </a>
              </Button>
            </Card>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Simulator;
