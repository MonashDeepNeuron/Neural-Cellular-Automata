import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { AlertCircle, Smartphone } from "lucide-react";
import { Button } from "@/components/ui/button";
import Link from 'next/link';
import createMetadata from '@/util/createMetadata';

export const metadata = createMetadata({
	title: 'WebGPU Setup',
	description: 'A tutorial for setting up webGPU on ios'
});

const WebGPUSetup = () => {
  return (
    <div className="min-h-screen bg-background">
      <main className="pt-24 pb-16">
        <div className="container mx-auto px-6">
          {/* Header */}
          <div className="max-w-4xl mx-auto mb-12 text-center">
            <div className="flex justify-center mb-4">
              <Smartphone className="w-16 h-16 text-primary" />
            </div>
            <h1 className="text-4xl md:text-5xl font-bold mb-4 bg-clip-text text-primary">
              Setting Up WebGPU on iOS
            </h1>
            <p className="text-lg text-muted-foreground">
              Enable experimental WebGPU support on your iPhone or iPad
            </p>
          </div>

          {/* Alert */}
          <div className="max-w-4xl mx-auto mb-8">
            <Card className="border-primary/20 bg-primary/5">
              <CardContent>
                <div className="flex gap-3">
                  <AlertCircle className="w-5 h-5 text-primary flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="font-semibold mb-1">Important Note</p>
                    <p className="text-sm text-muted-foreground">
                      WebGPU is not yet natively supported on iOS devices like iPhone and iPad. However, you can enable experimental support by following these steps.
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Setup Steps */}
          <div className="max-w-4xl mx-auto space-y-8">
            <Card>
              <CardHeader>
                <CardTitle>How to Enable WebGPU</CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-4">
                  <div className="flex gap-4">
                    <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center font-semibold text-primary">
                      1
                    </div>
                    <div className="flex-1">
                      <p className="font-medium mb-2">Open the Settings app on your device</p>
                    </div>
                  </div>

                  <div className="flex gap-4">
                    <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center font-semibold text-primary">
                      2
                    </div>
                    <div className="flex-1">
                      <p className="font-medium mb-2">Go to Safari</p>
                      <img 
                        src="webGPUSetup/Step1.jpg" 
                        alt="Navigate to Safari settings"
                        className="rounded-lg border shadow-sm max-w-sm"
                      />
                    </div>
                  </div>

                  <div className="flex gap-4">
                    <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center font-semibold text-primary">
                      3
                    </div>
                    <div className="flex-1">
                      <p className="font-medium mb-2">Go to Advanced</p>
                      <img 
                        src="webGPUSetup/Step2.jpg" 
                        alt="Navigate to Advanced settings"
                        className="rounded-lg border shadow-sm max-w-sm"
                      />
                    </div>
                  </div>

                  <div className="flex gap-4">
                    <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center font-semibold text-primary">
                      4
                    </div>
                    <div className="flex-1">
                      <p className="font-medium mb-2">Go to Experimental Features</p>
                      <img
                        src="webGPUSetup/Step3.jpg" 
                        alt="Navigate to Experimental Features"
                        className="rounded-lg border shadow-sm max-w-sm"
                      />
                    </div>
                  </div>

                  <div className="flex gap-4">
                    <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center font-semibold text-primary">
                      5
                    </div>
                    <div className="flex-1">
                      <p className="font-medium mb-2">Find WebGPU in the list and turn it on</p>
                      <img 
                        src="webGPUSetup/Step4.jpg" 
                        alt="Enable WebGPU"
                        className="rounded-lg border shadow-sm max-w-sm"
                      />
                    </div>
                  </div>

                  <div className="flex gap-4">
                    <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center font-semibold text-primary">
                      6
                    </div>
                    <div className="flex-1">
                      <p className="font-medium">Completely close and reopen Safari (swipe it away in the app switcher)</p>
                    </div>
                  </div>

                  <div className="flex gap-4">
                    <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center font-semibold text-primary">
                      7
                    </div>
                    <div className="flex-1">
                      <p className="font-medium">Return to this website and reload the page</p>
                    </div>
                  </div>
                </div>

                <div className="pt-4 border-t">
                  <div className="flex gap-3">
                    <AlertCircle className="w-5 h-5 text-amber-500 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-semibold mb-1 text-amber-600 dark:text-amber-400">Important</p>
                      <p className="text-sm text-muted-foreground">
                        WebGPU support on iOS is experimental. Some features may be unstable or perform differently compared to desktop browsers.
                      </p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Troubleshooting */}
            <Card>
              <CardHeader>
                <CardTitle>Troubleshooting</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <ul className="space-y-2 text-muted-foreground">
                  <li className="flex gap-2">
                    <span className="text-primary">•</span>
                    <span>Make sure your device is running <strong className="text-foreground">iOS 17 or later</strong></span>
                  </li>
                  <li className="flex gap-2">
                    <span className="text-primary">•</span>
                    <span>Use the <strong className="text-foreground">Safari browser</strong> (This is only tested for Safari)</span>
                  </li>
                </ul>
                
                <div className="pt-4">
                  <p className="text-muted-foreground mb-4">
                    If you're still having issues, please contact us or check back for future updates!
                  </p>
                  <Button asChild>
                    <Link href="/contact">Contact Us</Link>
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </main>
    </div>
  );
};

export default WebGPUSetup;
