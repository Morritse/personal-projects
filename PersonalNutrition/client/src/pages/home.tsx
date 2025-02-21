import { Button } from "@/components/ui/button";
import { Link } from "wouter";
import HeroSection from "@/components/hero-section";

export default function Home() {
  return (
    <div className="min-h-screen">
      <header className="container mx-auto px-4 py-6 flex justify-between items-center">
        <h1 className="text-2xl font-bold bg-gradient-to-r from-primary to-primary/80 bg-clip-text text-transparent">
          Empower
        </h1>
        <div className="flex items-center gap-4">
          <Link href="/about">
            <Button variant="ghost">Our Mission</Button>
          </Link>
          <Link href="/team">
            <Button variant="ghost">Meet Our Team</Button>
          </Link>
          <Link href="/questionnaire">
            <Button>Get Started</Button>
          </Link>
        </div>
      </header>

      <main>
        <HeroSection />

        <section className="container mx-auto px-4 py-16">
          <h2 className="text-3xl font-bold text-center mb-8">How It Works</h2>
          <div className="grid md:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl font-bold text-primary">1</span>
              </div>
              <h3 className="text-xl font-semibold mb-2">Tell Us About You</h3>
              <p className="text-muted-foreground">
                Complete our comprehensive health questionnaire
              </p>
            </div>
            <div className="text-center">
              <div className="w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl font-bold text-primary">2</span>
              </div>
              <h3 className="text-xl font-semibold mb-2">AI Analysis</h3>
              <p className="text-muted-foreground">
                Our AI nutritionist analyzes your needs
              </p>
            </div>
            <div className="text-center">
              <div className="w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl font-bold text-primary">3</span>
              </div>
              <h3 className="text-xl font-semibold mb-2">Custom Package</h3>
              <p className="text-muted-foreground">
                Receive your personalized supplement recommendations
              </p>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}