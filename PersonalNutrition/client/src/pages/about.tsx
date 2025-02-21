import { Button } from "@/components/ui/button";
import { Link } from "wouter";

export default function About() {
  return (
    <div className="min-h-screen bg-background">
      <header className="container mx-auto px-4 py-6">
        <Link href="/">
          <Button variant="ghost" className="font-semibold">
            ← Back to Home
          </Button>
        </Link>
      </header>

      <main className="container mx-auto px-4 py-8">
        <div className="max-w-3xl mx-auto space-y-12">
          <section>
            <h1 className="text-4xl font-bold mb-6">Our Mission</h1>
            <div className="prose prose-lg max-w-none">
              <p className="text-2xl font-medium text-muted-foreground mb-6">
                Nutrition should not be a one-sided conversation.
              </p>
              
              <p className="mb-6">
                Off-the-shelf supplement options like AG1, by definition, cannot be the best for{" "}
                <span className="text-primary font-semibold">YOU</span> when they have to be made for everyone. 
                You'll end up with a generic solution that doesn't help you achieve your goals.
              </p>

              <div className="my-8 p-6 bg-primary/5 rounded-lg border-l-4 border-primary">
                <p className="text-xl font-medium">
                  We want to empower you with the tools and knowledge to reach your personal goals, 
                  and help you on your journey to optimal health.
                </p>
              </div>
            </div>
          </section>

          <section className="space-y-6">
            <h2 className="text-3xl font-bold">Our Approach</h2>
            <div className="grid md:grid-cols-2 gap-8">
              <div className="space-y-4">
                <h3 className="text-xl font-semibold">Personalized Recommendations</h3>
                <p className="text-muted-foreground">
                  We suggest an individualized plan backed by scientific research, 
                  but you ultimately choose what goes into your mix. Your health journey 
                  should be guided by your needs and preferences.
                </p>
              </div>
              <div className="space-y-4">
                <h3 className="text-xl font-semibold">Active Collaboration</h3>
                <p className="text-muted-foreground">
                  If you aren't getting the results you want, we encourage active 
                  collaboration until you do. Your success is our priority, and we're 
                  here to adjust and refine your plan as needed.
                </p>
              </div>
            </div>
          </section>

          <section className="text-center">
            <Link href="/questionnaire">
              <Button size="lg" className="font-semibold">
                Start Your Journey
              </Button>
            </Link>
          </section>
        </div>
      </main>
    </div>
  );
}
