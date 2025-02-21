import QuestionnaireForm from "@/components/questionnaire-form";
import { Link } from "wouter";
import { Button } from "@/components/ui/button";

export default function Questionnaire() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-primary/5 to-background">
      <header className="container mx-auto px-4 py-6">
        <Link href="/">
          <Button variant="ghost" className="font-semibold">
            ← Back to Home
          </Button>
        </Link>
      </header>

      <main className="container mx-auto px-4 py-8">
        <div className="max-w-2xl mx-auto">
          <div className="text-center mb-8">
            <h1 className="text-4xl font-bold mb-2">Health Profile Questionnaire</h1>
            <p className="text-muted-foreground text-lg">
              Tell us about yourself so we can create your personalized supplement plan.
            </p>
          </div>
          <QuestionnaireForm />
        </div>
      </main>
    </div>
  );
}