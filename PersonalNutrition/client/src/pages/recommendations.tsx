import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useParams, Link } from "wouter";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import SupplementCard from "@/components/supplement-card";
import { getRecommendation } from "@/lib/api";

export default function Recommendations() {
  const { id } = useParams();
  const { data: recommendation, isLoading, error } = useQuery({
    queryKey: [`/api/recommendations/${id}`],
    enabled: !!id,
  });

  const [selectedSupplements, setSelectedSupplements] = useState<Set<string>>(new Set());

  const toggleSupplement = (name: string) => {
    setSelectedSupplements(prev => {
      const newSet = new Set(prev);
      if (newSet.has(name)) {
        newSet.delete(name);
      } else {
        newSet.add(name);
      }
      return newSet;
    });
  };

  if (isLoading) {
    return (
      <div className="min-h-screen container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto space-y-4">
          <div className="h-8 bg-muted animate-pulse rounded" />
          <div className="h-24 bg-muted animate-pulse rounded" />
          <div className="grid md:grid-cols-2 gap-4">
            <div className="h-64 bg-muted animate-pulse rounded" />
            <div className="h-64 bg-muted animate-pulse rounded" />
          </div>
        </div>
      </div>
    );
  }

  if (error || !recommendation) {
    return (
      <div className="min-h-screen container mx-auto px-4 py-8 flex items-center justify-center">
        <Card className="w-full max-w-md">
          <CardHeader>
            <CardTitle>Error</CardTitle>
            <CardDescription>
              Failed to load recommendations. Please try again.
            </CardDescription>
          </CardHeader>
          <CardFooter>
            <Link href="/questionnaire">
              <Button className="w-full">Return to Questionnaire</Button>
            </Link>
          </CardFooter>
        </Card>
      </div>
    );
  }

  // Initialize selected supplements if not already done
  if (selectedSupplements.size === 0 && recommendation.supplements.length > 0) {
    setSelectedSupplements(new Set(
      recommendation.supplements
        .filter(s => s.category === "core")
        .map(s => s.name)
    ));
  }

  const coreSupplements = recommendation.supplements.filter(s => s.category === "core");
  const recommendedSupplements = recommendation.supplements.filter(s => s.category === "recommended");

  return (
    <div className="min-h-screen bg-background">
      <header className="container mx-auto px-4 py-6">
        <Link href="/">
          <Button variant="ghost">← Back to Home</Button>
        </Link>
      </header>

      <main className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-3xl font-bold mb-2">Your Personalized Plan</h1>
          <p className="text-muted-foreground mb-8">{recommendation.explanation}</p>

          {coreSupplements.length > 0 && (
            <div className="mb-8">
              <h2 className="text-2xl font-semibold mb-4">Core Supplements</h2>
              <div className="grid md:grid-cols-2 gap-6">
                {coreSupplements.map((supplement, index) => (
                  <SupplementCard
                    key={index}
                    supplement={supplement}
                    selected={selectedSupplements.has(supplement.name)}
                    onToggle={() => toggleSupplement(supplement.name)}
                  />
                ))}
              </div>
            </div>
          )}

          {recommendedSupplements.length > 0 && (
            <div>
              <h2 className="text-2xl font-semibold mb-4">Recommended Supplements</h2>
              <div className="grid md:grid-cols-2 gap-6">
                {recommendedSupplements.map((supplement, index) => (
                  <SupplementCard
                    key={index}
                    supplement={supplement}
                    selected={selectedSupplements.has(supplement.name)}
                    onToggle={() => toggleSupplement(supplement.name)}
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}