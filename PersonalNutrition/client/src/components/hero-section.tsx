import { Button } from "@/components/ui/button";
import { Link } from "wouter";

export default function HeroSection() {
  return (
    <section className="relative bg-gradient-to-b from-primary/10 to-background pt-20 pb-32">
      <div className="container mx-auto px-4">
        <div className="max-w-2xl">
          <h1 className="text-5xl font-bold leading-tight mb-6">
            Personalized Supplements{" "}
            <span className="bg-gradient-to-r from-primary to-primary/80 bg-clip-text text-transparent">
              Tailored to You
            </span>
          </h1>

          {/* Mission statements */}
          <div className="space-y-4 mb-8">
            <p className="text-2xl font-semibold text-muted-foreground">
              Generic supplement packages are built for everybody—not{" "}
              <span className="text-primary font-bold">YOU</span>
            </p>
            <p className="text-xl text-muted-foreground">
              One size doesn't fit all, it fits none. That's why we're here to
              empower <span className="font-semibold">you</span> with the
              ability to take control and make informed decisions about your
              health.
            </p>
            <p className="text-xl font-medium">
              Whether you're a student, athlete, or working professional, we
              want to help you build a nutritional package that truly fits{" "}
              <span className="bg-gradient-to-r from-primary to-primary/80 bg-clip-text text-transparent font-bold">
                YOUR unique needs
              </span>
            </p>
          </div>

          <Link href="/questionnaire">
            <Button size="lg" className="font-semibold">
              Start Your Journey
            </Button>
          </Link>
        </div>
        <div className="absolute right-0 top-1/2 -translate-y-1/2 hidden lg:block w-1/3">
          <img
            src="https://images.unsplash.com/photo-1544367567-0f2fcb009e0b"
            alt="Wellness and Empowerment"
            className="rounded-lg shadow-xl"
          />
        </div>
      </div>
    </section>
  );
}