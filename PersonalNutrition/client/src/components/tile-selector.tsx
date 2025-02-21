import { Card } from "@/components/ui/card";
import { Brain, Dumbbell, Heart } from "lucide-react";

interface TileSelectorProps {
  selected?: string;
  onSelect: (category: string) => void;
}

export default function TileSelector({ selected, onSelect }: TileSelectorProps) {
  return (
    <div className="grid md:grid-cols-3 gap-6">
      <Card
        className={`p-6 cursor-pointer transition-all hover:border-primary ${
          selected === 'brain' ? 'border-primary bg-primary/5' : ''
        }`}
        onClick={() => onSelect('brain')}
      >
        <div className="text-center space-y-4">
          <div className="mx-auto w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center">
            <Brain className="w-6 h-6 text-primary" />
          </div>
          <div>
            <h3 className="font-semibold mb-2">Mind</h3>
            <p className="text-sm text-muted-foreground">
              Energy, focus, and cognitive enhancement
            </p>
          </div>
        </div>
      </Card>

      <Card
        className={`p-6 cursor-pointer transition-all hover:border-primary ${
          selected === 'body' ? 'border-primary bg-primary/5' : ''
        }`}
        onClick={() => onSelect('body')}
      >
        <div className="text-center space-y-4">
          <div className="mx-auto w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center">
            <Dumbbell className="w-6 h-6 text-primary" />
          </div>
          <div>
            <h3 className="font-semibold mb-2">Body</h3>
            <p className="text-sm text-muted-foreground">
              Muscle gain, weight loss, and athletic performance
            </p>
          </div>
        </div>
      </Card>

      <Card
        className={`p-6 cursor-pointer transition-all hover:border-primary ${
          selected === 'soul' ? 'border-primary bg-primary/5' : ''
        }`}
        onClick={() => onSelect('soul')}
      >
        <div className="text-center space-y-4">
          <div className="mx-auto w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center">
            <Heart className="w-6 h-6 text-primary" />
          </div>
          <div>
            <h3 className="font-semibold mb-2">Soul</h3>
            <p className="text-sm text-muted-foreground">
              Relaxation, better sleep, and stress reduction
            </p>
          </div>
        </div>
      </Card>
    </div>
  );
}
