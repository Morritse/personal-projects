import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";
import { Clock } from "lucide-react";

interface Supplement {
  name: string;
  description: string;
  dosage: string;
  benefits: string[];
  category: "core" | "recommended";
  timing: "morning" | "afternoon" | "evening";
  basePrice: number;
  form: "powder";
}

interface SupplementCardProps {
  supplement: Supplement;
  selected: boolean;
  onToggle: (selected: boolean) => void;
}

const timingColors = {
  morning: "text-yellow-500",
  afternoon: "text-orange-500",
  evening: "text-blue-500",
};

const timingLabels = {
  morning: "Morning",
  afternoon: "Afternoon",
  evening: "Evening",
};

export default function SupplementCard({ supplement, selected, onToggle }: SupplementCardProps) {
  return (
    <Card className={`transition-all ${selected ? 'border-primary shadow-lg' : ''}`}>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <div className="space-y-1">
          <CardTitle className="text-xl">{supplement.name}</CardTitle>
          <div className="flex flex-wrap gap-2">
            <Badge
              variant={supplement.category === "core" ? "default" : "secondary"}
            >
              {supplement.category === "core" ? "Core Supplement" : "Recommended"}
            </Badge>
            <div className={`flex items-center gap-1 ${timingColors[supplement.timing]}`}>
              <Clock className="h-4 w-4" />
              <span className="text-sm font-medium">{timingLabels[supplement.timing]}</span>
            </div>
          </div>
        </div>
        <div className="flex flex-col items-end gap-2">
          <Switch
            checked={selected}
            onCheckedChange={onToggle}
          />
          {supplement.category === "core" ? (
            <span className="text-sm text-muted-foreground">Included in base pack</span>
          ) : (
            <span className="text-sm font-medium">+${supplement.basePrice}/mo</span>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <CardDescription className="mb-4 text-base">
          {supplement.description}
        </CardDescription>
        <div className="space-y-4">
          <div>
            <h4 className="font-medium mb-1">Recommended Dosage:</h4>
            <p className="text-sm text-muted-foreground">
              {supplement.dosage} (powder form)
            </p>
          </div>
          <div>
            <h4 className="font-medium mb-1">Key Benefits:</h4>
            <ul className="list-disc list-inside text-sm text-muted-foreground space-y-1">
              {supplement.benefits.map((benefit, index) => (
                <li key={index}>{benefit}</li>
              ))}
            </ul>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}