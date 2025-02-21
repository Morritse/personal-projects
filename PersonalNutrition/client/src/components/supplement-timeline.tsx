import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { format } from "date-fns";
import { SelectSupplementUsage } from "@db/schema";

interface SupplementTimelineProps {
  supplements: SelectSupplementUsage[];
}

export default function SupplementTimeline({ supplements }: SupplementTimelineProps) {
  const activeSupplement = supplements.filter(s => s.active);
  const inactiveSupplement = supplements.filter(s => !s.active);

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-semibold">Supplement Timeline</h2>
      
      <div className="space-y-4">
        <h3 className="text-lg font-medium">Active Supplements</h3>
        <div className="grid md:grid-cols-2 gap-4">
          {activeSupplement.map((supplement) => (
            <Card key={supplement.id}>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-xl">{supplement.supplementName}</CardTitle>
                <Badge variant="default">Active</Badge>
              </CardHeader>
              <CardContent className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Dosage:</span>
                  <span className="font-medium">{supplement.dosage}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Frequency:</span>
                  <span className="font-medium">{supplement.frequency}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Started:</span>
                  <span className="font-medium">
                    {format(new Date(supplement.startDate), 'MMM d, yyyy')}
                  </span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Last taken:</span>
                  <span className="font-medium">
                    {format(new Date(supplement.lastTaken), 'MMM d, yyyy')}
                  </span>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>

      {inactiveSupplement.length > 0 && (
        <div className="space-y-4">
          <h3 className="text-lg font-medium">Past Supplements</h3>
          <div className="grid md:grid-cols-2 gap-4">
            {inactiveSupplement.map((supplement) => (
              <Card key={supplement.id} className="bg-muted/50">
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-xl">{supplement.supplementName}</CardTitle>
                  <Badge variant="secondary">Inactive</Badge>
                </CardHeader>
                <CardContent className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Last dosage:</span>
                    <span className="font-medium">{supplement.dosage}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Used from:</span>
                    <span className="font-medium">
                      {format(new Date(supplement.startDate), 'MMM d')} - {format(new Date(supplement.lastTaken), 'MMM d, yyyy')}
                    </span>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
