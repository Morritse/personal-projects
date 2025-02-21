import { Progress } from "@/components/ui/progress";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Brain, Dumbbell, Heart } from "lucide-react";
import { SelectGoalProgress } from "@db/schema";

interface GoalProgressTrackerProps {
  goals: SelectGoalProgress[];
}

export default function GoalProgressTracker({ goals }: GoalProgressTrackerProps) {
  const getGoalIcon = (goalName: string) => {
    switch (goalName.toLowerCase()) {
      case 'focus':
      case 'memory':
      case 'cognitive':
        return <Brain className="w-5 h-5 text-primary" />;
      case 'strength':
      case 'muscle':
      case 'fitness':
        return <Dumbbell className="w-5 h-5 text-primary" />;
      case 'wellness':
      case 'health':
      case 'energy':
        return <Heart className="w-5 h-5 text-primary" />;
      default:
        return <Brain className="w-5 h-5 text-primary" />;
    }
  };

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-semibold">Goal Progress</h2>
      <div className="grid md:grid-cols-2 gap-6">
        {goals.map((goal) => {
          const progressPercentage = Math.min(
            100,
            Math.round((goal.currentValue / goal.targetValue) * 100)
          );

          return (
            <Card key={goal.id}>
              <CardHeader className="flex flex-row items-center space-y-0 pb-2">
                <div className="flex flex-1 items-center space-x-4">
                  {getGoalIcon(goal.goalName)}
                  <CardTitle className="text-xl">{goal.goalName}</CardTitle>
                </div>
                <span className="text-2xl font-bold text-primary">
                  {progressPercentage}%
                </span>
              </CardHeader>
              <CardContent>
                <Progress
                  value={progressPercentage}
                  className="h-2 mt-2"
                />
                <div className="mt-2 text-sm text-muted-foreground">
                  {goal.currentValue} / {goal.targetValue} {goal.unit}
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>
    </div>
  );
}
