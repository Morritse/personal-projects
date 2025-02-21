import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Button } from "@/components/ui/button";
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Checkbox } from "@/components/ui/checkbox";
import { Card, CardContent } from "@/components/ui/card";
import { useMutation } from "@tanstack/react-query";
import { useLocation } from "wouter";
import { submitQuestionnaire } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";
import TileSelector from "./tile-selector";

const goalCategories = {
  brain: [
    { id: "energy", label: "Increase Energy" },
    { id: "focus", label: "Improve Focus" },
    { id: "memory", label: "Enhance Memory" },
    { id: "nootropics", label: "Cognitive Enhancement" },
    { id: "mental_clarity", label: "Mental Clarity" },
    { id: "stress_reduction", label: "Stress Management" }
  ],
  body: [
    { id: "muscle_gain", label: "Build Muscle" },
    { id: "weight_loss", label: "Weight Loss" },
    { id: "athletic", label: "Athletic Performance" },
    { id: "recovery", label: "Post-Workout Recovery" },
    { id: "endurance", label: "Endurance" },
    { id: "joint_health", label: "Joint Health" },
    { id: "immune_support", label: "Immune Support" }
  ],
  soul: [
    { id: "sleep", label: "Better Sleep" },
    { id: "stress", label: "Reduce Stress" },
    { id: "anxiety", label: "Manage Anxiety" },
    { id: "mood", label: "Mood Enhancement" },
    { id: "meditation", label: "Meditation Support" },
    { id: "relaxation", label: "Deep Relaxation" }
  ],
};

const formSchema = z.object({
  category: z.string().min(1, "Please select a category"),
  goals: z.array(z.string()).min(1, "Please select at least one goal"),
  age: z.string().min(1, "Age is required"),
  gender: z.string().min(1, "Gender is required"),
  heightFt: z.string().min(1, "Height (ft) is required"),
  heightIn: z.string().min(1, "Height (in) is required"),
  weight: z.string().min(1, "Weight is required"),
  activityLevel: z.string().min(1, "Activity level is required"),
  dietaryRestrictions: z.string(),
  healthConditions: z.string(),
});


export default function QuestionnaireForm() {
  const [step, setStep] = useState(1);
  const [, setLocation] = useLocation();
  const { toast } = useToast();

  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      category: "",
      goals: [],
      age: "",
      gender: "",
      heightFt: "",
      heightIn: "",
      weight: "",
      activityLevel: "",
      dietaryRestrictions: "",
      healthConditions: "",
    },
  });

  const mutation = useMutation({
    mutationFn: (values: z.infer<typeof formSchema>) => {
      const heightInCm = (parseInt(values.heightFt) * 30.48) + (parseInt(values.heightIn) * 2.54);
      const weightInKg = parseInt(values.weight) * 0.453592;

      return submitQuestionnaire({
        ...values,
        height: Math.round(heightInCm),
        weight: Math.round(weightInKg),
      });
    },
    onSuccess: (data) => {
      setLocation(`/recommendations/${data.id}`);
    },
    onError: (error) => {
      toast({
        title: "Error",
        description: "Failed to submit questionnaire. Please try again.",
        variant: "destructive",
      });
    },
  });

  function onSubmit(values: z.infer<typeof formSchema>) {
    mutation.mutate(values);
  }

  return (
    <div className="space-y-8">
      {/* Progress indicator */}
      <div className="flex justify-center mb-8">
        <div className="flex items-center space-x-2">
          <div className={`w-3 h-3 rounded-full ${step === 1 ? 'bg-primary' : 'bg-primary/20'}`} />
          <div className="w-12 h-0.5 bg-primary/20" />
          <div className={`w-3 h-3 rounded-full ${step === 2 ? 'bg-primary' : 'bg-primary/20'}`} />
        </div>
      </div>

      <Form {...form}>
        <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-8">
          {step === 1 && (
            <Card className="border-2 transition-all hover:border-primary/20">
              <CardContent className="pt-6 space-y-8">
                <FormField
                  control={form.control}
                  name="category"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel className="text-lg font-semibold">What area would you like to focus on?</FormLabel>
                      <FormControl>
                        <TileSelector
                          selected={field.value}
                          onSelect={(category) => {
                            field.onChange(category);
                            form.setValue('goals', []);
                          }}
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                {form.watch('category') && (
                  <FormField
                    control={form.control}
                    name="goals"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel className="text-lg font-semibold">Select your specific goals:</FormLabel>
                        <div className="grid sm:grid-cols-2 gap-4 mt-4">
                          {goalCategories[form.watch('category') as keyof typeof goalCategories]?.map((item) => (
                            <FormField
                              key={item.id}
                              control={form.control}
                              name="goals"
                              render={({ field }) => (
                                <FormItem
                                  key={item.id}
                                  className="flex items-start space-x-3 space-y-0 bg-card hover:bg-accent/5 p-4 rounded-lg transition-colors"
                                >
                                  <FormControl>
                                    <Checkbox
                                      checked={field.value?.includes(item.id)}
                                      onCheckedChange={(checked) => {
                                        return checked
                                          ? field.onChange([...field.value, item.id])
                                          : field.onChange(
                                              field.value?.filter(
                                                (value) => value !== item.id
                                              )
                                            );
                                      }}
                                    />
                                  </FormControl>
                                  <FormLabel className="font-medium cursor-pointer">
                                    {item.label}
                                  </FormLabel>
                                </FormItem>
                              )}
                            />
                          ))}
                        </div>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                )}

                <Button
                  type="button"
                  className="w-full"
                  disabled={!form.watch('category') || form.watch('goals').length === 0}
                  onClick={() => setStep(2)}
                >
                  Continue
                </Button>
              </CardContent>
            </Card>
          )}

          {step === 2 && (
            <Card className="border-2 transition-all hover:border-primary/20">
              <CardContent className="pt-6 space-y-8">
                <div className="grid sm:grid-cols-2 gap-6">
                  <FormField
                    control={form.control}
                    name="age"
                    render={({ field }) => (
                      <FormItem className="space-y-2">
                        <FormLabel className="text-base">Age</FormLabel>
                        <FormControl>
                          <Input
                            type="number"
                            min="1"
                            max="120"
                            className="text-lg"
                            {...field}
                          />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />

                  <FormField
                    control={form.control}
                    name="gender"
                    render={({ field }) => (
                      <FormItem className="space-y-2">
                        <FormLabel className="text-base">Gender</FormLabel>
                        <Select onValueChange={field.onChange} defaultValue={field.value}>
                          <FormControl>
                            <SelectTrigger className="text-lg">
                              <SelectValue placeholder="Select gender" />
                            </SelectTrigger>
                          </FormControl>
                          <SelectContent>
                            <SelectItem value="male">Male</SelectItem>
                            <SelectItem value="female">Female</SelectItem>
                            <SelectItem value="other">Other</SelectItem>
                          </SelectContent>
                        </Select>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </div>

                <div className="grid sm:grid-cols-3 gap-6">
                  <FormField
                    control={form.control}
                    name="heightFt"
                    render={({ field }) => (
                      <FormItem className="space-y-2">
                        <FormLabel className="text-base">Height (ft)</FormLabel>
                        <FormControl>
                          <Input
                            type="number"
                            min="1"
                            max="8"
                            className="text-lg"
                            {...field}
                          />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />

                  <FormField
                    control={form.control}
                    name="heightIn"
                    render={({ field }) => (
                      <FormItem className="space-y-2">
                        <FormLabel className="text-base">Height (in)</FormLabel>
                        <FormControl>
                          <Input
                            type="number"
                            min="0"
                            max="11"
                            className="text-lg"
                            {...field}
                          />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />

                  <FormField
                    control={form.control}
                    name="weight"
                    render={({ field }) => (
                      <FormItem className="space-y-2">
                        <FormLabel className="text-base">Weight (lbs)</FormLabel>
                        <FormControl>
                          <Input
                            type="number"
                            min="1"
                            className="text-lg"
                            {...field}
                          />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </div>

                <FormField
                  control={form.control}
                  name="activityLevel"
                  render={({ field }) => (
                    <FormItem className="space-y-4">
                      <FormLabel className="text-base">Activity Level</FormLabel>
                      <FormControl>
                        <RadioGroup
                          onValueChange={field.onChange}
                          defaultValue={field.value}
                          className="grid sm:grid-cols-3 gap-4"
                        >
                          {[
                            { value: "sedentary", label: "Sedentary", description: "Little to no exercise" },
                            { value: "moderate", label: "Moderately Active", description: "Exercise 3-5 times a week" },
                            { value: "very_active", label: "Very Active", description: "Exercise 6+ times a week" }
                          ].map((level) => (
                            <FormItem key={level.value} className="space-y-0">
                              <FormControl>
                                <label className="flex flex-col items-center space-y-2 rounded-lg border-2 p-4 cursor-pointer hover:bg-accent/5 [&:has([data-state=checked])]:border-primary">
                                  <RadioGroupItem value={level.value} className="sr-only" />
                                  <span className="font-medium">{level.label}</span>
                                  <span className="text-sm text-muted-foreground text-center">
                                    {level.description}
                                  </span>
                                </label>
                              </FormControl>
                            </FormItem>
                          ))}
                        </RadioGroup>
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <div className="grid sm:grid-cols-2 gap-6">
                  <FormField
                    control={form.control}
                    name="dietaryRestrictions"
                    render={({ field }) => (
                      <FormItem className="space-y-2">
                        <FormLabel className="text-base">Dietary Restrictions</FormLabel>
                        <FormControl>
                          <Textarea
                            placeholder="List any dietary restrictions or allergies"
                            className="resize-none min-h-[100px] text-base"
                            {...field}
                          />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />

                  <FormField
                    control={form.control}
                    name="healthConditions"
                    render={({ field }) => (
                      <FormItem className="space-y-2">
                        <FormLabel className="text-base">Health Conditions</FormLabel>
                        <FormControl>
                          <Textarea
                            placeholder="List any existing health conditions or medications"
                            className="resize-none min-h-[100px] text-base"
                            {...field}
                          />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </div>

                <div className="flex gap-4 pt-4">
                  <Button
                    type="button"
                    variant="outline"
                    size="lg"
                    className="flex-1"
                    onClick={() => setStep(1)}
                  >
                    Back
                  </Button>
                  <Button
                    type="submit"
                    size="lg"
                    className="flex-1"
                    disabled={mutation.isPending}
                  >
                    {mutation.isPending ? "Generating Recommendations..." : "Get Recommendations"}
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}
        </form>
      </Form>
    </div>
  );
}