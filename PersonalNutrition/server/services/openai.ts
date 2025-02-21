import OpenAI from "openai";
import { type InsertQuestionnaire } from "@db/schema";

const openai = new OpenAI();

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

interface Goal {
  name: string;
  description: string;
  targetValue: number;
  unit: string;
  timeframe: string;
}

interface RecommendationResponse {
  supplements: Supplement[];
  goals: Goal[];
  explanation: string;
  basePrice: number;
}

export async function generateRecommendations(questionnaire: InsertQuestionnaire): Promise<RecommendationResponse> {
  const prompt = `Given the following health profile, provide personalized supplement recommendations and trackable goals:

Health Profile:
- Age: ${questionnaire.age}
- Gender: ${questionnaire.gender}
- Height: ${questionnaire.height}cm
- Weight: ${questionnaire.weight}kg
- Activity Level: ${questionnaire.activityLevel}
- Focus Area: ${questionnaire.category}
- Specific Goals: ${JSON.stringify(questionnaire.goals)}
- Dietary Restrictions: ${questionnaire.dietaryRestrictions || 'None'}
- Health Conditions: ${questionnaire.healthConditions || 'None'}

Generate a personalized plan that includes:

1. Core Supplements (2 foundational supplements) - Base price $30:
   - Must include a comprehensive multivitamin/mineral blend
   - Must include omega-3 fatty acids
   - Explain why these specific formulations are crucial for this user's profile
   - Detail how they support the user's specific goals

2. Recommended Add-ons (2-4 targeted supplements) - $10-15 each:
   - Select supplements specifically matched to the user's:
     * Age and gender considerations
     * Activity level requirements
     * Health goals and focus areas
     * Current health conditions
   - Explain why each supplement was chosen for this specific profile

Based on the user's profile, select from these categories as appropriate:

Performance & Recovery (active users):
- Protein powder blends
- BCAAs
- Creatine monohydrate
- Beta-alanine
- L-citrulline

Cognitive Function:
- Lion's mane
- Bacopa monnieri
- L-theanine
- Alpha-GPC
- CDP-choline

Energy & Metabolism:
- Green tea extract
- Cordyceps
- Coenzyme Q10
- L-carnitine
- B-complex vitamins

Stress & Sleep:
- Ashwagandha
- Rhodiola rosea
- Magnesium glycinate
- L-theanine
- GABA

Immune Support:
- Vitamin C
- Zinc
- Elderberry
- Echinacea
- Beta-glucans

Joint & Bone Health:
- Glucosamine
- Chondroitin
- MSM
- Collagen peptides
- Vitamin D3/K2

3. Trackable Goals (2-4 specific goals):
   - Set goals that align with the supplement choices
   - Explain how the supplements support each goal
   - Include realistic timeframes based on the supplements' mechanisms of action
   - Define clear progress indicators

For each supplement include:
- Name
- Why it was specifically chosen for this user's profile
- How it supports their goals and addresses their needs
- Research-backed benefits relevant to their profile
- Precise dosage instructions
- Category ("core" or "recommended")
- Optimal timing ("morning", "afternoon", or "evening")
- Base price (core included in $30 base, recommended $10-15 each)

For each goal include:
- Name
- Why this goal was selected based on their profile
- How the recommended supplements support this goal
- Target value with scientific rationale
- Timeframe based on supplement efficacy

Provide a comprehensive explanation that:
1. Explains why each supplement was selected for this specific user
2. Details how the supplements work together for their goals
3. Addresses any specific health conditions or concerns
4. Explains expected benefits based on their profile
5. Provides a timeline for expected results

The explanation should focus on personalizing the recommendations to the user's specific needs and goals.

Respond in the following JSON format:
{
  "supplements": [
    {
      "name": "Supplement Name",
      "description": "Why this supplement for this user",
      "dosage": "Precise dosage",
      "benefits": ["benefit specific to user profile", "benefit 2", "benefit 3"],
      "category": "core or recommended",
      "timing": "morning/afternoon/evening",
      "basePrice": 0,
      "form": "powder"
    }
  ],
  "goals": [
    {
      "name": "Goal Name",
      "description": "Why this goal matches their profile",
      "targetValue": 100,
      "unit": "measurement unit",
      "timeframe": "timeline based on supplement effects"
    }
  ],
  "explanation": "Comprehensive explanation of why these recommendations suit this specific user",
  "basePrice": 30
}`;

  const response = await openai.chat.completions.create({
    model: "gpt-4",
    messages: [
      {
        role: "system",
        content: "You are an expert nutritionist and researcher specializing in evidence-based supplement recommendations. Focus on explaining why each supplement recommendation is specifically chosen for the user's profile, goals, and needs. Provide detailed scientific explanations while maintaining an approachable tone."
      },
      {
        role: "user",
        content: prompt
      }
    ],
    temperature: 0.7,
  });

  try {
    const result = JSON.parse(response.choices[0].message.content || '') as RecommendationResponse;
    return result;
  } catch (error) {
    console.error('Failed to parse OpenAI response:', error);
    throw new Error('Failed to generate recommendations');
  }
}