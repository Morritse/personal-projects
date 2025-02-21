import type { Express } from "express";
import { createServer, type Server } from "http";
import { db } from "@db";
import { questionnaires, recommendations, goalProgress, supplementUsage } from "@db/schema";
import { eq } from "drizzle-orm";
import { generateRecommendations } from "./services/openai";

export function registerRoutes(app: Express): Server {
  const httpServer = createServer(app);

  app.post("/api/questionnaire", async (req, res) => {
    try {
      const questionnaire = await db
        .insert(questionnaires)
        .values({
          age: parseInt(req.body.age),
          gender: req.body.gender,
          height: parseInt(req.body.height),
          weight: parseInt(req.body.weight),
          activityLevel: req.body.activityLevel,
          category: req.body.category,
          goals: req.body.goals,
          dietaryRestrictions: req.body.dietaryRestrictions,
          healthConditions: req.body.healthConditions,
        })
        .returning({ id: questionnaires.id });

      // Generate AI-powered recommendations
      const recommendationData = await generateRecommendations({
        age: parseInt(req.body.age),
        gender: req.body.gender,
        height: parseInt(req.body.height),
        weight: parseInt(req.body.weight),
        activityLevel: req.body.activityLevel,
        category: req.body.category,
        goals: req.body.goals,
        dietaryRestrictions: req.body.dietaryRestrictions,
        healthConditions: req.body.healthConditions,
      });

      // Store goals in the goal_progress table
      if (recommendationData.goals) {
        for (const goal of recommendationData.goals) {
          await db.insert(goalProgress).values({
            userId: null, // Remove user association for now
            goalName: goal.name,
            targetValue: goal.targetValue,
            currentValue: 0, // Start at 0
            unit: goal.unit,
          });
        }
      }

      const recommendation = await db
        .insert(recommendations)
        .values({
          questionnaireId: questionnaire[0].id,
          supplements: recommendationData.supplements,
          explanation: recommendationData.explanation,
        })
        .returning({ id: recommendations.id });

      res.json({ id: recommendation[0].id });
    } catch (error) {
      console.error('Error processing questionnaire:', error);
      res.status(500).send("Failed to process questionnaire");
    }
  });

  app.get("/api/recommendations/:id", async (req, res) => {
    try {
      const recommendation = await db.query.recommendations.findFirst({
        where: eq(recommendations.id, parseInt(req.params.id)),
      });

      if (!recommendation) {
        return res.status(404).send("Recommendation not found");
      }

      // Get associated goals - removing user filter since we don't have users yet
      const goals = await db.query.goalProgress.findMany();

      res.json({
        ...recommendation,
        goals,
      });
    } catch (error) {
      console.error('Error fetching recommendation:', error);
      res.status(500).send("Failed to fetch recommendation");
    }
  });

  return httpServer;
}