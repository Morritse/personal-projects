import { pgTable, text, serial, integer, boolean, jsonb, timestamp, decimal } from "drizzle-orm/pg-core";
import { createInsertSchema, createSelectSchema } from "drizzle-zod";

export const users = pgTable("users", {
  id: serial("id").primaryKey(),
  username: text("username").unique().notNull(),
  password: text("password").notNull(),
});

export const questionnaires = pgTable("questionnaires", {
  id: serial("id").primaryKey(),
  age: integer("age").notNull(),
  gender: text("gender").notNull(),
  height: integer("height").notNull(),
  weight: integer("weight").notNull(),
  activityLevel: text("activity_level").notNull(),
  category: text("category").notNull(),
  goals: jsonb("goals").notNull(),
  dietaryRestrictions: text("dietary_restrictions"),
  healthConditions: text("health_conditions"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const recommendations = pgTable("recommendations", {
  id: serial("id").primaryKey(),
  questionnaireId: integer("questionnaire_id").references(() => questionnaires.id).notNull(),
  supplements: jsonb("supplements").notNull(),
  explanation: text("explanation").notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const supplementPackages = pgTable("supplement_packages", {
  id: serial("id").primaryKey(),
  userId: integer("user_id").references(() => users.id).notNull(),
  basePrice: decimal("base_price", { precision: 10, scale: 2 }).notNull(),
  isActive: boolean("is_active").default(true).notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const supplementProducts = pgTable("supplement_products", {
  id: serial("id").primaryKey(),
  name: text("name").notNull(),
  description: text("description").notNull(),
  timing: text("timing").notNull(),
  basePrice: decimal("base_price", { precision: 10, scale: 2 }).notNull(),
  dosageInstructions: text("dosage_instructions").notNull(),
  isActive: boolean("is_active").default(true).notNull(),
});

export const packageSupplements = pgTable("package_supplements", {
  id: serial("id").primaryKey(),
  packageId: integer("package_id").references(() => supplementPackages.id).notNull(),
  supplementId: integer("supplement_id").references(() => supplementProducts.id).notNull(),
  quantity: integer("quantity").notNull(),
  timing: text("timing").notNull(),
});

export const goalProgress = pgTable("goal_progress", {
  id: serial("id").primaryKey(),
  userId: integer("user_id").references(() => users.id).notNull(),
  goalName: text("goal_name").notNull(),
  targetValue: integer("target_value").notNull(),
  currentValue: integer("current_value").notNull(),
  unit: text("unit").notNull(),
  startDate: timestamp("start_date").defaultNow().notNull(),
  lastUpdated: timestamp("last_updated").defaultNow().notNull(),
});

export const supplementUsage = pgTable("supplement_usage", {
  id: serial("id").primaryKey(),
  userId: integer("user_id").references(() => users.id).notNull(),
  supplementName: text("supplement_name").notNull(),
  dosage: text("dosage").notNull(),
  frequency: text("frequency").notNull(),
  startDate: timestamp("start_date").defaultNow().notNull(),
  lastTaken: timestamp("last_taken").defaultNow().notNull(),
  active: boolean("active").default(true).notNull(),
});

export const insertUserSchema = createInsertSchema(users);
export const selectUserSchema = createSelectSchema(users);
export const insertQuestionnaireSchema = createInsertSchema(questionnaires);
export const selectQuestionnaireSchema = createSelectSchema(questionnaires);
export const insertRecommendationSchema = createInsertSchema(recommendations);
export const selectRecommendationSchema = createSelectSchema(recommendations);
export const insertSupplementPackageSchema = createInsertSchema(supplementPackages);
export const selectSupplementPackageSchema = createSelectSchema(supplementPackages);
export const insertSupplementProductSchema = createInsertSchema(supplementProducts);
export const selectSupplementProductSchema = createSelectSchema(supplementProducts);
export const insertPackageSupplementSchema = createInsertSchema(packageSupplements);
export const selectPackageSupplementSchema = createSelectSchema(packageSupplements);
export const insertGoalProgressSchema = createInsertSchema(goalProgress);
export const selectGoalProgressSchema = createSelectSchema(goalProgress);
export const insertSupplementUsageSchema = createInsertSchema(supplementUsage);
export const selectSupplementUsageSchema = createSelectSchema(supplementUsage);

export type InsertUser = typeof users.$inferInsert;
export type SelectUser = typeof users.$inferSelect;
export type InsertQuestionnaire = typeof questionnaires.$inferInsert;
export type SelectQuestionnaire = typeof questionnaires.$inferSelect;
export type InsertRecommendation = typeof recommendations.$inferInsert;
export type SelectRecommendation = typeof recommendations.$inferSelect;
export type InsertSupplementPackage = typeof supplementPackages.$inferInsert;
export type SelectSupplementPackage = typeof supplementPackages.$inferSelect;
export type InsertSupplementProduct = typeof supplementProducts.$inferInsert;
export type SelectSupplementProduct = typeof supplementProducts.$inferSelect;
export type InsertPackageSupplement = typeof packageSupplements.$inferInsert;
export type SelectPackageSupplement = typeof packageSupplements.$inferSelect;
export type InsertGoalProgress = typeof goalProgress.$inferInsert;
export type SelectGoalProgress = typeof goalProgress.$inferSelect;
export type InsertSupplementUsage = typeof supplementUsage.$inferInsert;
export type SelectSupplementUsage = typeof supplementUsage.$inferSelect;