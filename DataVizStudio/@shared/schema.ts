import { z } from "zod";
import { createInsertSchema } from "drizzle-zod";

export const sectionSchema = z.object({
  id: z.string().optional(),
  type: z.enum(["text", "image", "table"]),
  content: z.any(),
  description: z.string().optional()
});

export const reportSchema = z.object({
  id: z.number(),
  title: z.string(),
  sections: z.array(sectionSchema),
  createdAt: z.date()
});

export const insertReportSchema = reportSchema.omit({ 
  id: true,
  createdAt: true 
});

export type Report = z.infer<typeof reportSchema>;
export type InsertReport = z.infer<typeof insertReportSchema>;
export type Section = z.infer<typeof sectionSchema>;
