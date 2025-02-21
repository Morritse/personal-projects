import { pgTable, text, serial, timestamp, jsonb } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

export const reports = pgTable("reports", {
  id: serial("id").primaryKey(),
  title: text("title").notNull(),
  sections: jsonb("sections").notNull(),
  createdAt: timestamp("created_at").defaultNow(),
});

export const sections = z.array(z.object({
  id: z.string(),
  type: z.enum(["text", "image", "table"]),
  content: z.any(),
}));

export const insertReportSchema = createInsertSchema(reports).omit({
  id: true,
  createdAt: true,
});

export type InsertReport = z.infer<typeof insertReportSchema>;
export type Report = typeof reports.$inferSelect;
export type Section = z.infer<typeof sections>[0];
