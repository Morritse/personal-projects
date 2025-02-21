import { InsertReport, Report } from "@shared/schema";

export interface IStorage {
  getReports(): Promise<Report[]>;
  getReport(id: number): Promise<Report | undefined>;
  createReport(report: InsertReport): Promise<Report>;
  updateReport(id: number, report: InsertReport): Promise<Report | undefined>;
}

export class MemStorage implements IStorage {
  private reports: Map<number, Report>;
  private currentId: number;

  constructor() {
    this.reports = new Map();
    this.currentId = 1;
  }

  async getReports(): Promise<Report[]> {
    return Array.from(this.reports.values());
  }

  async getReport(id: number): Promise<Report | undefined> {
    return this.reports.get(id);
  }

  async createReport(report: InsertReport): Promise<Report> {
    const id = this.currentId++;
    const newReport: Report = {
      ...report,
      id,
      createdAt: new Date(),
      sections: report.sections.map((section, index) => ({
        ...section,
        id: `section-${index}`,
        content: section.type === 'image' && typeof section.content === 'string' && !section.content.startsWith('data:') 
          ? `data:image/png;base64,${section.content}`
          : section.content
      }))
    };
    this.reports.set(id, newReport);
    return newReport;
  }

  async updateReport(id: number, report: InsertReport): Promise<Report | undefined> {
    const existing = this.reports.get(id);
    if (!existing) return undefined;

    const updated: Report = {
      ...existing,
      ...report,
      sections: report.sections.map((section, index) => ({
        ...section,
        id: `section-${index}`,
        content: section.type === 'image' && typeof section.content === 'string' && !section.content.startsWith('data:') 
          ? `data:image/png;base64,${section.content}`
          : section.content
      }))
    };
    this.reports.set(id, updated);
    return updated;
  }
}

export const storage = new MemStorage();