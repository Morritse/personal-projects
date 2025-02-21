export interface AnalyzedSection {
  type: "text" | "image" | "table";
  title: string;
  content: any;
  description?: string;
}

export interface ReportStructure {
  title: string;
  executiveSummary: string;
  sections: AnalyzedSection[];
}
