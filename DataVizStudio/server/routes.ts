import type { Express } from "express";
import { createServer } from "http";
import { storage } from "./storage";
import { enhanceText, analyzeImage, structureReport, suggestDataVisualization } from "./openai";
import { insertReportSchema } from "@shared/schema";

export async function registerRoutes(app: Express) {
  app.post("/api/reports/generate", async (req, res) => {
    const { title, text, images } = req.body;

    if (!text && (!images || images.length === 0)) {
      res.status(400).json({ message: "Text or images are required" });
      return;
    }

    try {
      console.log("Starting report generation with:", {
        hasTitle: !!title,
        textLength: text?.length || 0,
        numImages: images?.length || 0
      });

      let reportStructure;
      try {
        reportStructure = await structureReport({
          title,
          rawText: text || "",
          images: images || []
        });
        console.log("Report structure received:", {
          title: reportStructure.title,
          numSections: reportStructure.sections?.length || 0,
          hasSummary: !!reportStructure.executiveSummary
        });
      } catch (error) {
        console.error("Error in structuring report:", error);
        throw new Error(`Failed to structure report: ${error.message}`);
      }

      const sections = [];

      if (reportStructure.executiveSummary) {
        sections.push({
          id: "exec-summary",
          type: "text",
          content: reportStructure.executiveSummary
        });
      }

      if (images && images.length > 0) {
        console.log("Processing images...");
        for (let i = 0; i < images.length; i++) {
          try {
            const img = images[i];
            console.log(`Processing image ${i + 1}/${images.length}`);
            const analysis = await analyzeImage(img);

            // Ensure image has proper data URL format
            const imageContent = img.startsWith('data:') ? img : `data:image/png;base64,${img}`;

            sections.push({
              id: `image-${i}`,
              type: "image",
              content: imageContent,
              description: analysis.description
            });
          } catch (error) {
            console.error(`Error processing image ${i + 1}:`, error);
            // Still try to add the image even if analysis fails
            const imageContent = images[i].startsWith('data:') ? images[i] : `data:image/png;base64,${images[i]}`;
            sections.push({
              id: `image-${i}`,
              type: "image",
              content: imageContent
            });
          }
        }
      }

      if (reportStructure.sections) {
        reportStructure.sections.forEach((section: any, index: number) => {
          sections.push({
            id: `section-${index}`,
            type: section.type,
            content: section.content
          });
        });
      }

      const report = {
        title: reportStructure.title || title || "Generated Report",
        sections
      };

      // Store the report in memory for preview
      const savedReport = await storage.createReport(report);

      res.json({
        ...report,
        id: savedReport.id,
        previewUrl: `/preview/${savedReport.id}`
      });
    } catch (error: any) {
      console.error("Report generation error:", error);
      res.status(500).json({ 
        message: error.message || "Failed to generate report",
        details: error.stack
      });
    }
  });

  app.get("/api/reports", async (_req, res) => {
    const reports = await storage.getReports();
    res.json(reports);
  });

  app.get("/api/reports/:id", async (req, res) => {
    const report = await storage.getReport(parseInt(req.params.id));
    if (!report) {
      res.status(404).json({ message: "Report not found" });
      return;
    }
    res.json(report);
  });

  app.post("/api/reports", async (req, res) => {
    const parsed = insertReportSchema.safeParse(req.body);
    if (!parsed.success) {
      res.status(400).json({ message: "Invalid report data" });
      return;
    }
    const report = await storage.createReport(parsed.data);
    res.json(report);
  });

  app.patch("/api/reports/:id", async (req, res) => {
    const parsed = insertReportSchema.safeParse(req.body);
    if (!parsed.success) {
      res.status(400).json({ message: "Invalid report data" });
      return;
    }
    const report = await storage.updateReport(parseInt(req.params.id), parsed.data);
    if (!report) {
      res.status(404).json({ message: "Report not found" });
      return;
    }
    res.json(report);
  });

  app.post("/api/enhance", async (req, res) => {
    const { text } = req.body;
    if (!text) {
      res.status(400).json({ message: "Text is required" });
      return;
    }
    try {
      const enhanced = await enhanceText(text);
      res.json({ enhanced });
    } catch (error: any) {
      res.status(500).json({ message: error.message });
    }
  });

  app.post("/api/analyze/image", async (req, res) => {
    const { image } = req.body;
    if (!image) {
      res.status(400).json({ message: "Image is required" });
      return;
    }
    try {
      const analysis = await analyzeImage(image);
      res.json(analysis);
    } catch (error: any) {
      res.status(500).json({ message: error.message });
    }
  });

  app.post("/api/visualize", async (req, res) => {
    const { data } = req.body;
    if (!data) {
      res.status(400).json({ message: "Data is required" });
      return;
    }
    try {
      const visualization = await suggestDataVisualization(data);
      res.json(visualization);
    } catch (error: any) {
      res.status(500).json({ message: error.message });
    }
  });

  app.get("/preview/:id", async (req, res) => {
    const report = await storage.getReport(parseInt(req.params.id));
    if (!report) {
      res.status(404).send("Report not found");
      return;
    }

    const html = `
      <!DOCTYPE html>
      <html>
      <head>
        <title>${report.title}</title>
        <style>
          body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
          }
          h1, h2, h3 {
            color: #2c3e50;
          }
          .section {
            margin-bottom: 30px;
            padding: 20px;
            background: #f9f9f9;
            border-radius: 5px;
          }
          .image-section {
            text-align: center;
          }
          .image-section img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
            margin: 10px 0;
          }
          .image-description {
            font-style: italic;
            color: #666;
            margin-top: 10px;
          }
          table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
          }
          th, td {
            padding: 12px;
            border: 1px solid #ddd;
            text-align: left;
          }
          th {
            background-color: #f5f5f5;
          }
          .executive-summary {
            background-color: #f8f9fa;
            border-left: 4px solid #2c3e50;
            padding: 15px;
            margin: 20px 0;
          }
        </style>
      </head>
      <body>
        <h1>${report.title}</h1>

        ${report.sections.map((section: any) => {
          switch (section.type) {
            case "text":
              return `
                <div class="section">
                  ${section.title ? `<h2>${section.title}</h2>` : ''}
                  <div>${section.content}</div>
                </div>`;
            case "image":
              return `
                <div class="section image-section">
                  ${section.title ? `<h3>${section.title}</h3>` : ''}
                  <img src="${section.content}" alt="${section.title || 'Report image'}">
                  ${section.description ? `
                    <p class="image-description">${section.description}</p>
                  ` : ''}
                </div>`;
            case "table":
              return `
                <div class="section">
                  ${section.title ? `<h2>${section.title}</h2>` : ''}
                  <table>
                    <thead>
                      <tr>${section.content.headers.map((h: string) => `<th>${h}</th>`).join("")}</tr>
                    </thead>
                    <tbody>
                      ${section.content.rows.map((row: string[]) => 
                        `<tr>${row.map((cell: string) => `<td>${cell}</td>`).join("")}</tr>`
                      ).join("")}
                    </tbody>
                  </table>
                </div>`;
            default:
              return '';
          }
        }).join("")}
      </body>
      </html>
    `;

    res.setHeader("Content-Type", "text/html");
    res.send(html);
  });

  const httpServer = createServer(app);
  return httpServer;
}