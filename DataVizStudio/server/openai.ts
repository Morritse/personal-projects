import OpenAI from "openai";
import { AnalyzedSection, ReportStructure } from "./types";

const openai = new OpenAI({
  apiKey: process.env.OPENROUTER_API_KEY,
  baseURL: "https://openrouter.ai/api/v1",
  defaultHeaders: {
    "HTTP-Referer": "https://replit.com",
    "X-Title": "AI Report Generator"
  },
});

function cleanJsonResponse(response: string): string {
  // Remove markdown code block markers
  response = response.replace(/```json\n?/g, '').replace(/```\n?/g, '');

  // Remove any markdown headers
  response = response.replace(/^#+ .*$/gm, '');

  // Clean up any remaining markdown syntax
  response = response.replace(/[\*\_\#\`]/g, '');

  return response.trim();
}

function ensureJsonResponse(text: string): any {
  try {
    // First try direct JSON parse
    return JSON.parse(text);
  } catch (e) {
    try {
      // Try to find JSON-like structure
      const jsonMatch = text.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        return JSON.parse(jsonMatch[0]);
      }
      throw new Error("No valid JSON found in response");
    } catch (e2) {
      console.error("Failed to parse response:", text);
      throw new Error("Could not parse response as JSON");
    }
  }
}

export async function structureReport(content: {
  title?: string;
  rawText: string;
  images: string[];
}): Promise<ReportStructure> {
  try {
    console.log("Structuring report with:", {
      title: content.title,
      textLength: content.rawText.length,
      imageCount: content.images.length
    });

    const response = await openai.chat.completions.create({
      model: "google/gemini-pro",
      messages: [
        {
          role: "system",
          content: "You are an expert report writer. Your responses must be in valid JSON format. Structure content logically with sections that group related information together."
        },
        {
          role: "user",
          content: `Analyze this content and create a structured report. Your response must be in this exact JSON format:
          {
            "title": "report title",
            "executiveSummary": "brief summary of key points",
            "sections": [
              {
                "title": "section title",
                "type": "text",
                "content": "professionally written content that groups related information"
              }
            ]
          }

          Content to analyze: ${content.rawText}`
        }
      ],
      response_format: { type: "json_object" }
    });

    console.log("Raw response received:", response.choices[0].message.content);

    const cleanedResponse = cleanJsonResponse(response.choices[0].message.content || "{}");
    console.log("Cleaned response:", cleanedResponse);

    const result = ensureJsonResponse(cleanedResponse);
    let finalSections = [...(result.sections || [])];

    if (content.images && content.images.length > 0) {
      console.log("Processing images...");
      for (let i = 0; i < content.images.length; i++) {
        const img = content.images[i];
        console.log(`Processing image ${i + 1}/${content.images.length}`);

        try {
          const analysis = await analyzeImage(img);
          console.log(`Image ${i + 1} analysis:`, analysis);

          // Find most relevant section for this image
          const relevantSectionIndex = finalSections.findIndex(section =>
            section.type === "text" &&
            section.content.toLowerCase().includes(analysis.description?.toLowerCase() || "")
          );

          // Create image section
          const imgSection = {
            type: "image" as const,
            title: analysis.suggestedTitle || `Figure ${i + 1}`,
            content: img,
            description: analysis.description
          };

          // Insert image after its most relevant section or in a logical position
          if (relevantSectionIndex !== -1) {
            finalSections.splice(relevantSectionIndex + 1, 0, imgSection);
          } else {
            // If no relevant section found, try to group with other images or place at end
            const lastImageIndex = finalSections.slice().reverse().findIndex(s => s.type === "image");
            if (lastImageIndex !== -1) {
              finalSections.splice(finalSections.length - lastImageIndex, 0, imgSection);
            } else {
              finalSections.push(imgSection);
            }
          }
        } catch (error) {
          console.error(`Error processing image ${i + 1}:`, error);
          finalSections.push({
            type: "image",
            title: `Image ${i + 1}`,
            content: img,
            description: "Unable to analyze this image"
          });
        }
      }
    }

    const finalReport = {
      title: result.title || content.title || "Analysis Report",
      executiveSummary: result.executiveSummary || "",
      sections: finalSections
    };

    console.log("Final report structure:", {
      title: finalReport.title,
      summaryLength: finalReport.executiveSummary?.length || 0,
      numSections: finalReport.sections?.length || 0
    });

    return finalReport;
  } catch (error: any) {
    console.error("Report structuring error:", error);
    throw new Error(`Failed to structure report: ${error.message}`);
  }
}

export async function analyzeImage(base64Image: string): Promise<{
  description: string;
  suggestedTitle: string;
  relevantMetrics?: string[];
}> {
  try {
    console.log("Analyzing image...");

    // Handle both PNG and JPEG formats
    if (!base64Image.startsWith('data:image/')) {
      throw new Error("Invalid image format - must start with data:image/");
    }

    console.log("Sending image to Gemini for analysis...");

    const response = await openai.chat.completions.create({
      model: "google/gemini-pro-vision",
      messages: [
        {
          role: "system",
          content: "You are an expert at analyzing images. Your responses must be in valid JSON format only."
        },
        {
          role: "user",
          content: [
            {
              type: "text",
              text: "Analyze this image and provide ONLY a JSON response with these exact fields: description (string), suggestedTitle (string), and optionally relevantMetrics (string array). Do not include any explanatory text."
            },
            {
              type: "image_url",
              image_url: {
                url: base64Image
              }
            }
          ]
        }
      ],
      response_format: { type: "json_object" }
    });

    if (!response.choices?.[0]?.message?.content) {
      console.error("Invalid response format from Gemini:", response);
      throw new Error("Invalid response format from Gemini");
    }

    const cleanedResponse = cleanJsonResponse(response.choices[0].message.content);
    console.log("Cleaned image analysis response:", cleanedResponse);

    const result = ensureJsonResponse(cleanedResponse);
    console.log("Image analysis completed successfully");
    return result;
  } catch (error: any) {
    console.error("Error during image analysis:", error);
    const errorDetails = error.response?.data?.error || error.message;
    console.error("Detailed error:", errorDetails);
    throw new Error(`Failed to analyze image: ${error.message}`);
  }
}

export async function enhanceText(text: string): Promise<string> {
  try {
    console.log("Enhancing text:", text.substring(0, 100) + "...");
    const response = await openai.chat.completions.create({
      model: "google/gemini-pro",
      messages: [
        {
          role: "system",
          content: "You are a professional writing assistant. Your response must be in valid JSON format only."
        },
        {
          role: "user",
          content: `Enhance this text to be more professional and engaging while maintaining accuracy. Return ONLY a JSON object with an 'enhanced' field containing the improved version: ${text}`
        }
      ],
      response_format: { type: "json_object" }
    });

    const cleanedResponse = cleanJsonResponse(response.choices[0].message.content || "{}");
    const result = ensureJsonResponse(cleanedResponse);
    console.log("Text enhanced successfully");
    return result.enhanced || text;
  } catch (error: any) {
    console.error("Text enhancement error:", error);
    throw new Error(`Failed to enhance text: ${error.message}`);
  }
}

export async function suggestDataVisualization(data: string): Promise<{
  type: "table" | "list";
  structure: any;
}> {
  try {
    console.log("Suggesting visualization for data:", data.substring(0, 100) + "...");
    const response = await openai.chat.completions.create({
      model: "google/gemini-pro",
      messages: [
        {
          role: "system",
          content: "You are a data visualization expert. Your response must be in valid JSON format only."
        },
        {
          role: "user",
          content: `Analyze this content and suggest the best visualization format. Return ONLY a JSON object with 'type' ("table" or "list") and 'structure' fields: ${data}`
        }
      ],
      response_format: { type: "json_object" }
    });

    const cleanedResponse = cleanJsonResponse(response.choices[0].message.content || "{}");
    const result = ensureJsonResponse(cleanedResponse);
    console.log("Data visualization suggestion generated");
    return result;
  } catch (error: any) {
    console.error("Visualization suggestion error:", error);
    throw new Error(`Failed to suggest data visualization: ${error.message}`);
  }
}