import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Upload, Loader2 } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { apiRequest } from "@/lib/queryClient";
import { Section } from "@shared/schema";

interface Props {
  onGenerate: (report: { title: string; sections: Section[] }) => void;
}

export function ReportGenerator({ onGenerate }: Props) {
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [images, setImages] = useState<string[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const { toast } = useToast();

  const handleImageUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files) return;

    const newImages: string[] = [];
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      try {
        const base64 = await new Promise<string>((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve(reader.result as string);
          reader.onerror = reject;
          reader.readAsDataURL(file);
        });
        newImages.push(base64);
      } catch (error) {
        console.error("Failed to read image:", error);
        toast({
          title: "Error",
          description: `Failed to process image: ${file.name}`,
          variant: "destructive",
        });
      }
    }

    setImages([...images, ...newImages]);
  };

  const generateReport = async () => {
    if (!content && images.length === 0) {
      toast({
        title: "Error",
        description: "Please provide some content or images to generate the report",
        variant: "destructive",
      });
      return;
    }

    try {
      setIsGenerating(true);
      const res = await apiRequest("POST", "/api/reports/generate", {
        title,
        text: content,
        images: images.map(img => img.split(",")[1]) // Remove data:image/jpeg;base64, prefix
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.message || "Failed to generate report");
      }

      const report = await res.json();
      onGenerate(report);
      toast({
        title: "Success",
        description: "Report generated successfully",
      });
    } catch (error) {
      console.error("Failed to generate report:", error);
      toast({
        title: "Error",
        description: error.message || "Failed to generate report. Please try with a smaller amount of content.",
        variant: "destructive",
      });
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Generate Report</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <Input
          placeholder="Report Title (optional)"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
        />

        <div className="space-y-2">
          <Textarea
            placeholder="Paste your content here... Our AI will help structure and format it into a professional report."
            value={content}
            onChange={(e) => setContent(e.target.value)}
            className="min-h-[200px]"
          />
        </div>

        <div className="space-y-4">
          <Input
            type="file"
            accept="image/*"
            multiple
            onChange={handleImageUpload}
            className="hidden"
            id="image-upload"
          />
          <label
            htmlFor="image-upload"
            className="flex items-center justify-center gap-2 p-4 border-2 border-dashed rounded-lg cursor-pointer hover:bg-gray-50"
          >
            <Upload className="h-4 w-4" />
            <span>Upload Images</span>
          </label>

          {images.length > 0 && (
            <div className="grid grid-cols-2 gap-2">
              {images.map((image, index) => (
                <div key={index} className="relative">
                  <img
                    src={image}
                    alt={`Uploaded ${index + 1}`}
                    className="w-full h-32 object-cover rounded-lg"
                  />
                  <Button
                    variant="destructive"
                    size="sm"
                    className="absolute top-1 right-1"
                    onClick={() => setImages(images.filter((_, i) => i !== index))}
                  >
                    Remove
                  </Button>
                </div>
              ))}
            </div>
          )}
        </div>

        <Button
          className="w-full"
          onClick={generateReport}
          disabled={isGenerating}
        >
          {isGenerating ? (
            <>
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              Generating Report...
            </>
          ) : (
            "Generate Report"
          )}
        </Button>
      </CardContent>
    </Card>
  );
}