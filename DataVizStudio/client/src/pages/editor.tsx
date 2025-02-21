import { useState } from "react";
import { useParams, useLocation } from "wouter";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Report, Section } from "@shared/schema";
import { Button } from "@/components/ui/button";
import { ReportGenerator } from "@/components/report-generator";
import { ReportSection } from "@/components/report-section";
import { Save, Download, ExternalLink } from "lucide-react";
import { apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";

export default function Editor() {
  const { id } = useParams();
  const [, setLocation] = useLocation();
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const [title, setTitle] = useState<string>("");
  const [sections, setSections] = useState<Section[]>([]);
  const [isPreviewOpen, setIsPreviewOpen] = useState(false);

  const { data: report, isLoading } = useQuery<Report>({
    queryKey: [`/api/reports/${id}`],
    enabled: !!id,
  });

  const saveMutation = useMutation({
    mutationFn: async (report: { title: string; sections: Section[] }) => {
      if (id) {
        return apiRequest("PATCH", `/api/reports/${id}`, report);
      } else {
        return apiRequest("POST", "/api/reports", report);
      }
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/reports"] });
      toast({
        title: "Success",
        description: "Report saved successfully",
      });
      if (!id) {
        setLocation("/");
      }
    },
  });

  const updateSection = (id: string, content: unknown) => {
    setSections(
      sections.map((s) => (s.id === id ? { ...s, content } : s))
    );
  };

  const deleteSection = (id: string) => {
    setSections(sections.filter((s) => s.id !== id));
  };

  const exportHtml = async () => {
    const res = await apiRequest("POST", "/api/reports/export", {
      title,
      sections,
    });
    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${title}.html`;
    a.click();
  };

  const previewHtml = async () => {
    const res = await apiRequest("POST", "/api/reports/export", {
      title,
      sections,
    });
    const html = await res.text();
    const win = window.open("", "_blank");
    if (win) {
      win.document.write(html);
      win.document.close();
    }
  };

  const handleGeneratedReport = (report: { title: string; sections: Section[] }) => {
    setTitle(report.title);
    setSections(report.sections);
    toast({
      title: "Report Generated",
      description: "Your report has been generated successfully. You can now preview, edit, or export it.",
    });
  };

  if (isLoading) {
    return <div>Loading...</div>;
  }

  if (!id && sections.length === 0) {
    return (
      <div className="container mx-auto p-6">
        <ReportGenerator onGenerate={handleGeneratedReport} />
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6">
      <div className="flex items-center justify-between mb-8">
        <h1 className="text-2xl font-bold">{title}</h1>
        <div className="flex gap-2">
          <Button
            variant="outline"
            onClick={previewHtml}
          >
            <ExternalLink className="h-4 w-4 mr-2" />
            Preview
          </Button>
          <Button
            variant="outline"
            onClick={exportHtml}
          >
            <Download className="h-4 w-4 mr-2" />
            Export HTML
          </Button>
          <Button
            onClick={() => saveMutation.mutate({ title, sections })}
            disabled={saveMutation.isPending}
          >
            <Save className="h-4 w-4 mr-2" />
            Save
          </Button>
        </div>
      </div>

      <div className="space-y-4">
        {sections.map((section) => (
          <ReportSection
            key={section.id}
            section={section}
            onChange={(content) => updateSection(section.id, content)}
            onDelete={() => deleteSection(section.id)}
          />
        ))}
      </div>
    </div>
  );
}