import { Section } from "@shared/schema";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Table } from "@/components/ui/table";
import { Trash2, Upload } from "lucide-react";
import { TextEnhancer } from "./text-enhancer";
import { useState } from "react";

interface Props {
  section: Section;
  onChange: (content: any) => void;
  onDelete: () => void;
}

export function ReportSection({ section, onChange, onDelete }: Props) {
  const [imageFile, setImageFile] = useState<File | null>(null);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        onChange(reader.result as string);
      };
      reader.readAsDataURL(file);
      setImageFile(file);
    }
  };

  const renderContent = () => {
    switch (section.type) {
      case "text":
        return (
          <div className="space-y-4">
            <Textarea
              value={section.content}
              onChange={(e) => onChange(e.target.value)}
              className="min-h-[200px]"
              placeholder="Enter your analysis or data description here..."
            />
            <TextEnhancer
              text={section.content}
              onEnhanced={onChange}
            />
          </div>
        );
      case "image":
        return (
          <div className="space-y-4">
            <div className="flex items-center gap-4">
              <Input
                type="file"
                accept="image/*"
                onChange={handleImageUpload}
                className="hidden"
                id={`image-upload-${section.id}`}
              />
              <label 
                htmlFor={`image-upload-${section.id}`}
                className="flex items-center gap-2 px-4 py-2 border rounded-md cursor-pointer hover:bg-gray-50"
              >
                <Upload className="h-4 w-4" />
                {imageFile ? imageFile.name : "Upload Image"}
              </label>
              {section.content && (
                <Button
                  variant="outline"
                  onClick={() => {
                    onChange("");
                    setImageFile(null);
                  }}
                >
                  Remove Image
                </Button>
              )}
            </div>
            {section.content && (
              <div className="border rounded-lg p-2 bg-gray-50">
                <img
                  src={section.content}
                  alt="Section content"
                  className="max-w-full h-auto rounded-lg"
                />
              </div>
            )}
          </div>
        );
      case "table":
        const tableData = section.content || { headers: [], rows: [] };
        return (
          <div className="space-y-4">
            <Table>
              <thead>
                <tr>
                  {tableData.headers.map((header: string, i: number) => (
                    <th key={i}>
                      <Input
                        value={header}
                        onChange={(e) => {
                          const newHeaders = [...tableData.headers];
                          newHeaders[i] = e.target.value;
                          onChange({ ...tableData, headers: newHeaders });
                        }}
                        placeholder={`Column ${i + 1}`}
                      />
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {tableData.rows.map((row: string[], rowIndex: number) => (
                  <tr key={rowIndex}>
                    {row.map((cell: string, cellIndex: number) => (
                      <td key={cellIndex}>
                        <Input
                          value={cell}
                          onChange={(e) => {
                            const newRows = [...tableData.rows];
                            newRows[rowIndex][cellIndex] = e.target.value;
                            onChange({ ...tableData, rows: newRows });
                          }}
                          placeholder="Enter value"
                        />
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </Table>
            <div className="flex gap-2">
              <Button
                variant="outline"
                onClick={() => {
                  const newHeaders = [...tableData.headers, ""];
                  const newRows = tableData.rows.map(row => [...row, ""]);
                  onChange({ headers: newHeaders, rows: newRows });
                }}
              >
                Add Column
              </Button>
              <Button
                variant="outline"
                onClick={() => {
                  const newRow = new Array(tableData.headers.length).fill("");
                  onChange({
                    ...tableData,
                    rows: [...tableData.rows, newRow],
                  });
                }}
              >
                Add Row
              </Button>
            </div>
          </div>
        );
    }
  };

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <h3 className="capitalize">{section.type} Section</h3>
        <Button variant="ghost" size="icon" onClick={onDelete}>
          <Trash2 className="h-4 w-4" />
        </Button>
      </CardHeader>
      <CardContent>{renderContent()}</CardContent>
    </Card>
  );
}