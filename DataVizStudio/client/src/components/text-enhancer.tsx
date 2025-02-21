import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Wand2 } from "lucide-react";
import { apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";

interface Props {
  text: string;
  onEnhanced: (text: string) => void;
}

export function TextEnhancer({ text, onEnhanced }: Props) {
  const [isEnhancing, setIsEnhancing] = useState(false);
  const { toast } = useToast();

  const enhance = async () => {
    try {
      setIsEnhancing(true);
      const res = await apiRequest("POST", "/api/enhance", { text });
      const data = await res.json();
      onEnhanced(data.enhanced);
      toast({
        title: "Success",
        description: "Text enhanced successfully",
      });
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to enhance text",
        variant: "destructive",
      });
    } finally {
      setIsEnhancing(false);
    }
  };

  return (
    <Button
      variant="outline"
      onClick={enhance}
      disabled={isEnhancing || !text}
    >
      <Wand2 className="h-4 w-4 mr-2" />
      {isEnhancing ? "Enhancing..." : "Enhance Text"}
    </Button>
  );
}
