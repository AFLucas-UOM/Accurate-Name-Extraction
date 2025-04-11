import { Check, ChevronDown, ChevronUp, Clock, Target, Info } from "lucide-react";
import { useState, useRef, useEffect } from "react";

interface Model {
  id: string;
  name: string;
  description: string;
  speed: {
    label: string;
    level: number; // 0-3 scale
  };
  accuracy: {
    label: string;
    level: number; // 1-3 scale
  };
  bestFor: string;
}

interface ModelSelectionStepProps {
  onModelSelected: (modelId: string) => void;
  selectedModel: string;
  className?: string;
}

const models: Model[] = [
  {
    id: "anep",
    name: "ANEP",
    description: "Accurate Name Extraction Pipeline using custom OCR and NER method",
    speed: { label: "Fast", level: 3 },
    accuracy: { label: "High", level: 1 },
    bestFor: "Standard news ticker formats"
  },
  {
    id: "model1",
    name: "Google Vision OCR",
    description: "Google's Cloud-based OCR",
    speed: { label: "Medium", level: 2 },
    accuracy: { label: "Very High", level: 2 },
    bestFor: "Lower-thirds and overlays"
  },
  {
    id: "model2",
    name: "Llama 3.2 Vision",
    description: "Multimodal model fine-tuned for stylized or challenging text",
    speed: { label: "Slow", level: 1 },
    accuracy: { label: "Excellent", level: 3 },
    bestFor: "Stylised or difficult layouts"
  },
  {
    id: "all",
    name: "All Models",
    description: "Run all models and compare their performance",
    speed: { label: "Very Slow", level: 0 },
    accuracy: { label: "Comparative", level: 3 },
    bestFor: "Evaluation and benchmarking"
  }
];

const ModelSelectionStep = ({
  onModelSelected,
  selectedModel,
  className = "",
}: ModelSelectionStepProps) => {
  const [isComparisonExpanded, setIsComparisonExpanded] = useState(false);
  const comparisonRef = useRef<HTMLDivElement>(null);
  const [comparisonHeight, setComparisonHeight] = useState<number | undefined>(undefined);

  // Measure the height of the comparison table for smooth animation
  useEffect(() => {
    if (comparisonRef.current && isComparisonExpanded) {
      setComparisonHeight(comparisonRef.current.scrollHeight);
    }
  }, [isComparisonExpanded]);

  const toggleComparison = () => {
    setIsComparisonExpanded(!isComparisonExpanded);
  };

  // Helper function to render performance indicators
  const renderPerformanceLevel = (level: number, maxLevel: number = 3) => {
    return (
      <div className="flex gap-1">
        {Array.from({ length: maxLevel }).map((_, i) => (
          <div 
            key={i} 
            className={`h-2 w-6 rounded-full transition-colors duration-300 ${
              i < level 
                ? level === 3 ? "bg-green-500" : level === 2 ? "bg-blue-500" : "bg-yellow-500" 
                : "bg-gray-200 dark:bg-gray-700"
            }`} 
          />
        ))}
      </div>
    );
  };

  return (
    <div className={`w-full max-w-4xl mx-auto ${className}`}>
      <div className="mb-8">
        <h2 className="text-2xl font-bold mb-2">Select Analysis Model</h2>
        <p className="text-muted-foreground">
          Choose the model that best fits your video's graphic style
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-10">
        {models.map((model) => (
          <button
            key={model.id}
            className={`
              flex flex-col p-5 rounded-lg border-2 transition-all duration-300 text-left
              focus:outline-none focus:ring-0 focus:ring-transparent
              ${selectedModel === model.id 
                ? "border-[#2463EB] bg-[#f5faff] dark:bg-[#1a2d4a] shadow-md" 
                : "border-gray-200 dark:border-gray-700 hover:border-[#2463EB] dark:hover:border-[#2463EB] hover:shadow-md hover:bg-[#f5faff] dark:hover:bg-[#1a2d4a]"}
            `}            
            onClick={() => onModelSelected(model.id)}
            aria-pressed={selectedModel === model.id}
          >
            <div className="flex justify-between items-start mb-2">
              <h3 className="text-lg font-medium">{model.name}</h3>
              {selectedModel === model.id && (
                <div className="w-6 h-6 bg-primary rounded-full flex items-center justify-center">
                  <Check className="h-4 w-4 text-white" />
                </div>
              )}
            </div>
            <p className="text-muted-foreground text-sm">{model.description}</p>
            {model.id === "all" && (
              <div className="mt-3 text-xs flex items-center text-amber-600 dark:text-amber-400">
                <Info size={14} className="mr-1" />
                <span>Processing may take slightly longer</span>
              </div>
            )}
          </button>
        ))}
      </div>

      <div className={`bg-gray-50 dark:bg-[#172133] rounded-lg p-6 transition-all duration-300 border-2 ${isComparisonExpanded ? "border-transparent" : "border-transparent hover:border-[#2463EB] hover:bg-[#f5faff] hover:shadow-md dark:hover:border-[#2463EB] dark:hover:bg-[#172133]"}`}>
        <button 
          className="w-full flex justify-between items-center rounded-md transition-all duration-300 focus:outline-none focus:ring-0 focus:ring-transparenthover:outline hover:outline-2 hover:outline-[#2463EB]"
          onClick={toggleComparison}
          aria-expanded={isComparisonExpanded}
        >
          <h3 className="text-lg font-medium">Model Comparison</h3>
          <div className="text-gray-500 bg-gray-100 dark:bg-gray-800 p-1 rounded-full transition-transform duration-300">
            {isComparisonExpanded ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
          </div>
        </button>
        
        <div 
          ref={comparisonRef}
          className="overflow-hidden transition-all duration-500 ease-in-out"
          style={{ 
            maxHeight: isComparisonExpanded ? `${comparisonHeight}px` : "0px",
            opacity: isComparisonExpanded ? 1 : 0,
            marginTop: isComparisonExpanded ? "1rem" : "0"
          }}
        >
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left py-3 px-4 font-medium">Model</th>
                  <th className="text-left py-3 px-4 font-medium">
                    <div className="flex items-center gap-2">
                      <Clock size={16} className="text-blue-500" />
                      <span>Speed</span>
                    </div>
                  </th>
                  <th className="text-left py-3 px-4 font-medium">
                    <div className="flex items-center gap-2">
                      <Target size={16} className="text-green-500" />
                      <span>Accuracy</span>
                    </div>
                  </th>
                  <th className="text-left py-3 px-4 font-medium">Best Suited For</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                {models.map((model) => (
                  <tr 
                    key={model.id}
                    className={`transition-colors duration-300 ${
                      selectedModel === model.id 
                        ? "bg-primary/10" 
                        : "hover:bg-gray-100 dark:hover:bg-gray-800"
                    }`}
                  >
                    <td className="py-3 px-4 font-medium">{model.name}</td>
                    <td className="py-3 px-4">
                      <div className="flex flex-col gap-1">
                        <span>{model.speed.label}</span>
                        {renderPerformanceLevel(model.speed.level, 3)}
                      </div>
                    </td>
                    <td className="py-3 px-4">
                      <div className="flex flex-col gap-1">
                        <span>{model.accuracy.label}</span>
                        {renderPerformanceLevel(model.accuracy.level, 3)}
                      </div>
                    </td>
                    <td className="py-3 px-4">{model.bestFor}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelSelectionStep;