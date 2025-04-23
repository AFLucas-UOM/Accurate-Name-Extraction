import { Check, ChevronDown, ChevronUp, Clock, Target, Info, ExternalLink, Key, Star, AlertTriangle } from "lucide-react";
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
    level: number; // 0-3 scale
  };
  apiRequired: boolean;
  pricing: string;
  websiteUrl: string;
  bestFor: string;
  warning?: string;
}

interface ModelSelectionStepProps {
  onModelSelected: (modelId: string) => void;
  selectedModel: string;
  className?: string;
}

  const models: Model[] = [
  {
    id: "anep",
    name: "Accurate Name Extraction Pipeline (ANEP)",
    description: "Advanced custom pipeline using YOLOv12, multi-method OCR with Tesseract, spaCy + GliNER, and transformer-based NER for robust person name extraction from news graphics with maximum precision.",
    speed: { label: "Slow", level: 1 },
    accuracy: { label: "Moderate", level: 2 },
    apiRequired: false,
    pricing: "Free, open-source components",
    websiteUrl: "https://github.com/tesseract-ocr/tesseract",
    bestFor: "High volume processing, offline usage, and complete customization"
  },
  {
    id: "model2",
    name: "Llama 4 Maverick",
    description: "Lightweight pipeline using Llama 4 Maverick for OCR and name extraction. Ideal for short-form news videos with simple frames and layouts, providing a balance between speed and accuracy.",
    speed: { label: "Fast", level: 2 },
    accuracy: { label: "High", level: 2 },
    apiRequired: true,
    pricing: "Pay-per-use, starting at $10 for 1000 API request calls per day",
    websiteUrl: "https://openrouter.ai/meta-llama/llama-4-maverick:free",
    bestFor: "Fast OCR and name extraction in short news clips and simple video layouts.",
    warning: "Limited free API calls available (1000 per day). Each 1 minute video clip takes around 70 requests."
  },
  {
    id: "model1",
    name: "Google Cloud Vision & Gemini 1.5 Pro",
    description: "Hybrid pipeline leveraging Google Cloud Vision API for OCR and Gemini 1.5 Pro for accurate name extraction. Efficient and highly accurate on short-form news videos with distinct frames and various layouts.",
    speed: { label: "Very Fast", level: 3 },
    accuracy: { label: "Excellent", level: 3 },
    apiRequired: true,
    pricing: "Pay-per-use, starting at $1.50 per 1000 API calls",
    websiteUrl: "https://cloud.google.com/pricing?hl=en",
    bestFor: "High accuracy requirements, professional productions, and complex layouts"
  },
  {
    id: "all",
    name: "Comparative Analysis",
    description: "Run all models and compare their performance side by side. This option processes your media through all available extraction methods, allowing you to compare results, accuracy, and processing time to determine the optimal solution.",
    speed: { label: "Very Slow", level: 0 },
    accuracy: { label: "Comparative", level: 3 },
    apiRequired: true,
    pricing: "Combines costs of all selected services",
    websiteUrl: "",
    bestFor: "Benchmark testing, research purposes, and finding the optimal model for specific use cases",
    warning: "This analysis will take considerably longer as all three models will be processed sequentially."
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
  const [activeTooltip, setActiveTooltip] = useState<string | null>(null);

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
                : "bg-gray-300 dark:bg-gray-700"
            }`} 
          />
        ))}
      </div>
    );
  };

  // Modified tooltip behavior with delay before closing
  const [tooltipTimeout, setTooltipTimeout] = useState<NodeJS.Timeout | null>(null);

  const showTooltip = (id: string) => {
    // Clear any existing timeout
    if (tooltipTimeout) {
      clearTimeout(tooltipTimeout);
      setTooltipTimeout(null);
    }
    // Set the active tooltip
    setActiveTooltip(id);
  };

  const hideTooltip = () => {
    // Set a timeout to hide the tooltip after a delay
    const timeout = setTimeout(() => {
      setActiveTooltip(null);
    }, 300); // 300ms delay
    
    setTooltipTimeout(timeout);
  };

  return (
    <div className={`w-full max-w-5xl mx-auto ${className}`}>
      <div className="mb-8">
        <h2 className="text-2xl font-bold mb-2">Select Analysis Model</h2>
        <p className="text-muted-foreground">
          Choose the model that best fits your video's graphic style
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-10">
        {models.map((model) => (
          <div
            key={model.id}
            className={`
              relative flex flex-col p-5 rounded-lg border-2 transition-all duration-300
              ${selectedModel === model.id 
                ? "border-[#2463EB] bg-[#f5faff] dark:bg-[#1a2d4a] shadow-lg" 
                : "border-gray-200 dark:border-gray-700 hover:border-[#2463EB] dark:hover:border-[#2463EB] hover:shadow-md hover:bg-[#f5faff] dark:hover:bg-[#1a2d4a]"}
            `}
          >
            {/* API Badge - Improved styling for dark mode */}
            {model.apiRequired && (
              <div className="absolute -top-3 -right-3 group">
                <div className="bg-amber-100 dark:bg-amber-800 text-amber-800 dark:text-amber-100 text-xs font-medium rounded-full border border-amber-300 dark:border-amber-700 flex items-center overflow-hidden whitespace-nowrap transition-all duration-300 ease-in-out" 
                  style={{
                    width: 'calc(var(--group-hover-width, 26px))',
                    paddingLeft: '5px',
                    paddingRight: 'var(--group-hover-padding, 5px)',
                    paddingTop: '4px',
                    paddingBottom: '4px',
                    '--group-hover-width': 'var(--is-hovered, 26px)',
                    '--group-hover-padding': 'var(--is-hovered-padding, 5px)',
                    '--is-hovered': 'var(--is-group-hovered, 26px)',
                    '--is-hovered-padding': 'var(--is-group-hovered-padding, 5px)',
                    '--is-group-hovered': 'var(--group-hovered, 26px)',
                    '--is-group-hovered-padding': 'var(--group-hovered-padding, 5px)',
                    '--group-hovered': 'var(--hover-effect, 26px)',
                    '--group-hovered-padding': 'var(--hover-effect-padding, 5px)',
                    '--hover-effect': 'var(--hover-trigger, 26px)',
                    '--hover-effect-padding': 'var(--hover-trigger-padding, 5px)',
                    '--hover-trigger': 'var(--hover, 26px)',
                    '--hover-trigger-padding': 'var(--hover-padding, 5px)',
                    '--hover': 'var(--hov, 26px)',
                    '--hover-padding': 'var(--hov-padding, 5px)',
                    '--hov': 'var(--h, 26px)',
                    '--hov-padding': 'var(--h-padding, 5px)',
                    '--h': 'var(--base, 26px)',
                    '--h-padding': 'var(--base-padding, 5px)',
                    '--base': 'var(--init, 26px)',
                    '--base-padding': 'var(--init-padding, 5px)',
                    '--init': 'var(--start, 26px)',
                    '--init-padding': 'var(--start-padding, 5px)',
                    '--start': 'var(--begin, 26px)',
                    '--start-padding': 'var(--begin-padding, 5px)',
                    '--begin': 'var(--origin, 26px)',
                    '--begin-padding': 'var(--origin-padding, 5px)',
                    '--origin': 'var(--source, 26px)',
                    '--origin-padding': 'var(--source-padding, 5px)',
                    '--source': 'var(--root, 26px)',
                    '--source-padding': 'var(--root-padding, 5px)',
                    '--root': 'var(--init-val, 26px)',
                    '--root-padding': 'var(--init-val-padding, 5px)',
                    '--init-val': 'var(--default-val, 26px)',
                    '--init-val-padding': 'var(--default-val-padding, 5px)',
                    '--default-val': 'var(--base-val, 26px)',
                    '--default-val-padding': 'var(--base-val-padding, 5px)',
                    '--base-val': 'var(--initial-val, 26px)',
                    '--base-val-padding': 'var(--initial-val-padding, 5px)',
                    '--initial-val': 'var(--start-val, 26px)',
                    '--initial-val-padding': 'var(--start-val-padding, 5px)',
                    '--start-val': 'var(--begin-val, 26px)', 
                    '--start-val-padding': 'var(--begin-val-padding, 5px)',
                    '--begin-val': 'var(--value, 26px)',
                    '--begin-val-padding': 'var(--value-padding, 5px)',
                    '--value': 'var(--val, 26px)', 
                    '--value-padding': 'var(--val-padding, 5px)',
                    '--val': 'var(--v, 26px)',
                    '--val-padding': 'var(--v-padding, 5px)',
                    '--v': '26px',
                    '--v-padding': '5px'
                  } as React.CSSProperties & { [key: string]: string }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.setProperty('--v', '110px');
                    e.currentTarget.style.setProperty('--v-padding', '10px');
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.setProperty('--v', '26px');
                    e.currentTarget.style.setProperty('--v-padding', '5px');
                  }}
                >
                  <div className="flex items-center justify-center w-4 h-4 min-w-[16px]">
                    <Key size={12} />
                  </div>
                  <span className="ml-1 whitespace-nowrap overflow-hidden transition-opacity duration-300 ease-in-out opacity-0 group-hover:opacity-100">API Required</span>
                </div>
              </div>
            )}
            
            {/* Make entire card clickable by wrapping all content in a button */}
            <div 
              className="flex flex-col h-full cursor-pointer"
              onClick={() => onModelSelected(model.id)}
            >
              <div className="flex justify-between items-start mb-2">
                <h3 className="text-lg font-medium">
                  {model.name}
                  {/* Add warning sign for both models with warnings */}
                  {model.warning && (
                    <span 
                      className="inline-block ml-2 relative"
                      onMouseEnter={() => showTooltip(`warning-${model.id}`)}
                      onMouseLeave={hideTooltip}
                      onClick={(e) => e.stopPropagation()} // Prevent triggering card click when clicking warning
                    >
                      <div className="text-amber-500 dark:text-amber-400">
                        <AlertTriangle size={16} />
                      </div>
                      
                      {/* Warning tooltip */}
                      {activeTooltip === `warning-${model.id}` && (
                        <div 
                          className="absolute top-full left-0 mt-2 p-3 bg-white dark:bg-gray-800 shadow-lg rounded-md border border-amber-200 dark:border-amber-600 w-64 z-10 text-xs"
                          onMouseEnter={() => showTooltip(`warning-${model.id}`)}
                          onMouseLeave={hideTooltip}
                        >
                          <p className="font-medium mb-1 text-amber-700 dark:text-amber-400">Warning</p>
                          <p className="text-amber-700 dark:text-amber-300">
                            {model.warning}
                          </p>
                        </div>
                      )}
                    </span>
                  )}
                </h3>
                {selectedModel === model.id && (
                  <div className="w-6 h-6 bg-primary rounded-full flex items-center justify-center">
                    <Check className="h-4 w-4 text-white" />
                  </div>
                )}
              </div>
              <p className="text-muted-foreground text-sm mb-3 text-justify">{model.description}</p>

              <div className="mt-auto pt-4 border-t border-gray-200 dark:border-gray-600">
                <div className="flex justify-between items-center">
                  <div 
                    className="relative flex items-center gap-1 text-xs text-gray-500 cursor-pointer"
                    onMouseEnter={() => showTooltip(`pricing-${model.id}`)}
                    onMouseLeave={hideTooltip}
                    onClick={(e) => e.stopPropagation()} // Prevent triggering card click when clicking pricing info
                  >
                    {/* Changed from dollar to euro sign */}
                    <div className="text-gray-400">€</div>
                    <span>Pricing Info</span>
                    <Info size={12} className="text-gray-400" />
                    
                    {/* Pricing tooltip - with solid background */}
                    {activeTooltip === `pricing-${model.id}` && (
                      <div 
                        className="absolute bottom-full left-0 mb-2 p-3 bg-white dark:bg-gray-800 shadow-lg rounded-md border border-gray-200 dark:border-gray-600 w-64 z-10 text-xs"
                        onMouseEnter={() => showTooltip(`pricing-${model.id}`)}
                        onMouseLeave={hideTooltip}
                      >
                        <p className="font-medium mb-1">Pricing</p>
                        <p>{model.pricing}</p>
                      </div>
                    )}
                  </div>
                  
                  {/* Only show "Learn more" for non-ANEP models */}
                  {model.websiteUrl && model.id !== "anep" && (
                    <a 
                      href={model.websiteUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-xs flex items-center gap-1 text-blue-500 hover:text-blue-700 group dark:text-blue-400 dark:hover:text-blue-300 transition-colors"
                      onClick={(e) => e.stopPropagation()} // Prevent triggering card click when clicking external link
                    >
                      <span className="hover:underline group-hover:underline">Learn more</span>
                      <ExternalLink size={12} className="group-hover:underline" />
                    </a>
                  )}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Comparison Table */}
      <div className={`bg-gray-50 dark:bg-[#172133] rounded-lg p-6 transition-all duration-300 border-2 ${isComparisonExpanded ? "border-[#2463EB]" : "border-gray-200 dark:border-transparent hover:border-[#2463EB] hover:bg-[#f5faff] hover:shadow-md dark:hover:border-[#2463EB] dark:hover:bg-[#172133]"}`}>
        <button 
          className="w-full flex justify-between items-center rounded-md transition-all duration-300 focus:outline-none focus:ring-0 focus:ring-transparent"
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
                  <th className="text-left py-3 px-4 font-medium w-1/4">Model</th>
                  <th className="text-left py-3 px-4 font-medium w-1/6">
                    <div className="flex items-center gap-2">
                      <Clock size={16} className="text-blue-500" />
                      <span>Speed</span>
                    </div>
                  </th>
                  <th className="text-left py-3 px-4 font-medium w-1/6">
                    <div className="flex items-center gap-2">
                      <Target size={16} className="text-green-500" />
                      <span>Accuracy</span>
                    </div>
                  </th>
                  <th className="text-left py-3 px-4 font-medium w-1/4">
                    <div className="flex items-center gap-2">
                      <Star size={16} className="text-yellow-500" />
                      <span>Best Suited For</span>
                    </div>
                  </th>
                  <th className="text-left py-3 px-4 font-medium w-1/6 whitespace-nowrap">
                    <div className="flex items-center gap-2">
                      <Key size={16} className="text-amber-500" />
                      <span>API Required</span>
                    </div>
                  </th>
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
                    <td className="py-3 px-4">
                      <p className="text-sm text-gray-600 dark:text-gray-300 text-left">
                        {model.bestFor}
                      </p>
                    </td>
                    <td className="py-3 px-4">
                      <div className="flex items-center justify-start gap-2">
                        <span className={`w-5 h-5 flex items-center justify-center rounded-full ${model.apiRequired ? "bg-amber-100 text-amber-800" : "bg-green-100 text-green-800"}`}>
                          {model.apiRequired ? <Key size={12} /> : <Check size={12} />}
                        </span>
                        <span>{model.apiRequired ? "Yes" : "No"}</span>
                      </div>
                    </td>
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