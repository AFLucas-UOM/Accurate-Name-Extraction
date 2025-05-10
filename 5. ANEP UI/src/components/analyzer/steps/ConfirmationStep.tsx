import { useState, useEffect, useRef } from "react";
import {
  FileVideo,
  Clock,
  Cpu,
  BarChart,
  Info,
  CheckCircle2,
  HardDrive,
  AlertTriangle
} from "lucide-react";
import { Tooltip, TooltipTrigger, TooltipContent } from "@/components/ui/tooltip";

interface ConfirmationStepProps {
  videoFile: File;
  selectedModel: string;
  className?: string;
}

// Processing speeds - realistic estimates based on actual video processing
const MODEL_PROCESSING_SPEEDS = {
  anep: {
    name: "Accurate Name Extraction Pipeline (ANEP)",
    baseTime: 2, // Base overhead
    speedPerMinute: 0.4, // ~10-15 minutes for typical videos
    description: "Advanced custom pipeline using YOLOv12, multi-method OCR with Tesseract, spaCy + GliNER, and transformer-based NER"
  },
  model1: {
    name: "Google Cloud Vision & Gemini 1.5 Pro",
    baseTime: 1,
    speedPerMinute: 0.15, // ~3-5 minutes for typical videos
    description: "Hybrid pipeline leveraging Google Cloud Vision API for OCR and Gemini 1.5 Pro for accurate name extraction"
  },
  model2: {
    name: "Llama 4 Maverick",
    baseTime: 1.5,
    speedPerMinute: 0.25, // ~5-10 minutes for typical videos
    description: "Lightweight pipeline using Llama 4 Maverick for OCR and name extraction. Ideal for short-form news videos with simple frames and layouts, providing a balance between speed and accuracy."
  },
  all: {
    name: "Comparative Analysis",
    baseTime: 4,
    speedPerMinute: 0.8, // ~20-25 minutes for typical videos
    description: "Run all models and compare their performance side by side"
  }
};

const ConfirmationStep = ({
  videoFile,
  selectedModel,
  className = "",
}: ConfirmationStepProps) => {
  const [infoTooltipOpen, setInfoTooltipOpen] = useState(false);
  const [metaTooltipOpen, setMetaTooltipOpen] = useState(false);
  const [fileSizeFormatted, setFileSizeFormatted] = useState("");
  const [videoDurationSeconds, setVideoDurationSeconds] = useState<number | null>(null);
  const [videoDurationFormatted, setVideoDurationFormatted] = useState<string | null>(null);
  
  const infoTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const metaTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    // Format file size
    const fileSizeInMB = videoFile.size / (1024 * 1024);
    setFileSizeFormatted(
      fileSizeInMB >= 1
        ? `${fileSizeInMB.toFixed(2)} MB`
        : `${(videoFile.size / 1024).toFixed(2)} KB`
    );

    // Get video duration
    const videoElement = document.createElement("video");
    videoElement.preload = "metadata";
    videoElement.onloadedmetadata = () => {
      const duration = videoElement.duration;
      if (duration && !isNaN(duration)) {
        setVideoDurationSeconds(duration);
        const minutes = Math.floor(duration / 60);
        const seconds = Math.floor(duration % 60);
        setVideoDurationFormatted(`${minutes}:${seconds.toString().padStart(2, "0")}`);
      }
    };
    videoElement.src = URL.createObjectURL(videoFile);

    return () => {
      URL.revokeObjectURL(videoElement.src);
    };
  }, [videoFile]);

  // Calculate estimated processing time
  const calculateProcessingTime = (durationInSeconds: number | null) => {
    if (!durationInSeconds) return "~ Calculating...";
    
    const modelSpeed = MODEL_PROCESSING_SPEEDS[selectedModel as keyof typeof MODEL_PROCESSING_SPEEDS];
    if (!modelSpeed) return "~ Unknown";
    
    const videoDurationMinutes = durationInSeconds / 60;
    const totalMinutes = modelSpeed.baseTime + (videoDurationMinutes * modelSpeed.speedPerMinute);
    
    // Create a range (±15% variance)
    const minMinutes = Math.max(1, Math.floor(totalMinutes * 0.85));
    const maxMinutes = Math.ceil(totalMinutes * 1.15);
    
    // Format the time range
    if (minMinutes === maxMinutes) {
      return `~ ${minMinutes} minute${minMinutes !== 1 ? 's' : ''}`;
    } else if (maxMinutes < 60) {
      return `~ ${minMinutes}–${maxMinutes} minutes`;
    } else {
      // Convert to hours for longer estimates
      const minHours = Math.floor(minMinutes / 60);
      const minRemainMinutes = minMinutes % 60;
      const maxHours = Math.floor(maxMinutes / 60);
      const maxRemainMinutes = maxMinutes % 60;
      
      if (minHours === maxHours) {
        return `~ ${minHours}h ${minRemainMinutes}–${maxRemainMinutes}m`;
      } else {
        return `~ ${minHours}h ${minRemainMinutes}m–${maxHours}h ${maxRemainMinutes}m`;
      }
    }
  };

  const modelInfo = MODEL_PROCESSING_SPEEDS[selectedModel as keyof typeof MODEL_PROCESSING_SPEEDS] || {
    name: "Unknown Model",
    baseTime: 0,
    speedPerMinute: 0,
    description: "Unknown model description"
  };

  // Consolidated tooltip handlers
  const handleTooltipToggle = (
    setOpen: React.Dispatch<React.SetStateAction<boolean>>,
    timeoutRef: React.MutableRefObject<NodeJS.Timeout | null>
  ) => {
    setOpen(prev => !prev);
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    timeoutRef.current = setTimeout(() => setOpen(false), 5000);
  };

  const handleTooltipEnter = (
    setOpen: React.Dispatch<React.SetStateAction<boolean>>,
    timeoutRef: React.MutableRefObject<NodeJS.Timeout | null>
  ) => {
    setOpen(true);
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
  };

  const handleTooltipLeave = (
    timeoutRef: React.MutableRefObject<NodeJS.Timeout | null>,
    setOpen: React.Dispatch<React.SetStateAction<boolean>>
  ) => {
    if (!timeoutRef.current) setOpen(false);
  };

  // Cleanup timeouts on unmount
  useEffect(() => {
    return () => {
      if (infoTimeoutRef.current) clearTimeout(infoTimeoutRef.current);
      if (metaTimeoutRef.current) clearTimeout(metaTimeoutRef.current);
    };
  }, []);

  // Helper function for icon containers
  const IconContainer = ({ 
    children, 
    bgColor, 
    iconColor 
  }: { 
    children: React.ReactNode; 
    bgColor: string; 
    iconColor: string;
  }) => (
    <div className={`w-12 h-12 ${bgColor} rounded-lg flex items-center justify-center mr-4 mt-1 transition-all duration-300 hover:scale-105`}>
      <div className={iconColor}>
        {children}
      </div>
    </div>
  );

  const estimatedTime = calculateProcessingTime(videoDurationSeconds);

  return (
    <div className={`w-full max-w-5xl mx-auto ${className}`}>
      <h2 className="text-2xl font-bold mb-2">Confirm Analysis Details</h2>
      <p className="text-muted-foreground mb-6">
        Review your selection before starting the analysis
      </p>

      <div className="bg-gray-50 dark:bg-[#172133] rounded-lg p-6 border-2 border-gray-200 dark:border-transparent shadow-sm transition-all duration-300 hover:shadow-md hover:border-[#2463EB] dark:hover:border-[#2463EB] hover:bg-[#f5faff] dark:hover:bg-[#172133]">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-medium">Analysis Summary</h3>
          <div className="flex items-center text-sm text-emerald-600 dark:text-emerald-400">
            <CheckCircle2 className="w-4 h-4 mr-1" />
            <span>Ready to process</span>
          </div>
        </div>

        <div className="space-y-6">
          {/* Video File Information */}
          <div className="flex items-start">
            <IconContainer
              bgColor="bg-blue-100 dark:bg-blue-600/20"
              iconColor="text-blue-700 dark:text-blue-200"
            >
              <FileVideo className="h-6 w-6" />
            </IconContainer>
            <div className="flex-1">
              <div className="flex items-center gap-1 mb-1">
                <h4 className="font-medium">Video File</h4>
                <Tooltip open={metaTooltipOpen} onOpenChange={setMetaTooltipOpen}>
                  <TooltipTrigger
                    className="focus:outline-none rounded-full"
                    onClick={() => handleTooltipToggle(setMetaTooltipOpen, metaTimeoutRef)}
                    onPointerEnter={() => handleTooltipEnter(setMetaTooltipOpen, metaTimeoutRef)}
                    onPointerLeave={() => handleTooltipLeave(metaTimeoutRef, setMetaTooltipOpen)}
                  >
                    <Info className="w-4 h-4 text-muted-foreground cursor-pointer" />
                  </TooltipTrigger>
                  <TooltipContent side="right" className="w-52 p-3 text-sm space-y-2 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600">
                    <p className="font-medium mb-2">Video Statistics</p>
                    <div className="flex items-center justify-between text-muted-foreground">
                      <span className="flex items-center gap-1">
                        <HardDrive className="w-4 h-4" />
                        Size:
                      </span>
                      <span className="font-medium text-foreground">{fileSizeFormatted}</span>
                    </div>
                    {videoDurationFormatted && (
                      <div className="flex items-center justify-between text-muted-foreground">
                        <span className="flex items-center gap-1">
                          <Clock className="w-4 h-4" />
                          Duration:
                        </span>
                        <span className="font-medium text-foreground">{videoDurationFormatted}</span>
                      </div>
                    )}
                    <div className="flex items-center justify-between text-muted-foreground">
                      <span className="flex items-center gap-1">
                        <FileVideo className="w-4 h-4" />
                        Format:
                      </span>
                      <span className="font-medium text-foreground">
                        {videoFile.type || "Unknown"}
                      </span>
                    </div>
                  </TooltipContent>
                </Tooltip>
              </div>
              <p className="text-sm truncate max-w-md" title={videoFile.name}>
                {videoFile.name}
              </p>
            </div>
          </div>

          {/* Model Information */}
          <div className="flex items-start">
            <IconContainer
              bgColor="bg-purple-100 dark:bg-purple-600/20"
              iconColor="text-purple-700 dark:text-purple-200"
            >
              <Cpu className="h-6 w-6" />
            </IconContainer>
            <div className="flex-1">
              <h4 className="font-medium mb-1">Selected Model</h4>
              <p className="text-sm font-medium">{modelInfo.name}</p>
            </div>
          </div>

          {/* Estimated Time */}
          <div className="flex items-start">
            <IconContainer
              bgColor="bg-amber-100 dark:bg-amber-600/20"
              iconColor="text-amber-700 dark:text-amber-200"
            >
              <Clock className="h-6 w-6" />
            </IconContainer>
            <div className="flex-1">
              <div className="flex items-center gap-1 mb-1">
                <h4 className="font-medium">Estimated Time</h4>
                <Tooltip open={infoTooltipOpen} onOpenChange={setInfoTooltipOpen}>
                  <TooltipTrigger
                    className="focus:outline-none rounded-full"
                    onClick={() => handleTooltipToggle(setInfoTooltipOpen, infoTimeoutRef)}
                    onPointerEnter={() => handleTooltipEnter(setInfoTooltipOpen, infoTimeoutRef)}
                    onPointerLeave={() => handleTooltipLeave(infoTimeoutRef, setInfoTooltipOpen)}
                  >
                    <Info className="w-4 h-4 text-muted-foreground cursor-pointer" />
                  </TooltipTrigger>
                  <TooltipContent side="right" className="max-w-xs p-3 text-justify bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600">
                    <p className="font-medium mb-1">Processing Time Information</p>
                    <p className="text-xs text-muted-foreground">
                      Time estimates are calculated based on video duration and selected model. Actual processing time may vary depending on video complexity, resolution, and system resources.
                    </p>
                  </TooltipContent>
                </Tooltip>
              </div>
              <div className="flex items-center">
                <p className="font-medium text-amber-600 dark:text-amber-300">{estimatedTime}</p>
                {selectedModel === "all" && (
                  <div className="ml-2">
                    <span className="text-xs bg-amber-200 dark:bg-amber-800 text-amber-800 dark:text-amber-100 px-2 py-0.5 rounded-full">
                      Longest option
                    </span>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Expected Results */}
          <div className="flex items-start">
            <IconContainer
              bgColor="bg-green-100 dark:bg-green-600/20"
              iconColor="text-green-700 dark:text-green-200"
            >
              <BarChart className="h-6 w-6" />
            </IconContainer>
            <div className="flex-1">
              <h4 className="font-medium mb-1">Expected Results</h4>
              <div className="space-y-1 text-sm">
                <p>• Names detected in video graphics with timestamps</p>
                <p>• Confidence scores for each detection</p>
                {selectedModel === "all" && (
                  <p>• Comparative performance metrics across all models</p>
                )}
                {selectedModel === "model1" && (
                  <p>• High accuracy results with Google Vision API</p>
                )}
                {selectedModel === "anep" && (
                  <p>• Offline processing with complete data ownership</p>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ConfirmationStep;