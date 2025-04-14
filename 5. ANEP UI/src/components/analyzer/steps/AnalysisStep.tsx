import { useState, useEffect, useRef, useCallback } from "react";
import { 
  BarChart3, 
  AlertTriangle, 
  Clock, 
  CheckCircle, 
  XCircle, 
  Info, 
  FileVideo, 
  Cpu, 
  FastForward, 
  Pause, 
  Loader2, 
  Download,
  ChevronDown,
  ChevronUp,
  Clipboard,
  RefreshCw,
  Zap,
  Eye,
  TerminalSquare,
  BookOpen,
  Crop,
  Wrench,
  ExternalLink,
  Server,
  Film,
  FileDigit,
  Images,
  ImagePlus,
  ScanFace,
  Boxes,
  Type,
  FileText,
  Check,
  Layers,
  Brain,
  BarChart,
  Layout,
  PackageOpen,
  Eraser,
  ClipboardCheck
} from "lucide-react";

interface AnalysisStepProps {
  videoFile: File;
  selectedModel: string;
  onAnalysisComplete: (results: any) => void;
  className?: string;
  onSpeedUp?: () => void;
}

// Enhanced analysis result interface
interface NameDetection {
  name: string;
  confidence: number;
  timestamp: string;
  frames: number[];
  duration?: string;
}

interface AnalysisResults {
  names: NameDetection[];
  model: string;
  modelName: string;
  processingTime: string;
  videoMetadata: {
    filename: string;
    size: string;
    duration: string;
    resolution: string;
    type: string;
    frameRate?: string;
  };
  analysisDate: string;
  detectionSettings?: {
    yoloConfidence: number;
    maxOverlap: number;
  };
}

interface StageDefinition {
  id: string;
  message: string;
  icon: React.ReactNode;
  category: string;
  logs: string[];
}

// Status badge component for better visual indication
const StatusBadge = ({ stage }: { stage: string | StageDefinition }) => {
  const getCategoryColor = (category: string) => {
    switch (category) {
      case "setup": 
        return "bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300";
      case "processing": 
        return "bg-indigo-100 text-indigo-800 dark:bg-indigo-900/30 dark:text-indigo-300";
      case "detection": 
        return "bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300";
      case "extraction": 
        return "bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300";
      case "analysis": 
        return "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-300";
      case "finalization": 
        return "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300";
      default:
        return "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300";
    }
  };

  const getStageIcon = (stageText: string) => {
    if (stageText.includes("Initializing") || stageText.includes("Loading")) 
      return <FileVideo className="h-3 w-3 mr-1" />;
    if (stageText.includes("Detecting") || stageText.includes("Recognizing")) 
      return <Eye className="h-3 w-3 mr-1" />;
    if (stageText.includes("Processing") || stageText.includes("Preparing")) 
      return <Cpu className="h-3 w-3 mr-1" />;
    if (stageText.includes("Finalizing")) 
      return <CheckCircle className="h-3 w-3 mr-1" />;
    return <Loader2 className="h-3 w-3 mr-1 animate-spin" />;
  };

  const getStageColor = (stageText: string) => {
    if (stageText.includes("Initializing") || stageText.includes("Loading")) 
      return "bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300";
    if (stageText.includes("Detecting") || stageText.includes("Recognizing")) 
      return "bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300";
    if (stageText.includes("Processing") || stageText.includes("Preparing")) 
      return "bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300";
    if (stageText.includes("Finalizing")) 
      return "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300";
    return "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300";
  };

  if (typeof stage === 'string') {
    return (
      <span className={`px-3 py-1 rounded-full text-sm font-medium flex items-center ${getStageColor(stage)}`}>
        {getStageIcon(stage)}
        {stage}
      </span>
    );
  }

  return (
    <span className={`px-3 py-1 rounded-full text-sm font-medium flex items-center ${getCategoryColor(stage.category)}`}>
      {stage.icon}
      {stage.message}
    </span>
  );
};

// Log entry component for detailed process visualization
interface LogEntryProps {
  message: string;
  timestamp: string;
  type: 'info' | 'warning' | 'success' | 'error';
}

const LogEntry = ({ message, timestamp, type }: LogEntryProps) => {
  const getTypeIcon = () => {
    switch (type) {
      case 'info': return <Info className="h-4 w-4 text-blue-500" />;
      case 'warning': return <AlertTriangle className="h-4 w-4 text-amber-500" />;
      case 'success': return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'error': return <XCircle className="h-4 w-4 text-red-500" />;
    }
  };

  const getTypeStyle = () => {
    switch (type) {
      case 'info': return "border-blue-200 bg-blue-50 dark:border-blue-900 dark:bg-blue-900/10";
      case 'warning': return "border-amber-200 bg-amber-50 dark:border-amber-900 dark:bg-amber-900/10";
      case 'success': return "border-green-200 bg-green-50 dark:border-green-900 dark:bg-green-900/10";
      case 'error': return "border-red-200 bg-red-50 dark:border-red-900 dark:bg-red-900/10";
    }
  };

  return (
    <div className={`border-l-4 px-3 py-2 mb-1 ${getTypeStyle()} transition-all hover:translate-x-1`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {getTypeIcon()}
          <span className="text-sm">{message}</span>
        </div>
        <span className="text-xs text-gray-500 font-mono">{timestamp}</span>
      </div>
    </div>
  );
};

// Collapsible section component
interface CollapsibleSectionProps {
  title: string;
  icon: React.ReactNode;
  children: React.ReactNode;
  defaultOpen?: boolean;
}

const CollapsibleSection = ({ title, icon, children, defaultOpen = false }: CollapsibleSectionProps) => {
  const [isOpen, setIsOpen] = useState(defaultOpen);
  
  return (
    <div className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden mb-4">
      <button 
        onClick={() => setIsOpen(!isOpen)} 
        className="w-full flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-800 hover:bg-gray-100 dark:hover:bg-gray-750 transition-colors"
      >
        <div className="flex items-center gap-2">
          {icon}
          <span className="font-medium">{title}</span>
        </div>
        {isOpen ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
      </button>
      {isOpen && (
          <div className="p-4 bg-white dark:bg-[#162032]">
            {children}
          </div>
        )}
    </div>
  );
};

// Define stages with detailed messaging, icons, and categories
const stages: StageDefinition[] = [
  {
    id: "init",
    message: "Initializing analysis environment...",
    icon: <Server className="h-3 w-3 mr-1" />,
    category: "setup",
    logs: ["Allocating processing resources", "Loading analysis models into memory", "Preparing worker threads"]
  },
  {
    id: "decode",
    message: "Loading and decoding video content...",
    icon: <Film className="h-3 w-3 mr-1" />,
    category: "setup",
    logs: ["Reading video stream", "Verifying video integrity", "Buffering initial frames"]
  },
  {
    id: "metadata",
    message: "Extracting video metadata...",
    icon: <FileDigit className="h-3 w-3 mr-1" />,
    category: "setup",
    logs: ["Retrieving codec information", "Calculating frame rate and duration", "Reading embedded metadata"]
  },
  {
    id: "keyframes",
    message: "Generating keyframes from source...",
    icon: <Images className="h-3 w-3 mr-1" />,
    category: "processing",
    logs: ["Detecting scene changes", "Sampling frames at regular intervals", "Filtering redundant frames"]
  },
  {
    id: "preprocess",
    message: "Preprocessing frames for optimal detection...",
    icon: <ImagePlus className="h-3 w-3 mr-1" />,
    category: "processing",
    logs: ["Normalizing image contrast", "Applying noise reduction", "Converting colorspace for analysis"]
  },
  {
    id: "roi",
    message: "Locating regions of interest (ROIs)...",
    icon: <ScanFace className="h-3 w-3 mr-1" />,
    category: "detection",
    logs: ["Identifying lower third graphics", "Detecting caption areas", "Marking potential name regions"]
  },
  {
    id: "yolo",
    message: "Detecting text regions using YOLOv12...",
    icon: <Boxes className="h-3 w-3 mr-1" />,
    category: "detection",
    logs: ["Running neural object detection", "Analyzing confidence scores", "Filtering overlapping boxes"]
  },
  {
    id: "ocr",
    message: "Applying OCR to cropped ROIs...",
    icon: <Type className="h-3 w-3 mr-1" />,
    category: "extraction",
    logs: ["Converting regions to grayscale", "Running Tesseract OCR engine", "Extracting raw text data"]
  },
  {
    id: "postprocess",
    message: "Post-processing OCR output...",
    icon: <FileText className="h-3 w-3 mr-1" />,
    category: "extraction",
    logs: ["Correcting common OCR errors", "Filtering noise and artifacts", "Joining fragmented characters"]
  },
  {
    id: "validate",
    message: "Validating and normalizing extracted names...",
    icon: <Check className="h-3 w-3 mr-1" />,
    category: "analysis",
    logs: ["Checking for name patterns", "Normalizing capitalization", "Validating against name database"]
  },
  {
    id: "dedupe",
    message: "Running name disambiguation and de-duplication...",
    icon: <Layers className="h-3 w-3 mr-1" />,
    category: "analysis",
    logs: ["Grouping similar names", "Resolving partial matches", "Removing duplicate entries"]
  },
  {
    id: "nlp",
    message: "Applying NLP models to enhance recognition...",
    icon: <Brain className="h-3 w-3 mr-1" />,
    category: "analysis",
    logs: ["Analyzing name contexts", "Applying entity recognition", "Processing with BERT transformer"]
  },
  {
    id: "confidence",
    message: "Assigning confidence scores to entities...",
    icon: <BarChart className="h-3 w-3 mr-1" />,
    category: "finalization",
    logs: ["Calculating detection reliability", "Weighting multiple appearances", "Assigning confidence metrics"]
  },
  {
    id: "overlays",
    message: "Finalising visual overlays and report data...",
    icon: <Layout className="h-3 w-3 mr-1" />,
    category: "finalization",
    logs: ["Generating timestamp markers", "Creating bounding boxes", "Preparing visualization data"]
  },
  {
    id: "compile",
    message: "Compiling final result objects for export...",
    icon: <PackageOpen className="h-3 w-3 mr-1" />,
    category: "finalization",
    logs: ["Structuring JSON output", "Formatting result fields", "Preparing data for storage"]
  },
  {
    id: "cleanup",
    message: "Cleaning up temporary buffers and cache...",
    icon: <Eraser className="h-3 w-3 mr-1" />,
    category: "finalization",
    logs: ["Freeing allocated memory", "Closing file handles", "Removing temporary files"]
  },
  {
    id: "finalize",
    message: "Finalizing results and preparing report...",
    icon: <ClipboardCheck className="h-3 w-3 mr-1" />,
    category: "finalization",
    logs: ["Validating final outputs", "Generating summaries", "Packaging complete results"]
  }
];

const AnalysisStep = ({
  videoFile,
  selectedModel,
  onAnalysisComplete,
  className = "",
  onSpeedUp,
}: AnalysisStepProps) => {
  // State variables for progress, stage messaging, errors, timer and analysis state
  const [progress, setProgress] = useState(0);
  const [currentStage, setCurrentStage] = useState<string | StageDefinition>("Initializing...");
  const [error, setError] = useState<string | null>(null);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [estimatedTime, setEstimatedTime] = useState<number | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isCanceled, setIsCanceled] = useState(false);
  const [isCompleted, setIsCompleted] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [processingSpeed, setProcessingSpeed] = useState(1); // 1x, 2x, 3x speed
  const [showLogs, setShowLogs] = useState(false);
  const [logs, setLogs] = useState<LogEntryProps[]>([]);
  const [detectedNames, setDetectedNames] = useState<NameDetection[]>([]);
  // Removing technical details state and related states
  const [analysisOptions, setAnalysisOptions] = useState({
    yoloConfidence: 0.65,
    maxOverlap: 0.5,
  });

  const rerunRef = useRef(false);
  const cancelRef = useRef(false);
  const pauseRef = useRef(false);
  const speedRef = useRef(1);

  // Map model id to its display name with clearer documentation
  const getModelName = (modelId: string): string => {
    const modelMap: Record<string, string> = {
      "anep": "ANEP (Accurate Name Extraction Pipeline)",
      "model1": "Google Vision OCR",
      "model2": "Llama 3.2 Vision",
      "all": "All Models (Ensemble)",
    };
    
    return modelMap[modelId] || "Custom Model";
  };

  // Format elapsed seconds as mm:ss with improved formatting
  const formatElapsed = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  };

  // Add log entry
  const addLog = (message: string, type: 'info' | 'warning' | 'success' | 'error' = 'info') => {
    const now = new Date();
    const timestamp = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}`;
    
    setLogs(prevLogs => [
      { message, timestamp, type },
      ...prevLogs.slice(0, 99) // Keep last 100 logs
    ]);
  };

  // Function to handle speed change
  const handleSpeedChange = () => {
    const newSpeed = processingSpeed < 3 ? processingSpeed + 1 : 1;
    setProcessingSpeed(newSpeed);
    speedRef.current = newSpeed;
    addLog(`Processing speed changed to ${newSpeed}x`, 'info');
    
    if (onSpeedUp) {
      onSpeedUp();
    }
  };

  // Function to toggle pause
  const handlePauseToggle = () => {
    setIsPaused(!isPaused);
    pauseRef.current = !pauseRef.current;
    addLog(pauseRef.current ? "Analysis paused" : "Analysis resumed", pauseRef.current ? 'warning' : 'info');
  };

  // Function to copy logs
  const copyLogs = () => {
    const logText = logs.map(log => `[${log.timestamp}] ${log.type.toUpperCase()}: ${log.message}`).join('\n');
    navigator.clipboard.writeText(logText);
    addLog("Logs copied to clipboard", 'success');
  };

  // Helper function: Cancels the current run and waits until it's finished
  const cancelCurrentRun = async () => {
    cancelRef.current = true;
    addLog("Cancelling current analysis...", "warning");
    
    // Inform the user visually
    setCurrentStage("Cancelling current analysis...");
    
    // Let the UI update briefly
    await new Promise(resolve => setTimeout(resolve, 300));
    
    // Wait until analysis is fully stopped
    while (isAnalyzing) {
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    
    // Clear cancellation flag
    cancelRef.current = false;
  };

  // The core analysis simulation wrapped in a useCallback so it can be re-run.
  const runAnalysis = useCallback(async (clearLogs = true) => {
    // Reset all states to initial values
    setIsAnalyzing(true);
    setError(null);
    setProgress(0);
    setCurrentStage("Initializing...");
    setElapsedSeconds(0);
    setIsCompleted(false);
    cancelRef.current = false;
    pauseRef.current = false;
    setIsPaused(false);
    setIsCanceled(false);
    setDetectedNames([]);
    if (clearLogs) {
      setLogs([]);
    }

    const startTime = Date.now();

    // Add initial logs (these will remain visible if clearLogs is false)
    addLog(`Starting analysis of ${videoFile.name}`, 'info');
    addLog(`Using model: ${getModelName(selectedModel)}`, 'info');

    // Update elapsed timer every second
    const timer = setInterval(() => {
      if (!pauseRef.current) {
        setElapsedSeconds(Math.floor((Date.now() - startTime) / 1000));
      }
    }, 1000);

    try {
      // Prepare form data (if needed for a real API call)
      const formData = new FormData();
      formData.append("video", videoFile);
      formData.append("model", selectedModel);

      // Add analysis options (using the current analysisOptions state)
      Object.entries(analysisOptions).forEach(([key, value]) => {
        formData.append(key, value.toString());
      });

      let currentProgress = 0;
      let mockNames = [
        { name: "Donald Trump", confidence: 0.95, timestamp: "00:01:23", frames: [82, 124, 187] },
        { name: "Boris Johnson", confidence: 0.87, timestamp: "00:03:45", frames: [225, 228, 230] },
        { name: "Angela Merkel", confidence: 0.92, timestamp: "00:05:12", frames: [312, 315] },
        { name: "Emmanuel Macron", confidence: 0.89, timestamp: "00:07:33", frames: [453, 456] },
        { name: "Vladimir Putin", confidence: 0.91, timestamp: "00:09:48", frames: [589, 590, 591] },
        { name: "Justin Trudeau", confidence: 0.86, timestamp: "00:12:05", frames: [726, 730] },
        { name: "Narendra Modi", confidence: 0.88, timestamp: "00:13:27", frames: [807, 810] },
      ];

      // Loop through stages with smoother progress
      for (let i = 0; i < stages.length; i++) {
        if (cancelRef.current) {
          throw new Error("Analysis was canceled by user");
        }
        
        const currentStageObj = stages[i];
        setCurrentStage(currentStageObj);
        
        // Main stage log
        addLog(`Stage ${i + 1}/${stages.length}: ${currentStageObj.message}`, 'info');
        
        // Add detailed logs for this stage (2-3 per stage)
        if (currentStageObj.logs) {
          // Add a small delay to make logs appear naturally
          await new Promise((resolve) => setTimeout(resolve, 300 / speedRef.current));
          
          // Choose 1-3 logs randomly from the available logs for this stage
          const numLogs = Math.min(Math.floor(Math.random() * 2) + 1, currentStageObj.logs.length);
          const selectedLogs = [...currentStageObj.logs]
            .sort(() => 0.5 - Math.random())
            .slice(0, numLogs);
            
          for (const logMsg of selectedLogs) {
            if (cancelRef.current) break;
            await new Promise((resolve) => setTimeout(resolve, 150 / speedRef.current));
            addLog(`${logMsg}`, 'info');
          }
        }

        // Special process logs for key stages
        if (currentStageObj.id === "yolo") {
          await new Promise((resolve) => setTimeout(resolve, 250 / speedRef.current));
          addLog(`YOLOv12 detection running with confidence threshold: ${analysisOptions.yoloConfidence.toFixed(2)}`, 'info');
          
          await new Promise((resolve) => setTimeout(resolve, 300 / speedRef.current));
          const detectedRegions = 12 + Math.floor(Math.random() * 8);
          addLog(`Detected ${detectedRegions} potential text regions in frame`, 'success');
        }
        
        if (currentStageObj.id === "ocr") {
          await new Promise((resolve) => setTimeout(resolve, 350 / speedRef.current));
          addLog(`Processing ${4 + Math.floor(Math.random() * 6)} high-confidence text regions`, 'info');
        }

        if (currentStageObj.id === "validate") {
          await new Promise((resolve) => setTimeout(resolve, 200 / speedRef.current));
          addLog(`Found ${mockNames.length} potential name candidates`, 'success');
        }

        // Simulate name detection at specific stages
        if (currentStageObj.id === "roi") {
          addLog("Found first potential name pattern in lower third graphic", "success");
          setDetectedNames([mockNames[0]]);
        } else if (currentStageObj.id === "nlp") {
          addLog(`Disambiguated ${mockNames.slice(0, 3).map(n => n.name).join(", ")}`, "success");
          setDetectedNames(mockNames.slice(0, 3));
        } else if (currentStageObj.id === "confidence") {
          addLog("Assigned confidence scores to all detected names", "info");
          setDetectedNames(mockNames.slice(0, 5));
        }

        // Calculate target progress for this stage with weighted distribution
        const stageWeight = i < 3 ? 0.7 : 1.3;
        const targetProgress = ((i + 1) / stages.length) * 100 * stageWeight;
        const adjustedTarget = Math.min(95, targetProgress); // Cap at 95% until final completion

        // Determine stage-specific timing
        const baseStageTime = [0, 1, 2].includes(i) ? 300 : i >= 7 && i <= 12 ? 600 : 450;
        const progressStep = (adjustedTarget - currentProgress) / 15;

        // Simulate progress bar increment
        for (let j = 0; j < 15; j++) {
          if (cancelRef.current) {
            throw new Error("Analysis was canceled by user");
          }

          // Handle pause during analysis
          while (pauseRef.current) {
            await new Promise((resolve) => setTimeout(resolve, 100));
            if (cancelRef.current) {
              throw new Error("Analysis was canceled by user");
            }
          }

          // Wait with speed adjustment
          const adjustedTime = baseStageTime / speedRef.current;
          await new Promise((resolve) => setTimeout(resolve, adjustedTime));

          currentProgress += progressStep;
          setProgress(Math.min(95, Math.round(currentProgress)));

          // Add a log midway through this stage
          if (j === 7 && i > 2) {
            addLog(`Processing frame group ${Math.floor(Math.random() * 1000)}`, 'info');
            
            // Add occasional warnings for realism
            if (Math.random() > 0.7 && currentStageObj.id !== "finalize") {
              const warnings = [
                "Low contrast detected in some frames, adjusting parameters",
                "Potential OCR confusion between similar characters",
                "Frame dropped due to blur detection",
                "Name pattern appears clipped at frame edge"
              ];
              const warningMsg = warnings[Math.floor(Math.random() * warnings.length)];
              await new Promise((resolve) => setTimeout(resolve, 150 / speedRef.current));
              addLog(warningMsg, 'warning');
            }
          }
        }
      }

      // Finalizing stage with a short pause
      if (cancelRef.current) {
        throw new Error("Analysis was canceled by user");
      }
      setCurrentStage("Finalizing results and preparing report...");
      await new Promise((resolve) => setTimeout(resolve, 1500 / speedRef.current));

      if (cancelRef.current) {
        throw new Error("Analysis was canceled by user");
      }

      // Set final states
      setProgress(100);
      setIsCompleted(true);
      setDetectedNames(mockNames);
      addLog("Analysis completed successfully", "success");

      // Simulate a successful backend response with dummy data
      const dummyResults: AnalysisResults = {
        names: mockNames,
        model: selectedModel,
        modelName: getModelName(selectedModel),
        processingTime: formatElapsed(Math.floor((Date.now() - startTime) / 1000)),
        videoMetadata: {
          filename: videoFile.name,
          size: (videoFile.size / (1024 * 1024)).toFixed(2) + " MB",
          duration:
            "00:" +
            Math.floor(Math.random() * 60)
              .toString()
              .padStart(2, "0") +
            ":00",
          resolution: "1920x1080",
          type: videoFile.type,
          frameRate: "30 fps",
        },
        analysisDate: new Date().toISOString(),
        detectionSettings: {
          yoloConfidence: analysisOptions.yoloConfidence,
          maxOverlap: analysisOptions.maxOverlap,
        },
      };

      // Give time for the progress bar to reach 100%
      await new Promise((resolve) => setTimeout(resolve, 1000));
      onAnalysisComplete(dummyResults);
    } catch (err: any) {
      console.error("Analysis error:", err);
      setError(err.message || "An error occurred during analysis. Please try again.");
      addLog(err.message || "Analysis failed with an error", "error");
    } finally {
      clearInterval(timer);
      setIsAnalyzing(false);
    }
  }, [videoFile, selectedModel, onAnalysisComplete, analysisOptions]);

  useEffect(() => {
    // Only trigger if a re-run was requested and analysis is not currently running
    if (rerunRef.current && !isAnalyzing) {
      addLog("Restarting analysis with new settings...", "info");
      setProgress(0);
      setElapsedSeconds(0);
      setEstimatedTime(null);
      setCurrentStage("Initializing...");
      
      runAnalysis(false); 
      rerunRef.current = false;
    }
  }, [isAnalyzing, analysisOptions, runAnalysis]);

  // Run analysis on component mount or when videoFile/selectedModel changes
  useEffect(() => {
    runAnalysis();
    
    // Cleanup function to ensure proper cancellation if component unmounts
    return () => {
      cancelRef.current = true;
    };
  }, [runAnalysis]);

  // Handle user cancellation (for the manual cancel modal)
  const [showCancelModal, setShowCancelModal] = useState(false);

  const confirmCancel = () => {
    cancelRef.current = true;
    setIsCanceled(true);
    setError("Analysis was canceled by user");
    setShowCancelModal(false);
  };
  
  const handleCancel = () => {
    setShowCancelModal(true);
  };
  
  // Handle retry in case of error
  const handleRetry = () => {
    runAnalysis();
  };

  return (
    <div className={`w-full ${className}`}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center">
          <h2 className="text-2xl font-bold">Video Analysis</h2>
        </div>
        <div className="flex items-center space-x-3">
          <button 
            onClick={() => setShowLogs(!showLogs)} 
            className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
            title="Show/Hide Logs"
          >
            <TerminalSquare className="h-5 w-5" />
          </button>
          <div className="flex items-center bg-gray-100 dark:bg-gray-800 rounded px-2 py-1">
            <Clock className="h-4 w-4 text-gray-500 mr-2" />
            <span className="text-gray-700 dark:text-gray-300 font-mono">{formatElapsed(elapsedSeconds)}</span>
          </div>
        </div>
      </div>

      {/* Cancel modal */}
      {showCancelModal && (
        <div className="fixed inset-0 flex items-center justify-center z-50 bg-black bg-opacity-50 backdrop-blur-sm">
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-2xl p-6 max-w-sm w-full mx-4">
            <div className="flex items-center mb-3">
              <div className="flex items-center justify-center w-10 h-10 rounded-full bg-amber-100 dark:bg-amber-900/30 mr-3">
                <AlertTriangle className="h-5 w-5 text-amber-600 dark:text-amber-300" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Cancel Analysis</h3>
            </div>
            
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-6 flex items-center gap-2">
              Are you sure you want to cancel the analysis? <br className="hidden sm:inline" />
            </p>

            <div className="flex justify-end gap-3">
              <button 
                onClick={() => setShowCancelModal(false)} 
                className="px-4 py-2 rounded-lg text-sm bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 text-gray-800 dark:text-white"
              >
                No, Continue Analysis
              </button>
              <button 
                onClick={confirmCancel} 
                className="px-4 py-2 rounded-lg text-sm bg-red-500 hover:bg-red-600 text-white font-medium"
              >
                Confirm Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {!error ? (
        <div className="animate-fade-in space-y-4">
          <div className="flex items-center justify-between">
            <span className="inline-flex items-center space-x-2 px-3 py-1 rounded-full text-sm font-medium bg-gray-100 dark:bg-gray-800 text-gray-800 dark:text-gray-200">
              <FileVideo className="h-4 w-4 text-gray-500" />
              <span>{videoFile.name}</span>
            </span>
            
            <span className="inline-flex items-center space-x-2 px-3 py-1 rounded-full text-sm font-medium bg-gray-100 dark:bg-gray-800 text-gray-800 dark:text-gray-200">
              <Cpu className="h-4 w-4 text-gray-500 dark:text-gray-300" />
              <span>{getModelName(selectedModel)}</span>
            </span>

          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 transition-all">
            <div className="mb-4 flex justify-between items-center">
              <StatusBadge stage={currentStage} />
              
              <div className="flex items-center space-x-2">
                {isAnalyzing && !isCompleted && !error && (
                  <>
                    <button 
                      onClick={handlePauseToggle}
                      className="p-1 rounded hover:bg-gray-100 dark:hover:bg-gray-700"
                      title={isPaused ? "Resume Analysis" : "Pause Analysis"}
                    >
                      {isPaused ? 
                        <FastForward className="h-5 w-5 text-amber-500" /> : 
                        <Pause className="h-5 w-5 text-gray-500" />
                      }
                    </button>
                  </>
                )}
              </div>
            </div>

            <h3 className="text-xl font-semibold mb-4">
              {progress < 100 ? "Processing Video" : "Analysis Complete"}
            </h3>

            <div className="w-full mb-4">
              <div className="bg-gray-200 dark:bg-gray-700 rounded-full h-3 overflow-hidden">
                <div
                  className="bg-primary h-3 rounded-full transition-all duration-300 ease-out"
                  style={{ width: `${progress}%` }}
                />
              </div>
              
              <div className="flex justify-between w-full mt-1">
                <p className="text-xs text-gray-500">{progress}% complete</p>
                <p className="text-xs text-gray-500">{isCompleted ? "Completed" : "Estimated time: ~2 min"}</p>
              </div>
            </div>
            
            {/* Action buttons */}
            <div className="flex flex-wrap justify-center gap-3 mt-6">
              {isAnalyzing && !isCanceled && progress < 100 && (
                <button
                  type="button"
                  onClick={handleCancel}
                  className="px-4 py-2 bg-red-500 hover:bg-red-600 text-white rounded-lg shadow-sm transition-colors duration-200 flex items-center"
                >
                  <XCircle className="mr-2 h-4 w-4" />
                  Cancel Analysis
                </button>
              )}
            </div>
          </div>

          {/* Analysis Logs */}
          {showLogs && (
            <CollapsibleSection 
              title="Analysis Logs" 
              icon={<TerminalSquare className="h-4 w-4" />}
              defaultOpen={true}
            >
              <div className="flex justify-between items-center mb-2">
                <h4 className="text-sm font-medium">Process Log ({logs.length} entries)</h4>
                <button 
                  onClick={copyLogs}
                  className="text-xs px-2 py-1 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 rounded flex items-center"
                >
                  <Clipboard className="h-3 w-3 mr-1" />
                  Copy Logs
                </button>
              </div>
              <div className="bg-gray-50 dark:bg-gray-900 rounded border border-gray-200 dark:border-gray-700 max-h-60 overflow-y-auto p-1">
                {logs.length > 0 ? (
                  logs.map((log, idx) => (
                    <LogEntry 
                      key={idx} 
                      message={log.message} 
                      timestamp={log.timestamp} 
                      type={log.type} 
                    />
                  ))
                ) : (
                  <p className="text-center text-gray-500 py-4">No logs available yet</p>
                )}
              </div>
            </CollapsibleSection>
          )}

          {/* Technical Details and Explanation */}
          <CollapsibleSection 
            title="How It Works" 
            icon={<BookOpen className="h-4 w-4" />}
            defaultOpen={false}
          >
            <div className="space-y-4">
              {/* Process Cards */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {/* Frame Extraction */}
                <div className="bg-blue-50 dark:bg-[#121E3C] border border-blue-200 dark:border-blue-700/40 p-4 rounded-lg">
                  <h5 className="font-medium mb-2 flex items-center text-blue-600 dark:text-blue-400">
                    <FileVideo className="h-4 w-4 mr-2 text-blue-500" />
                    Frame Extraction
                  </h5>
                  <p className="text-sm text-blue-700 dark:text-blue-300">
                    Extracts key frames from the video where text or faces are most likely to appear.
                  </p>
                </div>

                {/* ROI Cropping */}
                <div className="bg-amber-50 dark:bg-[rgba(255,200,80,0.08)] border border-amber-200 dark:border-amber-500/20 p-4 rounded-lg">
                  <h5 className="font-medium mb-2 flex items-center text-amber-700 dark:text-amber-300">
                    <Crop className="h-4 w-4 mr-2 text-amber-500 dark:text-amber-300" />
                    ROI Cropping
                  </h5>
                  <p className="text-sm text-amber-700 dark:text-amber-200">
                    YOLOv12 detects and crops areas with news video graphics (e.g., lower thirds) to support the next step.
                  </p>
                </div>

                {/* OCR + NLP */}
                <div className="bg-purple-50 dark:bg-[rgba(150,100,255,0.08)] border border-purple-200 dark:border-purple-500/20 p-4 rounded-lg">
                  <h5 className="font-medium mb-2 flex items-center text-purple-700 dark:text-purple-300">
                    <Eye className="h-4 w-4 mr-2 text-purple-500 dark:text-purple-300" />
                    OCR + NLP Analysis
                  </h5>
                  <p className="text-sm text-purple-700 dark:text-purple-200">
                    Extracts text from graphics and identifies names using language models.
                  </p>
                </div>
              </div>

              {/* Model Summary */}
              <div className="bg-gray-50 dark:bg-[#162032] p-4 rounded-lg border border-gray-200 dark:border-gray-700">
                <h5 className="font-medium mb-3 text-gray-900 dark:text-white">
                  Model Information: {getModelName(selectedModel)}
                </h5>
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  {selectedModel === "anep" && "Uses a custom OCR pipeline with post-processing and BERT-based name recognition."}
                  {selectedModel === "model1" && "Relies on Google Cloud Vision OCR with enhanced filtering logic."}
                  {selectedModel === "model2" && "Llama Vision 3.2, a multimodal model combining vision and text understanding."}
                  {selectedModel === "all" && "A combination of all three models utilized for comparison and research purposes."}
                </p>
              </div>
            </div>
          </CollapsibleSection>

          {/* Information box */}
          <div className="bg-blue-50 dark:bg-[#121E3C] rounded-xl border border-blue-200 dark:border-blue-700/50 shadow-sm">
            <div className="p-5 flex items-start">
              <div className="flex-shrink-0 mr-4">
                <div className="w-10 h-10 rounded-full bg-blue-100 dark:bg-blue-600/20 flex items-center justify-center shadow-md">
                  <Info className="h-5 w-5 text-blue-600 dark:text-blue-300" />
                </div>
              </div>
              <div>
                <h4 className="text-base font-semibold text-blue-800 dark:text-blue-200 mb-1">
                  Processing Information
                </h4>
                <p className="text-sm text-blue-700 dark:text-blue-300 leading-relaxed">
                  Analysis may take several minutes depending on the video length and complexity. 
                  The <span className="font-medium">{getModelName(selectedModel)}</span> is optimized for quality results. 
                  You can continue working in other tabs while your video is being processed.
                </p>
              </div>
            </div>
          </div>
        </div>
      ) : (
        // Error state
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg overflow-hidden">
          <div className="bg-red-50 dark:bg-red-900/20 px-6 py-4 border-b border-red-100 dark:border-red-900/30">
            <div className="flex items-center">
              <AlertTriangle className="h-6 w-6 text-red-500 mr-3" />
              <h3 className="text-lg font-medium text-red-800 dark:text-red-300">Analysis Failed</h3>
            </div>
          </div>
          <div className="p-6">
            <p className="text-red-700 dark:text-red-400 mb-6">{error}</p>
            <div className="bg-gray-100 dark:bg-gray-800/30 border border-gray-300 dark:border-gray-600 rounded-lg p-4 mb-6">
              <div className="flex items-center mb-2">
                <Wrench className="h-4 w-4 text-gray-600 dark:text-gray-300 mr-2" />
                <h4 className="font-medium text-gray-800 dark:text-gray-200">Troubleshooting</h4>
              </div>
              <ul className="text-sm text-gray-700 dark:text-gray-400 space-y-2 pl-5 list-disc">
                <li>Check your network connection and ensure stable internet</li>
                <li>Verify the video file is not corrupted or protected</li>
                <li>Try a different video format (MP4 is recommended)</li>
                <li>Select a different analysis model</li>
              </ul>
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-6 flex items-center gap-1 flex-wrap">
              Still having trouble?{" "}
              <a
                href="mailto:andrea.f.lucas.22@um.edu.mt"
                className="hover:text-[#3C83F6] hover:underline transition-colors duration-200"
              >
                Please reach out to our support team
              </a>{" "}
              for further assistance.
              <a
                href="mailto:andrea.f.lucas.22@um.edu.mt"
                className="ml-1 text-gray-500 hover:text-[#3C83F6] transition-colors duration-200"
                title="Email support"
              >
                <ExternalLink className="h-4 w-4 inline-block" />
              </a>
            </p>
            <div className="flex justify-center">
              <button
                type="button"
                onClick={handleRetry}
                className="px-6 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg shadow-sm transition-colors duration-200 flex items-center"
              >
                <RefreshCw className="h-4 w-4 mr-2" />
                Retry Analysis
              </button>
            </div>
          </div>
        </div>
      )}
      {/* Additional sections for error details */}
      {error && (
        <div className="mt-6 space-y-6">
          {showLogs && (
            <CollapsibleSection 
              title="Analysis Logs" 
              icon={<TerminalSquare className="h-4 w-4" />}
              defaultOpen={true}
            >
              <div className="flex justify-between items-center mb-2">
                <h4 className="text-sm font-medium">Process Log ({logs.length} entries)</h4>
                <button 
                  onClick={copyLogs}
                  className="text-xs px-2 py-1 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 rounded flex items-center"
                >
                  <Clipboard className="h-3 w-3 mr-1" />
                  Copy Logs
                </button>
              </div>
              <div className="bg-gray-50 dark:bg-gray-900 rounded border border-gray-200 dark:border-gray-700 max-h-60 overflow-y-auto p-1">
                {logs.length > 0 ? (
                  logs.map((log, idx) => (
                    <LogEntry 
                      key={idx} 
                      message={log.message} 
                      timestamp={log.timestamp} 
                      type={log.type} 
                    />
                  ))
                ) : (
                  <p className="text-center text-gray-500 py-4">No logs available yet</p>
                )}
              </div>
            </CollapsibleSection>
          )}
          <CollapsibleSection 
            title="File Information" 
            icon={<FileVideo className="h-4 w-4" />}
            defaultOpen={false}
          >
            <div className="divide-y divide-gray-200 dark:divide-gray-700 text-sm">
              <div className="flex justify-between py-2">
                <span className="text-gray-600 dark:text-gray-300">Name:</span>
                <span className="font-mono text-gray-800 dark:text-white">{videoFile.name}</span>
              </div>
              <div className="flex justify-between py-2">
                <span className="text-gray-600 dark:text-gray-300">Size:</span>
                <span className="font-mono text-gray-800 dark:text-white">{(videoFile.size / (1024 * 1024)).toFixed(2)} MB</span>
              </div>
              <div className="flex justify-between py-2">
                <span className="text-gray-600 dark:text-gray-300">Type:</span>
                <span className="font-mono text-gray-800 dark:text-white">{videoFile.type || "Unknown"}</span>
              </div>
            </div>
          </CollapsibleSection>
        </div>
      )}
    </div>
  );
};

export default AnalysisStep;