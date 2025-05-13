import { useState, useEffect, useRef, useCallback, useReducer } from "react";
import { 
  BarChart3, 
  AlertTriangle, 
  Clock, 
  CheckCircle, 
  XCircle, 
  Info, 
  FileVideo, 
  Cpu, 
  Loader2, 
  ChevronDown,
  ChevronUp,
  Clipboard,
  RefreshCw,
  TerminalSquare,
  BookOpen,
  Wrench,
  ExternalLink,
  Server
} from "lucide-react";

// Types
type AnalysisStatus = "processing" | "success" | "error";
type LogType = 'info' | 'warning' | 'success' | 'error';

interface AnalysisStepProps {
  videoFile: File;
  selectedModel: string;
  onAnalysisComplete: (results: any) => void;
  className?: string;
  onDownloadResults?: () => void;
}

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
}

interface LogEntry {
  message: string;
  timestamp: string;
  type: LogType;
}

// Utility functions
const formatElapsed = (seconds: number) => {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
};

const processLogText = (text: string) => {
  // Remove ANSI color codes
  let processed = text
    .replace(/\x1b\[0m/g, '') // Reset color
    .replace(/\x1b\[31m/g, '') // Red
    .replace(/\x1b\[32m/g, '') // Green
    .replace(/\x1b\[33m/g, '') // Yellow
    .replace(/\x1b\[34m/g, ''); // Blue

  // Remove file paths and timestamps from inside logs
  processed = processed
    .replace(/\/[^\s]+\/[^\s]+\.py:\d+/g, '')
    .replace(/\[\d+:\d+:\d+\]\s+INFO\s+\[\d+:\d+:\d+\]/g, '[LOG]')
    .replace(/\[\d+:\d+:\d+\]\s+INFO\s+INFO\s+/g, '[LOG] ');

  // Remove verbose technical details
  processed = processed
    .replace(/Created directory: \/.*?(?=\s|$)/g, 'Created directory')
    .replace(/\/opt\/anaconda3\/envs\/.*?(?=\s|$)/g, '[env path]')
    .replace(/\/Volumes\/.*?(?=\s|$)/g, '[storage path]');

  // Simplify warning messages
  if (
    processed.includes('UserWarning') ||
    processed.includes('not used when initializing') ||
    processed.includes('expected if you are initializing')
  ) {
    return '[Technical warning message - hidden]';
  }

  // Custom override: exit code -9 means process was killed
  if (processed.includes('Process failed with exit code -9')) {
    return 'Terminated Execution Successfully by User';
  }
  return processed;
};


const filterLogs = (logs: LogEntry[]): LogEntry[] => {
  if (!logs || logs.length === 0) return [];
  
  // Keep track of seen similar messages to avoid duplication
  const seenMessages = new Set<string>();
  const filteredLogs: LogEntry[] = [];
  
  logs.forEach(log => {
    // Skip empty logs, whitespace-only logs, or technical warnings
    if (!log.message || log.message.trim() === '' || 
        log.message === '[Technical warning message - hidden]') {
      return;
    }
    
    // Create a simplified version of the message for deduplication
    let simplifiedMsg = log.message
      .replace(/\d+/g, 'N') // Replace numbers with N
      .replace(/frame \w+/g, 'frame X') // Normalize frame references
      .toLowerCase();
    
    // Skip certain technical or redundant logs
    if (log.message.includes('warnings.warn') || 
        log.message.includes('tokenizer that you are converting') ||
        log.message.includes('/site-packages/') ||
        log.message.includes('Default to no truncation') ||
        log.message.includes('weights of the model checkpoint') ||
        (log.message.includes('ROIs in frame') && seenMessages.has('found rois in frame'))) {
      return;
    }
    
    // Keep important logs regardless of duplication
    const isImportant = 
      log.message.includes('Starting analysis') ||
      log.message.includes('Processing video') ||
      log.message.includes('Detected') ||
      log.message.includes('Name Detection Summary') ||
      log.message.includes('success') ||
      log.message.includes('error') ||
      log.message.includes('warning') ||
      log.type !== 'info';
    
    if (isImportant || !seenMessages.has(simplifiedMsg)) {
      // Add to filtered logs and mark as seen
      const processedMessage = processLogText(log.message);
      
      // Skip empty messages after processing
      if (processedMessage && processedMessage.trim() !== '') {
        filteredLogs.push({
          ...log,
          message: processedMessage
        });
        seenMessages.add(simplifiedMsg);
      }
    }
  });
  
  return filteredLogs;
};

const getModelName = (modelId: string): string => {
  const modelMap: Record<string, string> = {
    "anep": "ANEP (Accurate Name Extraction Pipeline)",
    "model1": "Google Cloud Vision & Gemini 1.5 Pro",
    "model2": "Llama 4 Maverick",
    "all": "All Models (Ensemble)",
  };
  
  return modelMap[modelId] || "Custom Model";
};

const createTimestamp = () => {
  const now = new Date();
  return `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}`;
};

const getApiEndpoint = (selectedModel: string) => {
  switch(selectedModel) {
    case "anep": return "/api/run/anep";
    case "model1": return "/api/run/gcloud";
    case "model2": return "/api/run/llama";
    case "all": return "/api/run/ensemble";
    default: return "/api/run/anep";
  }
};

// Sub-components
const StatusBadge = ({ status, message }: { status: AnalysisStatus; message: string }) => {
  const getStatusColor = () => {
    switch (status) {
      case "processing": 
        return "bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300";
      case "success": 
        return "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300";
      case "error": 
        return "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300";
      default:
        return "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300";
    }
  };

  const getStatusIcon = () => {
    switch (status) {
      case "processing": 
        return <Loader2 className="h-3 w-3 mr-1 animate-spin" />;
      case "success": 
        return <CheckCircle className="h-3 w-3 mr-1" />;
      case "error": 
        return <XCircle className="h-3 w-3 mr-1" />;
      default:
        return <Info className="h-3 w-3 mr-1" />;
    }
  };

  return (
    <span className={`px-3 py-1 rounded-full text-sm font-medium flex items-center ${getStatusColor()}`}>
      {getStatusIcon()}
      {message}
    </span>
  );
};

const LogEntryItem = ({ message, timestamp, type }: LogEntry) => {
  const typeStyles = {
    info: "border-blue-200 bg-blue-50 dark:border-blue-900/40 dark:bg-blue-900/20 shadow-sm",
    warning: "border-amber-200 bg-amber-50 dark:border-amber-900/40 dark:bg-amber-900/20 shadow-sm",
    success: "border-green-200 bg-green-50 dark:border-green-900/40 dark:bg-green-900/20 shadow-sm",
    error: "border-red-200 bg-red-50 dark:border-red-900/40 dark:bg-red-900/20 shadow-sm"
  };

  const typeIcons = {
    info: <Info className="h-4 w-4 text-blue-500 dark:text-blue-400" />,
    warning: <AlertTriangle className="h-4 w-4 text-amber-500 dark:text-amber-400" />,
    success: <CheckCircle className="h-4 w-4 text-green-500 dark:text-green-400" />,
    error: <XCircle className="h-4 w-4 text-red-500 dark:text-red-400" />
  };

  return (
    <div className={`border-l-4 px-3 py-2 mb-2 rounded-r-md ${typeStyles[type]} transition-all duration-200 hover:translate-x-1`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {typeIcons[type]}
          <span className="text-sm font-mono">{message}</span>
        </div>
        <span className="text-xs text-gray-500 dark:text-gray-400 font-mono bg-gray-100 dark:bg-gray-800 px-2 py-0.5 rounded-full">{timestamp}</span>
      </div>
    </div>
  );
};

const CollapsibleSection = ({ 
  title, 
  icon, 
  children, 
  defaultOpen = false 
}: {
  title: string;
  icon: React.ReactNode;
  children: React.ReactNode;
  defaultOpen?: boolean;
}) => {
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

// Analysis component reducer
type AnalysisState = {
  progress: number;
  status: AnalysisStatus;
  statusMessage: string;
  error: string | null;
  elapsedSeconds: number;
  isAnalyzing: boolean;
  isCanceled: boolean;
  isCompleted: boolean;
  showLogs: boolean;
  logs: LogEntry[];
  processId: string | null;
  showCancelModal: boolean;
};

type AnalysisAction =
  | { type: 'START_ANALYSIS' }
  | { type: 'UPDATE_PROGRESS'; payload: number }
  | { type: 'SET_STATUS'; payload: { status: AnalysisStatus; message: string } }
  | { type: 'SET_ERROR'; payload: string }
  | { type: 'INCREMENT_TIME' }
  | { type: 'COMPLETE_ANALYSIS' }
  | { type: 'CANCEL_ANALYSIS' }
  | { type: 'TOGGLE_LOGS' }
  | { type: 'ADD_LOG'; payload: { message: string; type: LogType } }
  | { type: 'SET_PROCESS_ID'; payload: string }
  | { type: 'TOGGLE_CANCEL_MODAL' };

const initialState: AnalysisState = {
  progress: 0, // Starting at 0% instead of 5%
  status: "processing",
  statusMessage: "Initializing...",
  error: null,
  elapsedSeconds: 0,
  isAnalyzing: false,
  isCanceled: false,
  isCompleted: false,
  showLogs: false, // Logs hidden by default
  logs: [],
  processId: null,
  showCancelModal: false,
};

function analysisReducer(state: AnalysisState, action: AnalysisAction): AnalysisState {
  switch (action.type) {
    case 'START_ANALYSIS':
      return {
        ...initialState,
        isAnalyzing: true,
        progress: 1, // Start with 1% instead of 5%
        statusMessage: "Initializing analysis...",
        showLogs: state.showLogs, // Preserve log visibility preference
      };
    case 'UPDATE_PROGRESS':
      // Prevent progress from going backward once completed
      // And don't allow progress updates if we're already completed
      if (state.isCompleted || action.payload < state.progress) {
        return state;
      }
      return {
        ...state,
        progress: Math.min(action.payload, state.isCompleted ? 100 : 95) // Cap at 95% unless completed
      };
    case 'SET_STATUS':
      return {
        ...state,
        status: action.payload.status,
        statusMessage: action.payload.message
      };
    case 'SET_ERROR':
      return {
        ...state,
        error: action.payload,
        status: "error",
        statusMessage: "Analysis failed",
        isAnalyzing: false
      };
    case 'INCREMENT_TIME':
      // Only increment time if we're still analyzing
      if (!state.isAnalyzing) {
        return state;
      }
      return {
        ...state,
        elapsedSeconds: state.elapsedSeconds + 1
      };
    case 'COMPLETE_ANALYSIS':
      // Prevent marking as complete multiple times
      if (state.isCompleted) {
        return state;
      }
      return {
        ...state,
        progress: 100,
        status: "success",
        statusMessage: "Analysis completed",
        isCompleted: true,
        isAnalyzing: false
      };
    case 'CANCEL_ANALYSIS':
      return {
        ...state,
        isCanceled: true,
        error: "Analysis Was Canceled by User!",
        status: "error",
        statusMessage: "Analysis canceled",
        isAnalyzing: false,
        showCancelModal: false
      };
    case 'TOGGLE_LOGS':
      return {
        ...state,
        showLogs: !state.showLogs
      };
    case 'ADD_LOG':
      return {
        ...state,
        logs: [
          ...state.logs,
          { 
            message: action.payload.message, 
            timestamp: createTimestamp(),
            type: action.payload.type 
          }
        ]
      };
    case 'SET_PROCESS_ID':
      return {
        ...state,
        processId: action.payload
      };
    case 'TOGGLE_CANCEL_MODAL':
      return {
        ...state,
        showCancelModal: !state.showCancelModal
      };
    default:
      return state;
  }
}

// Main component
const AnalysisStep = ({
  videoFile,
  selectedModel,
  onAnalysisComplete,
  className = "",
  onDownloadResults,
}: AnalysisStepProps) => {
  const [state, dispatch] = useReducer(analysisReducer, initialState);
  const [detectedNames, setDetectedNames] = useState<NameDetection[]>([]);
  const [completionCountdown, setCompletionCountdown] = useState<number>(5);
  
  const logsContainerRef = useRef<HTMLDivElement>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const cancelRef = useRef(false);
  const timerRef = useRef<NodeJS.Timeout | number | null>(null);

  // Handle log events from server
  const handleLogEvent = useCallback((data: any) => {
    if (data.process_id === state.processId) {
      dispatch({ 
        type: 'ADD_LOG', 
        payload: { 
          message: data.message, 
          type: data.type as LogType 
        } 
      });
      
      // Check for name detection in logs
      if (data.message.includes("Names detected in frame") || 
          data.message.includes("Name Detection Summary")) {
        
        // Extract name detection data
        const nameMatches = data.message.match(/- ([A-Za-z\s]+) \(Methods:.+Confidence: ([\d\.]+)\)/g);
        
        if (nameMatches) {
          const newDetections: NameDetection[] = nameMatches.map((match: string) => {
            const nameMatch = match.match(/- ([A-Za-z\s]+) \(Methods:.+Confidence: ([\d\.]+)\)/);
            if (nameMatch) {
              return {
                name: nameMatch[1],
                confidence: parseFloat(nameMatch[2]),
                timestamp: data.message.includes("frame") ? 
                  data.message.match(/frame (\d+)/)?.[1] || "unknown" : "summary",
                frames: []
              };
            }
            return null;
          }).filter(Boolean);
          
          if (newDetections.length > 0) {
            setDetectedNames(prev => {
              // Combine with previous unique names
              const combined = [...prev];
              newDetections.forEach(detection => {
                const existingIndex = combined.findIndex(d => d.name === detection.name);
                if (existingIndex >= 0) {
                  // Update confidence if higher
                  if (detection.confidence > combined[existingIndex].confidence) {
                    combined[existingIndex].confidence = detection.confidence;
                  }
                } else {
                  combined.push(detection);
                }
              });
              return combined;
            });
          }
        }
      }
      
      // Only process progress updates if we're still analyzing and not completed
      if (state.isAnalyzing && !state.isCompleted) {
        const msg = data.message.toLowerCase();
        
        // Update progress based on log content
        if (msg.includes("starting") || msg.includes("initializing")) {
          dispatch({ type: 'UPDATE_PROGRESS', payload: 5 }); // Reduced from 15 to 5
        } 
        else if (msg.includes("processing") || msg.includes("analyzing") || 
                msg.includes("extracting") || msg.includes("frame")) {
          let increment = 2;
          if (msg.includes("50%") || msg.includes("half")) increment = 10;
          else if (msg.includes("complete")) increment = 8;
          else if (msg.includes("progress")) increment = 5;
          
          // Make sure we don't exceed 95% until fully complete
          dispatch({ type: 'UPDATE_PROGRESS', payload: Math.min(state.progress + increment, 95) });
        } 
        // Only mark as complete if we get a very clear completion message
        else if ((msg.includes("completed") && msg.includes("100%")) || 
                (msg.includes("finished") && msg.includes("successfully")) || 
                (msg.includes("process completed successfully")) ||
                (msg.includes("done") && msg.includes("all processing"))) {
          // Make sure we haven't already marked as complete
          if (!state.isCompleted) {
            dispatch({ type: 'COMPLETE_ANALYSIS' });
          }
        } 
        else if (msg.includes("error") || msg.includes("failed") || msg.includes("exception")) {
          dispatch({ type: 'SET_ERROR', payload: data.message });
        }
      }
    }
  }, [state.processId, state.progress, state.isAnalyzing, state.isCompleted]);

  // Auto-scroll logs to bottom when new logs come in
  useEffect(() => {
    if (logsContainerRef.current) {
      logsContainerRef.current.scrollTop = logsContainerRef.current.scrollHeight;
    }
  }, [state.logs]);

  // Set up event source for real-time logs
  useEffect(() => {
    if (!state.processId) return;
    
    // Close existing connection if any
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }
    
    // Connect to SSE stream
    const eventSource = new EventSource('/api/events');
    eventSourceRef.current = eventSource;
    
    // Listen for messages
    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handleLogEvent(data);
      } catch (err) {
        console.error("Error parsing SSE data:", err);
      }
    };
    
    // Handle errors
    eventSource.onerror = (err) => {
      console.error("SSE connection error:", err);
      dispatch({ 
        type: 'ADD_LOG', 
        payload: { message: "Lost connection to server log stream", type: "error" } 
      });
      eventSource.close();
    };
    
    // Cleanup on unmount
    return () => {
      eventSource.close();
    };
  }, [state.processId, handleLogEvent]);

  // Timer effect for analysis time tracking
  useEffect(() => {
    if (!state.isAnalyzing) return;
    
    timerRef.current = setInterval(() => {
      dispatch({ type: 'INCREMENT_TIME' });
    }, 1000);
    
    return () => {
      if (timerRef.current !== null) {
        clearInterval(timerRef.current);
      }
    };
  }, [state.isAnalyzing]);

  // Countdown timer effect after completion
  useEffect(() => {
    if (state.isCompleted && completionCountdown > 0) {
      const timer = setTimeout(() => {
        setCompletionCountdown(prevCount => prevCount - 1);
      }, 1000);
      
      return () => clearTimeout(timer);
    } else if (state.isCompleted && completionCountdown === 0) {
      // Create a result object using file metadata and detected names
      const analysisResults: AnalysisResults = {
        names: detectedNames,
        model: selectedModel,
        modelName: getModelName(selectedModel),
        processingTime: formatElapsed(state.elapsedSeconds),
        videoMetadata: {
          filename: videoFile.name,
          size: (videoFile.size / (1024 * 1024)).toFixed(2) + " MB",
          duration: "Processed",
          resolution: "Extracted", 
          type: videoFile.type,
        },
        analysisDate: new Date().toISOString(),
      };
      
      // Navigate to next step
      onAnalysisComplete(analysisResults);
    }
  }, [state.isCompleted, completionCountdown, detectedNames, selectedModel, videoFile, state.elapsedSeconds, onAnalysisComplete]);

  // Poll for process status
  useEffect(() => {
    if (!state.processId || !state.isAnalyzing) return;
    
    const checkStatus = async () => {
      try {
        const response = await fetch(`/api/process/${state.processId}/status`);
        const data = await response.json();
        
        if (!data.active && state.isAnalyzing) {
          dispatch({ type: 'COMPLETE_ANALYSIS' });
        } else if (state.isAnalyzing) {
          // Gradually increment progress based on time
          const timeBasedIncrement = state.elapsedSeconds > 10 ? 2 : 0;
          dispatch({ 
            type: 'UPDATE_PROGRESS', 
            payload: Math.min(state.progress + timeBasedIncrement, 95) 
          });
        }
      } catch (err) {
        console.error("Error checking process status:", err);
      }
    };
    
    // Poll every 5 seconds
    const interval = setInterval(checkStatus, 5000);
    
    return () => clearInterval(interval);
  }, [state.processId, state.isAnalyzing, state.elapsedSeconds, state.progress]);

  // Cancel the analysis and terminate the process on the server
  const cancelAnalysis = async () => {
    if (!state.processId) {
      // If we don't have a process ID, just mark as cancelled in the UI
      dispatch({ type: 'CANCEL_ANALYSIS' });
      return;
    }
    
    try {
      // Send request to cancel the process
      const response = await fetch(`/api/process/${state.processId}/cancel`, {
        method: 'POST',
      });
      
      if (!response.ok) {
        throw new Error('Failed to cancel process');
      }
      
      dispatch({ 
        type: 'ADD_LOG', 
        payload: { message: "Sent termination request to server", type: 'info' } 
      });
      
      // Update UI state to show cancellation
      dispatch({ type: 'CANCEL_ANALYSIS' });
      
    } catch (err: any) {
      console.error("Error cancelling process:", err);
      dispatch({ 
        type: 'ADD_LOG', 
        payload: { message: "Failed to cancel process on server: " + (err.message || "Unknown error"), type: 'error' } 
      });
      // Still update UI to show cancellation even if server request failed
      dispatch({ type: 'CANCEL_ANALYSIS' });
    }
  };

  // Run the analysis
  const runAnalysis = useCallback(async () => {
    // Reset state and start analysis
    dispatch({ type: 'START_ANALYSIS' });
    dispatch({ 
      type: 'ADD_LOG', 
      payload: { message: `Starting analysis of ${videoFile.name}`, type: 'info' } 
    });
    dispatch({ 
      type: 'ADD_LOG', 
      payload: { message: `Using model: ${getModelName(selectedModel)}`, type: 'info' } 
    });

    try {
      // Make API call
      dispatch({ 
        type: 'SET_STATUS', 
        payload: { 
          status: "processing", 
          message: `Processing with ${getModelName(selectedModel)}...` 
        } 
      });
      
      dispatch({ 
        type: 'ADD_LOG', 
        payload: { message: `Contacting server at ${getApiEndpoint(selectedModel)}`, type: 'info' } 
      });
      
      // Update to 3% instead of 10%
      dispatch({ type: 'UPDATE_PROGRESS', payload: 3 });
      
      // Make the API call to start processing
      const response = await fetch(getApiEndpoint(selectedModel), {
        method: 'POST',
        headers: { 'Accept': 'application/json' }
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || 'Failed to process video');
      }
      
      // Get the response
      const result = await response.json();
      
      // Store the process ID for streaming logs
      if (result.process_id) {
        dispatch({ type: 'SET_PROCESS_ID', payload: result.process_id });
        dispatch({ 
          type: 'ADD_LOG', 
          payload: { message: `Server processing started with ID: ${result.process_id}`, type: 'success' } 
        });
      } else {
        throw new Error('Server did not return a process ID');
      }
      
      dispatch({ type: 'UPDATE_PROGRESS', payload: 3 });
      dispatch({ 
        type: 'SET_STATUS', 
        payload: { status: "processing", message: "Server is processing your video..." } 
      });
      
    } catch (err: any) {
      console.error("Analysis error:", err);
      dispatch({ type: 'SET_ERROR', payload: err.message || "An error occurred during analysis. Please try again." });
      dispatch({ 
        type: 'ADD_LOG', 
        payload: { message: err.message || "Analysis failed with an error", type: "error" } 
      });
    }
  }, [videoFile, selectedModel]);

  // Run analysis when component mounts
  useEffect(() => {
    runAnalysis();
    
    // Cleanup function
    return () => {
      cancelRef.current = true;
      if (timerRef.current !== null) {
        clearInterval(timerRef.current);
      }
      
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, [runAnalysis]);

  // Utility functions
  const copyLogs = () => {
    const logText = filterLogs(state.logs).map(log => `[${log.timestamp}] ${log.type.toUpperCase()}: ${log.message}`).join('\n');
    navigator.clipboard.writeText(logText);
    dispatch({ 
      type: 'ADD_LOG', 
      payload: { message: "Logs copied to clipboard", type: 'success' } 
    });
  };

  // Render JSX
  return (
    <div className={`w-full ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center">
          <h2 className="text-2xl font-bold">Video Analysis</h2>
        </div>
        <div className="flex items-center space-x-3">
          <button 
            onClick={() => dispatch({ type: 'TOGGLE_LOGS' })} 
            className={`p-1.5 rounded-md transition-colors ${state.showLogs ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300' : 'text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-800'}`}
            title="Show/Hide Logs"
          >
            <TerminalSquare className="h-5 w-5" />
          </button>
          <div className="flex items-center bg-gray-100 dark:bg-gray-800 rounded-md px-3 py-1.5 shadow-sm">
            <Clock className="h-4 w-4 text-gray-500 dark:text-gray-400 mr-2" />
            <span className="text-gray-700 dark:text-gray-300 font-mono">{formatElapsed(state.elapsedSeconds)}</span>
          </div>
        </div>
      </div>

      {/* Cancel modal */}
      {state.showCancelModal && (
        <div className="fixed inset-0 flex items-center justify-center z-50 bg-black bg-opacity-50 backdrop-blur-sm">
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-2xl p-6 max-w-sm w-full mx-4">
            <div className="flex items-center mb-3">
              <div className="flex items-center justify-center w-10 h-10 rounded-full bg-amber-100 dark:bg-amber-900/30 mr-3">
                <AlertTriangle className="h-5 w-5 text-amber-600 dark:text-amber-300" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Cancel Analysis</h3>
            </div>
            
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-6">
              Are you sure you want to cancel the analysis?
            </p>

            <div className="flex justify-end gap-3">
              <button 
                onClick={() => dispatch({ type: 'TOGGLE_CANCEL_MODAL' })} 
                className="px-4 py-2 rounded-lg text-sm bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 text-gray-800 dark:text-white"
              >
                No, Continue Analysis
              </button>
              <button 
                onClick={cancelAnalysis} 
                className="px-4 py-2 rounded-lg text-sm bg-red-500 hover:bg-red-600 text-white font-medium"
              >
                Confirm Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {!state.error ? (
        <div className="animate-fade-in space-y-4">
          {/* Video and model info */}
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

          {/* Main progress card */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 transition-all mb-6">
            <div className="mb-4 flex justify-between items-center">
              <StatusBadge status={state.status} message={state.statusMessage} />
            </div>

            <h3 className="text-xl font-semibold mb-5">
              {state.progress < 100 ? "Processing Video" : "Analysis Complete"}
            </h3>

            <div className="w-full mb-5">
              <div className="bg-gray-200 dark:bg-gray-700 rounded-full h-3 overflow-hidden">
                <div
                  className="bg-primary h-3 rounded-full transition-all duration-300 ease-out"
                  style={{ width: `${state.progress}%` }}
                />
              </div>
              
              <div className="flex justify-between w-full mt-2">
                <p className="text-xs text-gray-500">{Math.round(state.progress)}% complete</p>
                <p className="text-xs text-gray-500">{state.isCompleted ? "Completed" : "Processing..."}</p>
              </div>
            </div>
            
            {/* Action buttons */}
            {state.isAnalyzing && !state.isCanceled && state.progress < 100 && (
              <div className="flex justify-center mt-6">
                <button
                  type="button"
                  onClick={() => dispatch({ type: 'TOGGLE_CANCEL_MODAL' })}
                  className="px-4 py-2 bg-red-500 hover:bg-red-600 text-white rounded-lg shadow-sm transition-colors duration-200 flex items-center"
                >
                  <XCircle className="mr-2 h-4 w-4" />
                  Cancel Analysis
                </button>
              </div>
            )}
            
            {/* Completed actions */}
            {state.isCompleted && (
              <div className="flex flex-col items-center mt-6">
                <div className="text-green-600 dark:text-green-400 text-sm mb-4">
                  Analysis complete. Proceeding to results in <span className="font-medium">{completionCountdown}s</span>...
                </div>
                
                {detectedNames.length > 0 && (
                  <div className="mt-4 w-full">
                    <div className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-2">
                      Detected Names ({detectedNames.length}):
                    </div>
                    <div className="bg-gray-50 dark:bg-gray-900 p-2 rounded-lg border border-gray-200 dark:border-gray-700">
                      {detectedNames.map((name, idx) => (
                        <div key={idx} className="flex items-center justify-between p-2 border-b last:border-0 border-gray-200 dark:border-gray-700">
                          <div className="flex items-center">
                            <div className="h-8 w-8 rounded-full bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center mr-3">
                              <svg className="h-4 w-4 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                              </svg>
                            </div>
                            <div>
                              <div className="text-sm font-medium text-gray-800 dark:text-gray-200">{name.name}</div>
                              <div className="text-xs text-gray-500">At {name.timestamp === "summary" ? "multiple timestamps" : `frame ${name.timestamp}`}</div>
                            </div>
                          </div>
                          <div className="flex items-center">
                            <div className="px-2 py-1 bg-blue-100 dark:bg-blue-900/20 text-xs text-blue-700 dark:text-blue-300 rounded-full">
                              {Math.round(name.confidence * 100)}% confidence
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Analysis Logs */}
          {state.showLogs && filterLogs(state.logs).length > 0 && (
            <CollapsibleSection 
              title="CLI Output & Analysis Logs" 
              icon={<TerminalSquare className="h-4 w-4" />}
              defaultOpen={false}
            >
              <div className="flex justify-between items-center mb-3">
                <div className="flex items-center">
                  <h4 className="text-sm font-medium bg-gray-100 dark:bg-gray-800 px-3 py-1 rounded-full">Process Log ({filterLogs(state.logs).length} entries)</h4>
                  <div className="ml-2 px-2 py-1 rounded-full bg-blue-100 dark:bg-blue-900/30 text-xs text-blue-700 dark:text-blue-300 flex items-center">
                    <Server className="h-3 w-3 mr-1" />
                    <span>Server Status: {state.processId ? "Connected" : "Waiting"}</span>
                  </div>
                </div>
                <div className="flex gap-2">
                  <button 
                    onClick={copyLogs}
                    className="text-xs px-3 py-1.5 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 rounded-md flex items-center shadow-sm transition-colors"
                  >
                    <Clipboard className="h-3 w-3 mr-1" />
                    Copy Logs
                  </button>
                </div>
              </div>
              <div 
                ref={logsContainerRef}
                className="bg-gray-50 dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700 max-h-96 overflow-y-auto p-2 shadow-inner"
                style={{ scrollBehavior: 'smooth' }}
              >
                {filterLogs(state.logs).map((log, idx) => (
                  <LogEntryItem key={idx} {...log} />
                ))}
              </div>
            </CollapsibleSection>
          )}

          {/* Model information */}
          <CollapsibleSection 
            title="Model Information" 
            icon={<BookOpen className="h-4 w-4" />}
            defaultOpen={false}
          >
            <div className="p-2">
              <div className="flex items-center mb-3">
                <div className="h-10 w-10 rounded-full bg-primary/10 flex items-center justify-center mr-3 flex-shrink-0">
                  <Cpu className="h-5 w-5 text-primary" />
                </div>
                <div>
                  <h5 className="font-medium text-gray-900 dark:text-white">
                    {getModelName(selectedModel)}
                  </h5>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    {selectedModel === "anep" && "Advanced OCR and NER pipeline"}
                    {selectedModel === "model1" && "Google Cloud Vision with Gemini 1.5 Pro"}
                    {selectedModel === "model2" && "Llama 4 Maverick (Multimodal Vision-Language Model)"}
                    {selectedModel === "all" && "Combined ensemble of all models"}
                  </p>
                </div>
              </div>
              
              <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed my-2">
                {selectedModel === "anep" && "Optimized for name extraction from video frames with text overlays like news, interviews, and documentaries."}
                {selectedModel === "model1" && "Uses cloud-based OCR with AI context understanding for reliable text and name extraction."}
                {selectedModel === "model2" && "Leverages visual and textual understanding for detecting names in complex scenes."}
                {selectedModel === "all" && "Maximum accuracy through combined analysis, ideal for critical research."}
              </p>
              
              <div className="flex items-center mt-4 text-xs text-blue-600 dark:text-blue-400">
                <Info className="h-4 w-4 mr-2 text-blue-500" />
                <span>
                  {selectedModel === "anep" && "Best for most video content with good performance balance"}
                  {selectedModel === "model1" && "Reliable for clear text with consistent quality"}
                  {selectedModel === "model2" && "Ideal for short news clips and simple video layouts"}
                  {selectedModel === "all" && "Recommended for research purposes, and finding the optimal model for specific use cases"}
                </span>
              </div>
            </div>
          </CollapsibleSection>

          {/* Information box */}
          <div className="bg-blue-50 dark:bg-[#121E3C] rounded-xl border border-blue-200 dark:border-blue-700/50 shadow-sm">
            <div className="p-4 flex items-start">
              <div className="flex-shrink-0 mr-3">
                <div className="w-8 h-8 rounded-full bg-blue-100 dark:bg-blue-600/20 flex items-center justify-center shadow-md">
                  <Info className="h-4 w-4 text-blue-600 dark:text-blue-300" />
                </div>
              </div>
              <div>
                <h4 className="text-sm font-semibold text-blue-800 dark:text-blue-200 mb-1">
                  Processing Information
                </h4>
                <p className="text-xs text-blue-700 dark:text-blue-300 leading-relaxed">
                  Analysis may take several minutes depending on the video length and complexity. 
                  You can continue working in other tabs while your video is being processed.
                </p>
              </div>
            </div>
          </div>
        </div>
      ) : (
        // Error state
        <div className="animate-fade-in space-y-6">
          {/* Error card */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg overflow-hidden">
            <div className="bg-red-50 dark:bg-red-900/20 px-6 py-5 border-b border-red-100 dark:border-red-900/30">
              <div className="flex items-center">
                <div className="h-10 w-10 rounded-full bg-red-100 dark:bg-red-500/20 flex items-center justify-center mr-4">
                  <AlertTriangle className="h-5 w-5 text-red-500 dark:text-red-400" />
                </div>
                <div>
                  <h3 className="text-lg font-medium text-red-800 dark:text-red-300">Analysis Failed</h3>
                  <p className="text-sm text-red-600 dark:text-red-400 mt-1">The process encountered an error and could not complete</p>
                </div>
              </div>
            </div>
            <div className="p-6">
              <div className="mb-6 p-4 bg-red-50 dark:bg-red-900/10 rounded-lg border border-red-100 dark:border-red-900/30">
                <p className="text-red-700 dark:text-red-400 font-medium text-sm">
                  {processLogText(state.error || "")}
                </p>
              </div>
              
              <div className="flex justify-center">
                <button
                  type="button"
                  onClick={runAnalysis}
                  className="px-6 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg shadow-sm transition-colors duration-200 flex items-center"
                >
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Retry Analysis
                </button>
              </div>
            </div>
          </div>
          
          {/* Troubleshooting card */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
            <div className="flex items-center mb-4">
              <Wrench className="h-5 w-5 text-gray-600 dark:text-gray-300 mr-2" />
              <h4 className="text-lg font-medium text-gray-800 dark:text-gray-200">Troubleshooting Steps</h4>
            </div>
            
            <div className="grid md:grid-cols-2 gap-3">
              <div className="bg-gray-50 dark:bg-gray-900/30 p-3 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 flex items-start">
                <div className="mt-0.5 mr-2 flex-shrink-0">
                  <svg className="h-4 w-4 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                </div>
                <p className="text-sm text-gray-700 dark:text-gray-300">Check your network connection and ensure stable internet</p>
              </div>
              
              <div className="bg-gray-50 dark:bg-gray-900/30 p-3 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 flex items-start">
                <div className="mt-0.5 mr-2 flex-shrink-0">
                  <svg className="h-4 w-4 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                </div>
                <p className="text-sm text-gray-700 dark:text-gray-300">Verify the video file is not corrupted or protected</p>
              </div>
              
              <div className="bg-gray-50 dark:bg-gray-900/30 p-3 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 flex items-start">
                <div className="mt-0.5 mr-2 flex-shrink-0">
                  <svg className="h-4 w-4 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                  </svg>
                </div>
                <p className="text-sm text-gray-700 dark:text-gray-300">Try a different video format (MP4 is recommended)</p>
              </div>
              
              <div className="bg-gray-50 dark:bg-gray-900/30 p-3 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 flex items-start">
                <div className="mt-0.5 mr-2 flex-shrink-0">
                  <svg className="h-4 w-4 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                </div>
                <p className="text-sm text-gray-700 dark:text-gray-300">Select a different analysis model</p>
              </div>
            </div>
          </div>
          
          {/* Additional logs for error state */}
          {state.error && state.showLogs && filterLogs(state.logs).length > 0 && (
            <CollapsibleSection 
              title="Error Details & Debug Logs" 
              icon={<TerminalSquare className="h-4 w-4" />}
              defaultOpen={false}
            >
              <div className="flex justify-between items-center mb-3">
                <div className="flex items-center">
                  <h4 className="text-sm font-medium bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-300 px-3 py-1 rounded-full flex items-center">
                    <AlertTriangle className="h-3 w-3 mr-1" />
                    Error Logs ({filterLogs(state.logs).length} entries)
                  </h4>
                </div>
                <div className="flex gap-2">
                  <button 
                    onClick={copyLogs}
                    className="text-xs px-3 py-1.5 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 rounded-md flex items-center shadow-sm transition-colors"
                  >
                    <Clipboard className="h-3 w-3 mr-1" />
                    Copy Logs
                  </button>
                </div>
              </div>
              <div 
                ref={logsContainerRef}
                className="bg-gray-50 dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700 max-h-60 overflow-y-auto p-2 shadow-inner"
              >
                {filterLogs(state.logs).map((log, idx) => (
                  <LogEntryItem key={idx} {...log} />
                ))}
              </div>
            </CollapsibleSection>
          )}
        </div>
      )}
    </div>
  );
};

export default AnalysisStep;