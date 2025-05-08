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
  return text
    .replace(/\x1b\[0m/g, '') // Reset color
    .replace(/\x1b\[31m/g, '') // Red
    .replace(/\x1b\[32m/g, '') // Green
    .replace(/\x1b\[33m/g, '') // Yellow
    .replace(/\x1b\[34m/g, ''); // Blue
};

const createTimestamp = () => {
  const now = new Date();
  return `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}`;
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
    info: "border-blue-200 bg-blue-50 dark:border-blue-900 dark:bg-blue-900/10",
    warning: "border-amber-200 bg-amber-50 dark:border-amber-900 dark:bg-amber-900/10",
    success: "border-green-200 bg-green-50 dark:border-green-900 dark:bg-green-900/10",
    error: "border-red-200 bg-red-50 dark:border-red-900 dark:bg-red-900/10"
  };

  const typeIcons = {
    info: <Info className="h-4 w-4 text-blue-500" />,
    warning: <AlertTriangle className="h-4 w-4 text-amber-500" />,
    success: <CheckCircle className="h-4 w-4 text-green-500" />,
    error: <XCircle className="h-4 w-4 text-red-500" />
  };

  return (
    <div className={`border-l-4 px-3 py-2 mb-1 ${typeStyles[type]} transition-all hover:translate-x-1`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {typeIcons[type]}
          <span className="text-sm font-mono">{processLogText(message)}</span>
        </div>
        <span className="text-xs text-gray-500 font-mono">{timestamp}</span>
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
  progress: 0,
  status: "processing",
  statusMessage: "Initializing...",
  error: null,
  elapsedSeconds: 0,
  isAnalyzing: false,
  isCanceled: false,
  isCompleted: false,
  showLogs: true,
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
        progress: 5,
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
        error: "Analysis was canceled by user",
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
  className = ""
}: AnalysisStepProps) => {
  const [state, dispatch] = useReducer(analysisReducer, initialState);
  
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
      
      // Only process progress updates if we're still analyzing and not completed
      if (state.isAnalyzing && !state.isCompleted) {
        const msg = data.message.toLowerCase();
        
        // Update progress based on log content
        if (msg.includes("starting") || msg.includes("initializing")) {
          dispatch({ type: 'UPDATE_PROGRESS', payload: 15 });
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

  // Timer effect
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

  // Poll for process status
  useEffect(() => {
    if (!state.processId || !state.isAnalyzing) return;
    
    const checkStatus = async () => {
      try {
        const response = await fetch(`/api/process/${state.processId}/status`);
        const data = await response.json();
        
        if (!data.active && state.isAnalyzing) {
          dispatch({ type: 'COMPLETE_ANALYSIS' });
          
          // Create a basic result object using file metadata
          const analysisResults: AnalysisResults = {
            names: [], // This would come from the actual API response
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
          
          // Call the completion handler
          onAnalysisComplete(analysisResults);
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
  }, [state.processId, state.isAnalyzing, state.elapsedSeconds, state.progress, onAnalysisComplete, selectedModel, videoFile]);

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
      
      dispatch({ type: 'UPDATE_PROGRESS', payload: 10 });
      
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
      
      dispatch({ type: 'UPDATE_PROGRESS', payload: 25 });
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
    const logText = state.logs.map(log => `[${log.timestamp}] ${log.type.toUpperCase()}: ${log.message}`).join('\n');
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
            className={`text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 ${state.showLogs ? 'text-primary' : ''}`}
            title="Show/Hide Logs"
          >
            <TerminalSquare className="h-5 w-5" />
          </button>
          <div className="flex items-center bg-gray-100 dark:bg-gray-800 rounded px-2 py-1">
            <Clock className="h-4 w-4 text-gray-500 mr-2" />
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
                onClick={() => dispatch({ type: 'CANCEL_ANALYSIS' })} 
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
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 transition-all">
            <div className="mb-4 flex justify-between items-center">
              <StatusBadge status={state.status} message={state.statusMessage} />
            </div>

            <h3 className="text-xl font-semibold mb-4">
              {state.progress < 100 ? "Processing Video" : "Analysis Complete"}
            </h3>

            <div className="w-full mb-4">
              <div className="bg-gray-200 dark:bg-gray-700 rounded-full h-3 overflow-hidden">
                <div
                  className="bg-primary h-3 rounded-full transition-all duration-300 ease-out"
                  style={{ width: `${state.progress}%` }}
                />
              </div>
              
              <div className="flex justify-between w-full mt-1">
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
          </div>

          {/* Analysis Logs */}
          {state.showLogs && (
            <CollapsibleSection 
              title="CLI Output & Analysis Logs" 
              icon={<TerminalSquare className="h-4 w-4" />}
              defaultOpen={true}
            >
              <div className="flex justify-between items-center mb-2">
                <h4 className="text-sm font-medium">Process Log ({state.logs.length} entries)</h4>
                <button 
                  onClick={copyLogs}
                  className="text-xs px-2 py-1 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 rounded flex items-center"
                >
                  <Clipboard className="h-3 w-3 mr-1" />
                  Copy Logs
                </button>
              </div>
              <div 
                ref={logsContainerRef}
                className="bg-gray-50 dark:bg-gray-900 rounded border border-gray-200 dark:border-gray-700 max-h-96 overflow-y-auto p-1"
                style={{ scrollBehavior: 'smooth' }}
              >
                {state.logs.length > 0 ? (
                  state.logs.map((log, idx) => (
                    <LogEntryItem key={idx} {...log} />
                  ))
                ) : (
                  <p className="text-center text-gray-500 py-4">Waiting for process output...</p>
                )}
              </div>
            </CollapsibleSection>
          )}

          {/* Model information */}
          <CollapsibleSection 
            title="Model Information" 
            icon={<BookOpen className="h-4 w-4" />}
            defaultOpen={false}
          >
            <div className="bg-gray-50 dark:bg-[#162032] p-4 rounded-lg border border-gray-200 dark:border-gray-700">
              <h5 className="font-medium mb-3 text-gray-900 dark:text-white">
                Selected Model: {getModelName(selectedModel)}
              </h5>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                {selectedModel === "anep" && "Uses a custom OCR pipeline with post-processing and BERT-based name recognition."}
                {selectedModel === "model1" && "Relies on Google Cloud Vision OCR with enhanced filtering logic."}
                {selectedModel === "model2" && "Llama 4 Maverick, a multimodal model combining vision and text understanding."}
                {selectedModel === "all" && "A combination of all three models utilized for comparison and research purposes."}
              </p>
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
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg overflow-hidden">
          <div className="bg-red-50 dark:bg-red-900/20 px-6 py-4 border-b border-red-100 dark:border-red-900/30">
            <div className="flex items-center">
              <AlertTriangle className="h-6 w-6 text-red-500 mr-3" />
              <h3 className="text-lg font-medium text-red-800 dark:text-red-300">Analysis Failed</h3>
            </div>
          </div>
          <div className="p-6">
            <p className="text-red-700 dark:text-red-400 mb-6">{state.error}</p>
            
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
                <li>Check the server logs for more detailed error information</li>
              </ul>
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
      )}
      
      {/* Additional logs for error state */}
      {state.error && state.showLogs && (
        <CollapsibleSection 
          title="Analysis Logs" 
          icon={<TerminalSquare className="h-4 w-4" />}
          defaultOpen={true}
        >
          <div className="flex justify-between items-center mb-2">
            <h4 className="text-sm font-medium">Process Log ({state.logs.length} entries)</h4>
            <button 
              onClick={copyLogs}
              className="text-xs px-2 py-1 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 rounded flex items-center"
            >
              <Clipboard className="h-3 w-3 mr-1" />
              Copy Logs
            </button>
          </div>
          <div 
            ref={logsContainerRef}
            className="bg-gray-50 dark:bg-gray-900 rounded border border-gray-200 dark:border-gray-700 max-h-60 overflow-y-auto p-1"
          >
            {state.logs.length > 0 ? (
              state.logs.map((log, idx) => (
                <LogEntryItem key={idx} {...log} />
              ))
            ) : (
              <p className="text-center text-gray-500 py-4">No logs available yet</p>
            )}
          </div>
        </CollapsibleSection>
      )}
    </div>
  );
};

export default AnalysisStep;