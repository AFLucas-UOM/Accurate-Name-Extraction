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

interface AnalysisStepProps {
  videoFile: File;
  selectedModel: string;
  onAnalysisComplete: (results: any) => void;
  className?: string;
}

// Analysis result interface
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

// Status badge component
const StatusBadge = ({ status, message }: { status: string; message: string }) => {
  const getStatusColor = (status: string) => {
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

  const getStatusIcon = (status: string) => {
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
    <span className={`px-3 py-1 rounded-full text-sm font-medium flex items-center ${getStatusColor(status)}`}>
      {getStatusIcon(status)}
      {message}
    </span>
  );
};

// Log entry component
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

  // Function to convert ANSI color codes to normal text
  const processText = (text: string) => {
    return text
      .replace(/\x1b\[0m/g, '') // Reset color
      .replace(/\x1b\[31m/g, '') // Red
      .replace(/\x1b\[32m/g, '') // Green
      .replace(/\x1b\[33m/g, '') // Yellow
      .replace(/\x1b\[34m/g, ''); // Blue
  };

  return (
    <div className={`border-l-4 px-3 py-2 mb-1 ${getTypeStyle()} transition-all hover:translate-x-1`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {getTypeIcon()}
          <span className="text-sm font-mono">{processText(message)}</span>
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

const AnalysisStep = ({
  videoFile,
  selectedModel,
  onAnalysisComplete,
  className = ""
}: AnalysisStepProps) => {
  // State variables
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState("processing");
  const [statusMessage, setStatusMessage] = useState("Initializing...");
  const [error, setError] = useState<string | null>(null);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isCanceled, setIsCanceled] = useState(false);
  const [isCompleted, setIsCompleted] = useState(false);
  const [showLogs, setShowLogs] = useState(true); // Show logs by default
  const [logs, setLogs] = useState<LogEntryProps[]>([]);
  const [processId, setProcessId] = useState<string | null>(null);
  
  const logsContainerRef = useRef<HTMLDivElement>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const cancelRef = useRef(false);
  const pollingRef = useRef<NodeJS.Timeout | number | null>(null);

  // Map model id to its display name
  const getModelName = (modelId: string): string => {
    const modelMap: Record<string, string> = {
      "anep": "ANEP (Accurate Name Extraction Pipeline)",
      "model1": "Google Cloud Vision & Gemini 1.5 Pro",
      "model2": "Llama 4 Maverick",
      "all": "All Models (Ensemble)",
    };
    
    return modelMap[modelId] || "Custom Model";
  };

  // Format elapsed seconds as mm:ss
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
      ...prevLogs,
      { message, timestamp, type }
    ]);
  };

  // Auto-scroll logs to bottom when new logs come in
  useEffect(() => {
    if (logsContainerRef.current) {
      logsContainerRef.current.scrollTop = logsContainerRef.current.scrollHeight;
    }
  }, [logs]);

  // Copy logs to clipboard
  const copyLogs = () => {
    const logText = logs.map(log => `[${log.timestamp}] ${log.type.toUpperCase()}: ${log.message}`).join('\n');
    navigator.clipboard.writeText(logText);
    addLog("Logs copied to clipboard", 'success');
  };

  // Get API endpoint based on selected model
  const getApiEndpoint = () => {
    switch(selectedModel) {
      case "anep": return "/api/run/anep";
      case "model1": return "/api/run/gcloud";
      case "model2": return "/api/run/llama";
      case "all": return "/api/run/ensemble";
      default: return "/api/run/anep";
    }
  };

  // Connect to the event source for real-time logs
  useEffect(() => {
    if (!processId) return;
    
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
        
        // Only process logs for our process ID
        if (data.process_id === processId) {
          addLog(data.message, data.type as 'info' | 'warning' | 'success' | 'error');
          
          // More comprehensive progress update based on log content
          const msg = data.message.toLowerCase();
          
          // Initial stages
          if (msg.includes("starting") || msg.includes("initializing") || msg.includes("begin")) {
            setProgress(p => Math.max(p, 15));
          } 
          // Processing stages - more keywords and larger increments
          else if (
            msg.includes("processing") || 
            msg.includes("analyzing") || 
            msg.includes("extracting") || 
            msg.includes("frame") || 
            msg.includes("loading") ||
            msg.includes("reading") ||
            msg.includes("step") ||
            msg.includes("progress")
          ) {
            setProgress(p => {
              // Analyze different progress keywords for different increment amounts
              let increment = 2; // default increment
              
              // Higher increments for more significant progress indicators
              if (msg.includes("50%") || msg.includes("half")) {
                increment = 10;
              } else if (msg.includes("complete")) { 
                increment = 8;
              } else if (msg.includes("progress")) {
                increment = 5;
              }
              
              const newProgress = p + increment;
              return Math.min(newProgress, 95); // Cap at 95% until fully complete
            });
          } 
          // Completion
          else if (msg.includes("completed") || msg.includes("finished") || msg.includes("done") || msg.includes("100%")) {
            setProgress(100);
            setStatus("success");
            setStatusMessage("Analysis completed");
            setIsCompleted(true);
            setIsAnalyzing(false);
          } 
          // Error handling
          else if (msg.includes("error") || msg.includes("failed") || msg.includes("exception")) {
            setStatus("error");
            setStatusMessage("Analysis failed");
            setError(data.message);
            setIsAnalyzing(false);
          }
        }
      } catch (err) {
        console.error("Error parsing SSE data:", err);
      }
    };
    
    // Handle errors
    eventSource.onerror = (err) => {
      console.error("SSE connection error:", err);
      addLog("Lost connection to server log stream", "error");
      eventSource.close();
    };
    
    // Cleanup on unmount
    return () => {
      eventSource.close();
    };
  }, [processId]);
  
  // Poll for process status and gradually increment progress
  useEffect(() => {
    if (!processId || !isAnalyzing) return;
    
    const checkStatus = async () => {
      try {
        const response = await fetch(`/api/process/${processId}/status`);
        const data = await response.json();
        
        if (!data.active && isAnalyzing) {
          // Process has completed
          setProgress(100);
          setIsCompleted(true);
          setIsAnalyzing(false);
          setStatus("success");
          setStatusMessage("Analysis completed");
          
          // Create a basic result object using file metadata
          const analysisResults: AnalysisResults = {
            names: [], // This would come from the actual API response
            model: selectedModel,
            modelName: getModelName(selectedModel),
            processingTime: formatElapsed(elapsedSeconds),
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
        } else if (isAnalyzing) {
          // Gradually increment progress based on time
          setProgress(prevProgress => {
            // Cap progress at 95% until completion
            const timeBasedIncrement = elapsedSeconds > 10 ? 2 : 0;
            return Math.min(prevProgress + timeBasedIncrement, 95);
          });
        }
      } catch (err) {
        console.error("Error checking process status:", err);
      }
    };
    
    // Poll every 5 seconds
    const interval = setInterval(checkStatus, 5000);
    
    return () => clearInterval(interval);
  }, [processId, isAnalyzing, elapsedSeconds, onAnalysisComplete, selectedModel, videoFile]);

  // Run the analysis
  const runAnalysis = useCallback(async () => {
    // Reset all states
    setIsAnalyzing(true);
    setError(null);
    setProgress(5); // Start at 5% to show immediate feedback
    setStatusMessage("Initializing analysis...");
    setStatus("processing");
    setElapsedSeconds(0);
    setIsCompleted(false);
    cancelRef.current = false;
    setIsCanceled(false);
    setLogs([]);
    setProcessId(null);

    const startTime = Date.now();
    
    // Store the timer reference in a variable that will persist outside this function
    const timerRef = setInterval(() => {
      setElapsedSeconds(Math.floor((Date.now() - startTime) / 1000));
    }, 1000);
    
    // Update the global timer reference for cleanup on component unmount
    pollingRef.current = timerRef;

    // Add initial logs
    addLog(`Starting analysis of ${videoFile.name}`, 'info');
    addLog(`Using model: ${getModelName(selectedModel)}`, 'info');

    try {
      // Make the API call to the appropriate endpoint
      setStatusMessage(`Processing with ${getModelName(selectedModel)}...`);
      addLog(`Contacting server at ${getApiEndpoint()}`, 'info');
      
      setProgress(10);
      
      // Make the API call to start processing
      const response = await fetch(getApiEndpoint(), {
        method: 'POST',
        headers: {
          'Accept': 'application/json'
        }
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        clearInterval(timerRef); // Clean up timer on error
        throw new Error(errorText || 'Failed to process video');
      }
      
      // Get the response
      const result = await response.json();
      
      // Store the process ID for streaming logs
      if (result.process_id) {
        setProcessId(result.process_id);
        addLog(`Server processing started with ID: ${result.process_id}`, 'success');
      } else {
        clearInterval(timerRef); // Clean up timer on error
        throw new Error('Server did not return a process ID');
      }
      
      setProgress(25);
      setStatusMessage("Server is processing your video...");
      
      // We'll let the event source and polling handle progress updates from here
      // The timer will keep running until completion or component unmount
      
    } catch (err: any) {
      console.error("Analysis error:", err);
      setError(err.message || "An error occurred during analysis. Please try again.");
      setStatus("error");
      setStatusMessage("Analysis failed");
      addLog(err.message || "Analysis failed with an error", "error");
      setIsAnalyzing(false);
      
      // Timer is already cleared in the try block on error
    }
    // No finally block that clears the timer - it will keep running until completion or component unmount
  }, [videoFile, selectedModel, onAnalysisComplete]);

  // Run analysis when component mounts
  useEffect(() => {
    runAnalysis();
    
    // Cleanup function
    return () => {
      cancelRef.current = true;
      if (pollingRef.current !== null) {
        window.clearInterval(pollingRef.current);
      }
      
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, [runAnalysis]);

  // Handle cancel confirmation modal
  const [showCancelModal, setShowCancelModal] = useState(false);

  const confirmCancel = () => {
    cancelRef.current = true;
    setIsCanceled(true);
    setError("Analysis was canceled by user");
    setStatus("error");
    setStatusMessage("Analysis canceled");
    setShowCancelModal(false);
    setIsAnalyzing(false);
    
    // Close event source
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }
    
    // Clean up timer
    if (pollingRef.current !== null) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
  };
  
  const handleCancel = () => {
    setShowCancelModal(true);
  };
  
  // Handle retry
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
            className={`text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 ${showLogs ? 'text-primary' : ''}`}
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
              <StatusBadge status={status} message={statusMessage} />
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
                <p className="text-xs text-gray-500">{Math.round(progress)}% complete</p>
                <p className="text-xs text-gray-500">{isCompleted ? "Completed" : "Processing..."}</p>
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

          {/* Analysis Logs with CLI output integrated */}
          {showLogs && (
            <CollapsibleSection 
              title="CLI Output & Analysis Logs" 
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
              <div 
                ref={logsContainerRef}
                className="bg-gray-50 dark:bg-gray-900 rounded border border-gray-200 dark:border-gray-700 max-h-96 overflow-y-auto p-1"
                style={{ scrollBehavior: 'smooth' }}
              >
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
                  <p className="text-center text-gray-500 py-4">Waiting for process output...</p>
                )}
              </div>
            </CollapsibleSection>
          )}

          {/* About the analysis */}
          <CollapsibleSection 
            title="Model Information" 
            icon={<BookOpen className="h-4 w-4" />}
            defaultOpen={false}
          >
            <div className="space-y-4">
              {/* Model Summary */}
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
                <li>Check the server logs for more detailed error information</li>
              </ul>
            </div>
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
      {error && showLogs && (
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
          <div 
            ref={logsContainerRef}
            className="bg-gray-50 dark:bg-gray-900 rounded border border-gray-200 dark:border-gray-700 max-h-60 overflow-y-auto p-1"
          >
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
    </div>
  );
};

export default AnalysisStep;