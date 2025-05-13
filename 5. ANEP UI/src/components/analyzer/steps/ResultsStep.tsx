import { useState, useEffect } from "react";
import { Download, RefreshCw, Copy, Maximize2, X, Calendar, Clock, FileType, HardDrive, ChevronRight, CheckCircle, Tag, Info, Code, Eye, EyeOff, Film, Database, Loader2, AlertCircle, Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Input } from "@/components/ui/input";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

interface NameDetection {
  name: string;
  confidence: number;
  timestamp: string;
  frames?: number[];
  method?: string[];
  first_seen?: string;
  last_seen?: string;
  duration?: number;
  aliases?: string[];
  count?: number;
  mentions?: number;
  first_appearance?: string;
  last_appearance?: string;
}

interface VideoMetadata {
  filename: string;
  size: string;
  duration: string;
  type: string;
  frameRate?: string;
  fps?: number;
}

interface AnalysisResults {
  names: NameDetection[];
  model: string;
  modelName: string;
  processingTime: string;
  videoMetadata: VideoMetadata;
  analysisDate: string;
  totalFrames?: number;
  uniqueNames?: number;
  source?: string;
  processingTimeSeconds?: number;
}

interface ApiData {
  // ANEP format
  folder?: string;
  file?: string;
  unique_names?: number;
  total_instances?: number;
  people?: Array<{
    name: string;
    aliases?: string[];
    mentions?: number;
    first_seen?: string;
    last_seen?: string;
    duration?: number;
    active_periods?: Array<{
      start: string;
      end: string;
      duration: number;
    }>;
    // Google Cloud / LLaMA format
    first_appearance?: string;
    last_appearance?: string;
    count?: number;
    confidence?: number;
  }>;
  // Google Cloud specific
  source?: string;
  total_frames?: number;
  distinct_frames?: number;
  frames_with_text?: number;
  processing_time?: number;
  processing_time_seconds?: number;
  duration?: number;
  // LLaMA specific
  video_info?: {
    fps?: number;
    duration_seconds?: number;
    filename?: string;
  };
  processing_stats?: {
    processing_time_seconds?: number;
  };
  api_stats?: any;
  error?: string;
}

interface ComparisonData {
  anep?: ApiData;
  gcloud?: ApiData;
  llama?: ApiData;
  summary?: {
    total_people_found?: number;
    common_people?: string[];
    unique_to_anep?: string[];
    unique_to_gcloud?: string[];
    unique_to_llama?: string[];
    processing_times?: {
      anep?: string | number;
      gcloud?: string | number;
      llama?: string | number;
    };
  };
}

interface ResultsStepProps {
  results?: AnalysisResults;
  model?: string;
  onRestart: () => void;
  className?: string;
  videoMetadata?: VideoMetadata;
  processId?: string;
}

const ResultsStep = ({
  results: initialResults,
  model: modelProp,
  onRestart,
  className = "",
  videoMetadata: parentVideoMetadata,
  processId,
}: ResultsStepProps) => {
  const { toast } = useToast();
  const [activeTab, setActiveTab] = useState("names");
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [selectedName, setSelectedName] = useState<NameDetection | null>(null);
  const [jsonFilter, setJsonFilter] = useState("");
  const [showLineNumbers, setShowLineNumbers] = useState(true);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<AnalysisResults | null>(initialResults || null);
  const [enhancedMetadata, setEnhancedMetadata] = useState<VideoMetadata | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [videoDurationSeconds, setVideoDurationSeconds] = useState<number | null>(null);

  // Extract model from either prop or results object
  const model = modelProp || initialResults?.model || localStorage.getItem('selectedModel') || 'anep';

  // Load video duration from localStorage on mount
  useEffect(() => {
    const storedDuration = localStorage.getItem("vid_dur");
    if (storedDuration) {
      const duration = parseFloat(storedDuration);
      if (!isNaN(duration)) {
        setVideoDurationSeconds(duration);
      }
    }
  }, []);

  // Check if process is still running
  useEffect(() => {
    if (processId) {
      const checkProcessStatus = async () => {
        try {
          const response = await fetch(`http://localhost:5050/api/process/${processId}/status`);
          const data = await response.json();
          setIsProcessing(data.active);
        } catch (error) {
          console.error("Error checking process status:", error);
          setIsProcessing(false);
        }
      };

      const interval = setInterval(checkProcessStatus, 2000);
      checkProcessStatus();

      return () => clearInterval(interval);
    }
  }, [processId]);

  // Fetch results from API based on model
  useEffect(() => {
    // If we already have results, use them
   if (
      initialResults &&
      Array.isArray(initialResults.names) &&
      initialResults.names.length > 0
    ) {
      setResults(initialResults);
      setEnhancedMetadata(getEnhancedVideoMetadata(initialResults.videoMetadata));
      setIsLoading(false);
      return;
    }

    const fetchResults = async () => {
      setIsLoading(true);
      setError(null);

      if (!model) {
        setError("No model specified for analysis");
        setIsLoading(false);
        return;
      }

      try {
        let apiUrl = "";
        switch (model) {
          case "anep":
            apiUrl = "http://localhost:5050/api/anep/latest-results";
            break;
          case "model1":
            apiUrl = "http://localhost:5050/api/gcloud/latest-results";
            break;
          case "model2":
            apiUrl = "http://localhost:5050/api/llama/latest-results";
            break;
          case "all":
            apiUrl = "http://localhost:5050/api/results/compare";
            break;
          default:
            console.warn(`Unexpected model value: ${model}, defaulting to ANEP`);
            apiUrl = "http://localhost:5050/api/anep/latest-results";
        }

        console.log("API URL:", apiUrl);
        const response = await fetch(apiUrl);
        
        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(errorData.error || `Failed to fetch results: ${response.statusText}`);
        }

        const data = await response.json();
        
        let transformedResults: AnalysisResults;
        
        if (model === "all") {
          const comparisonData = data as ComparisonData;
          const allNames: NameDetection[] = [];
          
          if (comparisonData.anep?.people) {
            comparisonData.anep.people.forEach((person) => {
              allNames.push({
                name: person.name,
                confidence: 0.95,
                timestamp: person.first_seen || "Unknown",
                method: ["ANEP"],
                aliases: person.aliases,
                duration: person.duration,
                first_seen: person.first_seen,
                last_seen: person.last_seen,
                mentions: person.mentions,
              });
            });
          }
          
          if (comparisonData.gcloud?.people) {
            comparisonData.gcloud.people.forEach((person) => {
              allNames.push({
                name: person.name,
                confidence: 0.9,
                timestamp: person.first_appearance || "Unknown",
                method: ["Google Cloud Vision"],
                count: person.count,
                first_appearance: person.first_appearance,
                last_appearance: person.last_appearance,
              });
            });
          }
          
          if (comparisonData.llama?.people) {
            comparisonData.llama.people.forEach((person) => {
              allNames.push({
                name: person.name,
                confidence: 0.85,
                timestamp: person.first_appearance || "Unknown",
                method: ["LLaMA"],
                count: person.count,
                first_appearance: person.first_appearance,
                last_appearance: person.last_appearance,
              });
            });
          }
          
          const uniqueNames = Array.from(
            new Map(allNames.map(item => [`${item.name}-${item.timestamp}`, item])).values()
          );
          
          const processingTimes = comparisonData.summary?.processing_times || {};
          const totalTime = Object.values(processingTimes)
            .map(time => typeof time === 'number' ? time : 0)
            .reduce((acc: number, time) => acc + time, 0);
          
          transformedResults = {
            names: uniqueNames,
            model: "all",
            modelName: "All Models (Ensemble)",
            processingTime: totalTime > 0 ? `${totalTime.toFixed(2)} seconds (combined)` : "Unknown",
            videoMetadata: getEnhancedVideoMetadata(parentVideoMetadata),
            analysisDate: new Date().toISOString(),
            uniqueNames: comparisonData.summary?.total_people_found || uniqueNames.length,
          };
        } else {
          const apiData = data as ApiData;
          const names: NameDetection[] = [];
          let processingTime = "Unknown";
          let metadata = getEnhancedVideoMetadata(parentVideoMetadata);
          let totalFrames = 0;
          let uniqueNames = 0;
          let processingTimeSeconds = 0;
          
          // If API provides video duration, use it to update localStorage
          if (apiData.video_info?.duration_seconds) {
            localStorage.setItem("vid_dur", apiData.video_info.duration_seconds.toString());
            setVideoDurationSeconds(apiData.video_info.duration_seconds);
          } else if (apiData.duration) {
            localStorage.setItem("vid_dur", apiData.duration.toString());
            setVideoDurationSeconds(apiData.duration);
          }
          
          switch (model) {
            case "anep":
              console.log("Processing ANEP data, people:", apiData.people);
              if (apiData.people && Array.isArray(apiData.people)) {
                apiData.people.forEach((person) => {
                  names.push({
                    name: person.name,
                    confidence: 0.95,
                    timestamp: person.first_seen || "Unknown",
                    first_seen: person.first_seen,
                    last_seen: person.last_seen,
                    duration: person.duration,
                    aliases: person.aliases,
                    method: ["ANEP"],
                    mentions: person.mentions,
                  });
                });
              }
              processingTime = apiData.processing_time_seconds !== undefined
                ? `${apiData.processing_time_seconds.toFixed(2)} seconds`
                : "Unknown";
              processingTimeSeconds = apiData.processing_time_seconds || 0;
              totalFrames = apiData.total_instances || 0;
              uniqueNames = apiData.unique_names || names.length;

              console.log("ANEP names array:", names);
              break;
              
            case "model1": // Google Cloud
              console.log("Processing Google Cloud data, people:", apiData.people);
              if (apiData.people && Array.isArray(apiData.people)) {
                apiData.people.forEach((person) => {
                  const confidence = person.confidence || 0.9;
                  names.push({
                    name: person.name,
                    confidence: confidence,
                    timestamp: person.first_appearance || "Unknown",
                    method: ["Google Cloud Vision"],
                    count: person.count,
                    first_appearance: person.first_appearance,
                    last_appearance: person.last_appearance,
                  });
                });
              }
              processingTime = apiData.processing_time 
                ? `${apiData.processing_time.toFixed(2)} seconds` 
                : "Unknown";
              processingTimeSeconds = apiData.processing_time || 0;
              totalFrames = apiData.total_frames || 0;
              uniqueNames = apiData.people?.length || 0;
              
              if (apiData.distinct_frames) {
                metadata = { ...metadata, frameRate: `${apiData.distinct_frames} distinct frames` };
              }
              break;
              
            case "model2": // LLaMA
              console.log("Processing LLaMA data, people:", apiData.people);
              if (apiData.people && Array.isArray(apiData.people)) {
                apiData.people.forEach((person) => {
                  const confidence = person.confidence || 0.85;
                  names.push({
                    name: person.name,
                    confidence: confidence,
                    timestamp: person.first_appearance || "Unknown",
                    method: ["LLaMA"],
                    count: person.count,
                    first_appearance: person.first_appearance,
                    last_appearance: person.last_appearance,
                  });
                });
              }
              processingTime = apiData.processing_stats?.processing_time_seconds 
                ? `${apiData.processing_stats.processing_time_seconds.toFixed(2)} seconds` 
                : "Unknown";
              processingTimeSeconds = apiData.processing_stats?.processing_time_seconds || 0;
              uniqueNames = apiData.people?.length || 0;
              
              if (apiData.video_info) {
                metadata = {
                  ...metadata,
                  filename: apiData.video_info.filename || metadata.filename,
                  fps: apiData.video_info.fps,
                };
              }
              break;
          }
          
          transformedResults = {
            names,
            model,
            modelName: getModelName(model),
            processingTime,
            videoMetadata: metadata,
            analysisDate: new Date().toISOString(),
            totalFrames,
            uniqueNames,
            source: apiData.source,
            processingTimeSeconds,
          };

          console.log("Transformed results:", transformedResults);
        }
        
        setResults(transformedResults);
        setEnhancedMetadata(getEnhancedVideoMetadata(transformedResults.videoMetadata));
        
      } catch (err) {
        console.error("Error fetching results:", err);
        setError(err instanceof Error ? err.message : "An error occurred while fetching results");
        
        if (isProcessing) {
          setError("Analysis is still in progress. Please wait for it to complete before viewing results.");
        }
      } finally {
        setIsLoading(false);
      }
    };

    if (isProcessing) {
      setIsLoading(true);
      setError("Analysis is still in progress. Waiting for completion...");
      return;
    }

    fetchResults();
  }, [model, parentVideoMetadata, isProcessing, initialResults]);

    // Helper functions
    const getDefaultVideoMetadata = (): VideoMetadata => {
    const storedMetadata = localStorage.getItem("video_metadata");
    if (storedMetadata) {
      try {
        const parsed = JSON.parse(storedMetadata);
        if (videoDurationSeconds !== null) {
          parsed.duration = formatDuration(videoDurationSeconds);
        }
        return parsed;
      } catch (e) {
        console.error("Failed to parse stored metadata:", e);
      }
    }

    // Fallback — currently returns "Unknown"
    return {
      filename: localStorage.getItem("CurrentVideoName") || "Unknown",
      size: localStorage.getItem("vid_siz") || "Unknown",
      duration: videoDurationSeconds !== null ? formatDuration(videoDurationSeconds) : "Unknown",
      type: localStorage.getItem("video_type") || "video/mp4",
    };
  };


  const getEnhancedVideoMetadata = (providedMetadata?: VideoMetadata): VideoMetadata => {
    const baseMetadata = providedMetadata || getDefaultVideoMetadata();
    
    // Always use vid_dur from localStorage if available
    if (videoDurationSeconds !== null) {
      baseMetadata.duration = formatDuration(videoDurationSeconds);
    } else {
      // Fallback to stored duration
      const storedDuration = localStorage.getItem("vid_dur");
      if (storedDuration) {
        const durationInSeconds = parseFloat(storedDuration);
        if (!isNaN(durationInSeconds)) {
          baseMetadata.duration = formatDuration(durationInSeconds);
        }
      }
    }

    return baseMetadata;
  };

  const getModelName = (modelId: string): string => {
    const modelMap: Record<string, string> = {
      "anep": "ANEP (Accurate Name Extraction Pipeline)",
      "model1": "Google Cloud Vision & Gemini 1.5 Pro",
      "model2": "Llama 4 Maverick",
      "all": "All Models (Ensemble)",
    };
    
    return modelMap[modelId] || "Unknown Model";
  };

  const formatDuration = (seconds: number): string => {
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    return `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const formatFileSize = (sizeString: string): string => {
    if (typeof sizeString === 'string' && (sizeString.includes('MB') || sizeString.includes('KB'))) {
      return sizeString;
    }
    
    const bytes = parseFloat(sizeString);
    if (isNaN(bytes)) return sizeString;
    
    if (bytes < 1024) return bytes + " MB";
    else if (bytes < 1048576) return (bytes / 1024).toFixed(2) + " KB";
    else return (bytes / 1048576).toFixed(2) + " MB";
  };

  const formatTimestamp = (timestamp: string) => {
    if (/^\d{2}:\d{2}:\d{2}$/.test(timestamp)) {
      return timestamp;
    }
    
    const seconds = parseFloat(timestamp);
    if (!isNaN(seconds)) {
      const hrs = Math.floor(seconds / 3600);
      const mins = Math.floor((seconds % 3600) / 60);
      const secs = Math.floor(seconds % 60);
      
      return `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    
    return timestamp;
  };

  // Loading state
  if (isLoading) {
    return (
      <div className={`w-full ${className}`}>
        <div className="flex flex-col items-center justify-center py-12">
          <Loader2 className="h-8 w-8 animate-spin text-primary mb-4" />
          <p className="text-lg font-medium">
            {isProcessing ? "Analysis in progress..." : "Loading results..."}
          </p>
          <p className="text-sm text-muted-foreground mt-1">
            {isProcessing 
              ? "The model is still processing your video. This may take a few minutes."
              : `Fetching analysis data for ${getModelName(model)}`
            }
          </p>
          {isProcessing && (
            <Button 
              variant="outline" 
              onClick={onRestart}
              className="mt-4"
            >
              <RefreshCw className="h-4 w-4 mr-2" />
              Cancel and Start Over
            </Button>
          )}
        </div>
      </div>
    );
  }

  // Error state
  if (error || !results) {
    return (
      <div className={`w-full ${className}`}>
        <Alert variant={isProcessing ? "default" : "destructive"}>
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>{isProcessing ? "Processing" : "Error"}</AlertTitle>
          <AlertDescription>
            <div className="flex flex-col gap-3">
              <p>{error || "No results available"}</p>
              {isProcessing && (
                <p className="text-sm">
                  The analysis is still running. Results will appear when processing is complete.
                </p>
              )}
              <Button 
                variant="outline" 
                onClick={onRestart}
                className="w-fit"
              >
                <RefreshCw className="h-4 w-4 mr-2" />
                {isProcessing ? "Cancel and Start Over" : "Try Again"}
              </Button>
            </div>
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  // Helper function to clean results
  const getCleanResults = (results: AnalysisResults, metadata: VideoMetadata) => {
    const cleanedMetadata = { ...metadata };
    delete (cleanedMetadata as any)["resolution"];

    const filteredResults = {
      ...results,
      videoMetadata: cleanedMetadata
    };

    return JSON.parse(
      JSON.stringify(filteredResults, (key, value) => {
        if (value === undefined) return undefined;
        return value;
      })
    );
  };

  const handleDownload = () => {
    const cleanResults = getCleanResults(results, enhancedMetadata!);
    const jsonBlob = new Blob([JSON.stringify(cleanResults, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(jsonBlob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${model}-analysis-results-${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    
    toast({
      title: "Results downloaded",
      description: "The results have been saved as a JSON file.",
    });
  };

  const handleCopyToClipboard = () => {
    const cleanResults = getCleanResults(results, enhancedMetadata!);
    const formattedJson = JSON.stringify(cleanResults, null, 2);
    
    navigator.clipboard.writeText(formattedJson)
      .then(() => {
        toast({
          title: "✨ Copied to clipboard",
          description: "Results data has been copied to your clipboard.",
        });
      })
      .catch(() => {
        toast({
          title: "Copy failed",
          description: "Failed to copy results to clipboard.",
          variant: "destructive",
        });
      });
  };

  const handleRowClick = (name: NameDetection) => {
    setSelectedName(name);
  };

  const toggleFullscreen = () => {
    setIsFullscreen(!isFullscreen);
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.9) return "bg-green-500";
    if (confidence >= 0.7) return "bg-blue-500";
    if (confidence >= 0.5) return "bg-yellow-500";
    return "bg-red-500";
  };

  // Tab component
  const Tab = ({ id, label, active, onClick, icon }: { id: string, label: string, active: boolean, onClick: () => void, icon?: React.ReactNode }) => (
    <button
      className={`px-4 py-3 text-sm font-medium transition-colors flex items-center gap-2 ${
        active 
          ? "text-primary border-b-2 border-primary" 
          : "text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-800/50"
      }`}
      onClick={onClick}
    >
      {icon}
      {label}
    </button>
  );

  // Fullscreen modal component
  const FullscreenModal = () => {
    if (!isFullscreen) return null;

    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm overflow-auto p-4">
        <div className="bg-white dark:bg-[#1e293b] rounded-xl shadow-xl w-full max-w-7xl max-h-[90vh] flex flex-col overflow-hidden">
          <div className="p-4 border-b dark:border-gray-700 flex items-center justify-between">
            <h3 className="text-xl font-semibold text-gray-900 dark:text-white">Names Analysis Results</h3>
            <Button
              variant="ghost"
              size="icon"
              onClick={toggleFullscreen}
              className="hover:bg-red-50 dark:hover:bg-red-900/20"
            >
              <X className="h-5 w-5 text-red-500" />
            </Button>
          </div>
          
          <div className="overflow-auto p-6 flex-1 bg-gray-50 dark:bg-[#111827]">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="md:col-span-2 bg-white dark:bg-[#1e293b] border border-gray-200 dark:border-gray-700 rounded-lg shadow-sm overflow-hidden">
                <div className="p-4 border-b border-gray-200 dark:border-gray-700 flex justify-between items-center">
                  <h4 className="font-semibold text-gray-900 dark:text-white">Names Detected</h4>
                  <Badge variant="outline" className="flex items-center gap-1 text-gray-700 dark:text-gray-300 dark:border-gray-600">
                    <Tag className="h-3 w-3" />
                    <span>{results.names.length} names</span>
                  </Badge>
                </div>
                <div className="overflow-x-auto max-h-[60vh]">
                  <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                    <thead className="bg-gray-50 dark:bg-[#1a2234] sticky top-0">
                      <tr>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                          Name
                        </th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                          Timestamp
                        </th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                          Confidence
                        </th>
                        {model === "all" && (
                          <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                            Source
                          </th>
                        )}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-200 dark:divide-gray-700 bg-white dark:bg-[#1e293b]">
                      {results.names.map((entry, index) => (
                        <tr 
                          key={index} 
                          className={`hover:bg-gray-50 dark:hover:bg-[#2d3c56] cursor-pointer transition-colors ${
                            selectedName === entry ? "bg-blue-50 dark:bg-blue-900/30" : ""
                          }`}
                          onClick={() => handleRowClick(entry)}
                        >
                          <td className="px-4 py-3 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white">
                            {entry.name}
                          </td>
                          <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                            {formatTimestamp(entry.timestamp)}
                          </td>
                          <td className="px-4 py-3 whitespace-nowrap text-sm">
                            <div className="flex items-center">
                              <div className="w-24 bg-gray-200 dark:bg-gray-700 rounded-full h-2 mr-2">
                                <div
                                  className={`${getConfidenceColor(entry.confidence)} h-2 rounded-full`}
                                  style={{ width: `${entry.confidence * 100}%` }}
                                ></div>
                              </div>
                              <span className="font-mono text-gray-700 dark:text-gray-300">{(entry.confidence * 100).toFixed(0)}%</span>
                            </div>
                          </td>
                          {model === "all" && (
                            <td className="px-4 py-3 whitespace-nowrap text-sm">
                              <Badge variant="outline" className="text-xs">
                                {entry.method?.join(", ") || "Unknown"}
                              </Badge>
                            </td>
                          )}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
              
              <div className="bg-white dark:bg-[#1e293b] border border-gray-200 dark:border-gray-700 rounded-lg shadow-sm p-4">
                {selectedName ? (
                  <div className="space-y-4">
                    <div className="flex items-start justify-between">
                      <h4 className="text-lg font-semibold text-gray-900 dark:text-white">{selectedName.name}</h4>
                      <Badge className={`${selectedName.confidence >= 0.8 ? 'bg-green-500' : selectedName.confidence >= 0.6 ? 'bg-blue-500' : 'bg-yellow-500'} text-white`}>
                        {(selectedName.confidence * 100).toFixed(0)}% confidence
                      </Badge>
                    </div>
                    
                    <div className="space-y-3 pt-2">
                      <div className="flex items-center gap-2">
                        <Clock className="h-4 w-4 text-gray-500 dark:text-gray-400" />
                        <span className="text-sm text-gray-700 dark:text-gray-300">
                          Appeared at: <span className="font-mono">
                            {formatTimestamp(selectedName.first_appearance || selectedName.first_seen || selectedName.timestamp)}
                          </span>
                        </span>
                      </div>
                      
                      {(selectedName.first_seen || selectedName.first_appearance) && (selectedName.last_seen || selectedName.last_appearance) && (
                        <div className="pt-2 space-y-2">
                          <div className="flex items-center gap-2">
                            <span className="text-sm text-gray-700 dark:text-gray-300">
                              First seen: <span className="font-mono">
                                {formatTimestamp(selectedName.first_seen || selectedName.first_appearance || "")}
                              </span>
                            </span>
                          </div>
                          <div className="flex items-center gap-2">
                            <span className="text-sm text-gray-700 dark:text-gray-300">
                              Last seen: <span className="font-mono">
                                {formatTimestamp(selectedName.last_seen || selectedName.last_appearance || "")}
                              </span>
                            </span>
                          </div>
                          {selectedName.duration && (
                            <div className="flex items-center gap-2">
                              <span className="text-sm text-gray-700 dark:text-gray-300">
                                Duration: <span className="font-mono">{selectedName.duration.toFixed(2)}s</span>
                              </span>
                            </div>
                          )}
                        </div>
                      )}
                      
                      {(selectedName.count || selectedName.mentions) && (
                        <div className="flex items-center gap-2">
                          <span className="text-sm text-gray-700 dark:text-gray-300">
                            Occurrences: <span className="font-mono">{selectedName.count || selectedName.mentions}</span>
                          </span>
                        </div>
                      )}
                      
                      {selectedName.aliases && selectedName.aliases.length > 0 && (
                        <div className="pt-2">
                          <div className="flex items-center gap-2 mb-2">
                            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                              Aliases ({selectedName.aliases.length})
                            </span>
                          </div>
                          <div className="flex flex-wrap gap-1">
                            {selectedName.aliases.map((alias, idx) => (
                              <Badge key={idx} variant="outline" className="text-xs dark:border-gray-600 dark:text-gray-300">
                                {alias}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}
                      
                      {selectedName.method && (
                        <div className="pt-2">
                          <div className="flex items-center gap-2">
                            <span className="text-sm text-gray-700 dark:text-gray-300">
                              Detected by: <Badge variant="outline">{selectedName.method.join(", ")}</Badge>
                            </span>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                ) : (
                  <div className="h-full flex flex-col items-center justify-center text-center py-10 text-gray-500 dark:text-gray-400">
                    <div className="w-16 h-16 rounded-full bg-gray-100 dark:bg-gray-800 flex items-center justify-center mb-4">
                      <ChevronRight className="h-8 w-8" />
                    </div>
                    <h4 className="text-base font-medium mb-2 text-gray-700 dark:text-gray-300">Select a name from the table</h4>
                    <p className="text-sm max-w-xs">Click on any row in the table to see detailed information about the detected name.</p>
                  </div>
                )}
              </div>
            </div>

            <div className="bg-white dark:bg-[#1e293b] border border-gray-200 dark:border-gray-700 rounded-lg shadow-sm mt-6 p-5">
              <h4 className="font-medium text-lg mb-4 flex items-center text-gray-900 dark:text-white">
                <HardDrive className="h-5 w-5 mr-2 text-gray-500 dark:text-gray-400" />
                <span>File and Model Information</span>
              </h4>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-3">
                  <h5 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Video Metadata</h5>
                  <dl className="grid grid-cols-2 gap-2">
                    <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">Filename:</dt>
                    <dd className="text-sm text-gray-700 dark:text-gray-300 truncate">{enhancedMetadata?.filename}</dd>
                    
                    <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">File Size:</dt>
                    <dd className="text-sm text-gray-700 dark:text-gray-300">{formatFileSize(enhancedMetadata?.size || "Unknown")}</dd>

                    <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">Duration:</dt>
                    <dd className="text-sm text-gray-700 dark:text-gray-300">{enhancedMetadata?.duration}</dd>

                    <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">File Type:</dt>
                    <dd className="text-sm text-gray-700 dark:text-gray-300">{enhancedMetadata?.type}</dd>
                    
                    {enhancedMetadata?.fps && (
                      <>
                        <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">Frame Rate:</dt>
                        <dd className="text-sm text-gray-700 dark:text-gray-300">{enhancedMetadata.fps} FPS</dd>
                      </>
                    )}
                  </dl>
                </div>
                
                <div className="space-y-3">
                  <h5 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Analysis Information</h5>
                  <dl className="grid grid-cols-2 gap-2">
                    <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">Model Used:</dt>
                    <dd className="text-sm text-gray-700 dark:text-gray-300">{getModelName(results.model)}</dd>
                    
                    <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">Processing Time:</dt>
                    <dd className="text-sm text-gray-700 dark:text-gray-300">{results.processingTime}</dd>
                    
                    <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">Names Found:</dt>
                    <dd className="text-sm text-gray-700 dark:text-gray-300">{results.names.length}</dd>
                    
                    {results.uniqueNames !== undefined && (
                      <>
                        <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">Unique Names:</dt>
                        <dd className="text-sm text-gray-700 dark:text-gray-300">{results.uniqueNames}</dd>
                      </>
                    )}
                    
                    {results.totalFrames && (
                      <>
                        <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">Total Frames:</dt>
                        <dd className="text-sm text-gray-700 dark:text-gray-300">{results.totalFrames}</dd>
                      </>
                    )}
                    
                    <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">Analysed On:</dt>
                    <dd className="text-sm text-gray-700 dark:text-gray-300">{new Date(results.analysisDate).toLocaleString()}</dd>
                  </dl>
                </div>
              </div>
            </div>
          </div>
          
          <div className="p-4 border-t dark:border-gray-700 flex justify-end gap-3 bg-white dark:bg-[#1e293b]">
            <Button 
              variant="outline" 
              className="flex items-center gap-2 border-gray-200 dark:border-gray-700 dark:text-gray-300 dark:hover:bg-gray-700"
              onClick={handleCopyToClipboard}
            >
              <Copy className="h-4 w-4" />
              <span>Copy JSON</span>
            </Button>
            <Button 
              variant="default" 
              className="flex items-center gap-2"
              onClick={handleDownload}
            >
              <Download className="h-4 w-4" />
              <span>Download Results</span>
            </Button>
          </div>
        </div>
      </div>
    );
  };

  // Helper function for the JSON tab to prepare JSON data
  const prepareJsonData = () => {
    const cleanedResults = getCleanResults(results, enhancedMetadata!);
    const jsonString = JSON.stringify(cleanedResults, null, 2);
    const lines = jsonString.split('\n');
    
    // Find the last non-empty line
    let lastNonEmptyIndex = lines.length - 1;
    while (lastNonEmptyIndex >= 0 && lines[lastNonEmptyIndex].trim() === '') {
      lastNonEmptyIndex--;
    }
    
    // Trim empty lines from the end
    const trimmedLines = lines.slice(0, lastNonEmptyIndex + 1);
    
    return {
      trimmedJsonString: trimmedLines.join('\n'),
      trimmedLines: trimmedLines,
      originalLength: jsonString.length
    };
  };

  // Main component render
  return (
    <div className={`w-full ${className}`}>
      <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-6">
        <div>
          <h2 className="text-2xl font-bold mb-2">Analysis Results</h2>
          <p className="text-muted-foreground">
            {results.names.length} names detected{results.uniqueNames ? ` (${results.uniqueNames} unique)` : ""} in {enhancedMetadata?.filename}
          </p>
          {model === "all" && (
            <div className="flex items-center gap-2 mt-2">
              <Badge variant="outline" className="text-xs">ANEP</Badge>
              <Badge variant="outline" className="text-xs">Google Cloud</Badge>
              <Badge variant="outline" className="text-xs">LLaMA</Badge>
            </div>
          )}
        </div>
        <div className="flex gap-3 mt-4 md:mt-0">
          <Button
            variant="outline"
            size="sm"
            className="flex items-center gap-2"
            onClick={onRestart}
          >
            <RefreshCw className="h-4 w-4" />
            <span>New Analysis</span>
          </Button>
          <Button
            variant="default"
            size="sm"
            className="flex items-center gap-2"
            onClick={handleDownload}
          >
            <Download className="h-4 w-4" />
            <span>Download</span>
          </Button>
        </div>
      </div>

      <div className="bg-white dark:bg-gray-800 border rounded-lg shadow-sm overflow-hidden">
        <div className="border-b">
          <div className="flex">
            <Tab
              id="names"
              label="Names"
              icon={<Tag className="h-4 w-4" />}
              active={activeTab === "names"}
              onClick={() => setActiveTab("names")}
            />
            <Tab
              id="json"
              label="JSON Data"
              icon={<Code className="h-4 w-4" />}
              active={activeTab === "json"}
              onClick={() => setActiveTab("json")}
            />
            <Tab
              id="metadata"
              label="Metadata"
              icon={<Info className="h-4 w-4" />}
              active={activeTab === "metadata"}
              onClick={() => setActiveTab("metadata")}
            />
          </div>
        </div>

        <div className="p-4">
          {activeTab === "names" && (
            <div className="animate-fade-in">
              <div className="mb-4 flex items-center justify-between">
                <h3 className="text-lg font-medium flex items-center">
                  <CheckCircle className="h-4 w-4 text-green-500 mr-2" />
                    <span>
                      {results.names.length} {results.names.length === 1 ? "Name" : "Names"} Detected
                    </span>
                </h3>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={toggleFullscreen}
                  className="flex items-center gap-2"
                >
                  <Maximize2 className="h-4 w-4" />
                  <span>Fullscreen</span>
                </Button>
              </div>

              {results.names.length === 0 ? (
                <div className="text-center py-8 text-gray-500">
                  No names were detected in this video.
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                    <thead className="bg-gray-50 dark:bg-gray-800">
                      <tr>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                          Name
                        </th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                          First Seen
                        </th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                          Confidence
                        </th>
                        {model === "all" && (
                          <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                            Source
                          </th>
                        )}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                      {results.names.map((entry, index) => (
                        <tr 
                          key={index} 
                          className="hover:bg-gray-50 cursor-pointer transition-colors dark:hover:bg-[#162032]"
                          onClick={() => {
                            handleRowClick(entry);
                            toggleFullscreen();
                          }}
                        >
                          <td className="px-4 py-3 whitespace-nowrap text-sm font-medium">
                            {entry.name}
                          </td>
                          <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                            {formatTimestamp(entry.first_appearance || entry.first_seen || entry.timestamp)}
                          </td>
                          <td className="px-4 py-3 whitespace-nowrap text-sm">
                            <div className="flex items-center">
                              <div className="w-16 bg-gray-200 dark:bg-gray-700 rounded-full h-2 mr-2">
                                <div
                                  className={`${getConfidenceColor(entry.confidence)} h-2 rounded-full`}
                                  style={{ width: `${entry.confidence * 100}%` }}
                                ></div>
                              </div>
                              <span>{(entry.confidence * 100).toFixed(0)}%</span>
                            </div>
                          </td>
                          {model === "all" && (
                            <td className="px-4 py-3 whitespace-nowrap text-sm">
                              <Badge variant="outline" className="text-xs">
                                {entry.method?.join(", ") || "Unknown"}
                              </Badge>
                            </td>
                          )}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}

          {activeTab === "json" && (
            <div className="animate-fade-in">
              <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:justify-between">
                <div className="relative flex-1 max-w-md">
                  <Search className="h-4 w-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                  <Input 
                    placeholder="Filter JSON properties..." 
                    className="pl-9 text-sm"
                    value={jsonFilter}
                    onChange={(e) => setJsonFilter(e.target.value)}
                  />
                </div>
                <div className="flex gap-2">
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Button
                          variant="outline"
                          size="sm"
                          className="flex items-center gap-2"
                          onClick={() => setShowLineNumbers(!showLineNumbers)}
                        >
                          {showLineNumbers ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                          <span className="hidden sm:inline">{showLineNumbers ? "Hide" : "Show"} Line Numbers</span>
                        </Button>
                      </TooltipTrigger>
                      <TooltipContent>
                        {showLineNumbers ? "Hide" : "Show"} line numbers
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                  
                  <Button
                    variant="outline"
                    size="sm"
                    className="flex items-center gap-2"
                    onClick={handleCopyToClipboard}
                  >
                    <Copy className="h-4 w-4" />
                    <span className="hidden sm:inline">Copy</span>
                  </Button>
                </div>
              </div>

              <div className="bg-gray-50 dark:bg-gray-900 rounded-md overflow-hidden border border-gray-200 dark:border-gray-700">
                <div className="p-2 bg-gray-100 dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 flex justify-between items-center">
                  <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    analysis-results.json
                  </span>
                  <Badge variant="outline" className="text-xs bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300">
                    {prepareJsonData().originalLength} bytes
                  </Badge>
                </div>
                
                <div className="overflow-x-auto relative">
                  <div className="p-4 font-mono text-sm flex">
                    {(() => {
                      const { trimmedLines } = prepareJsonData();
                      
                      return (
                        <>
                          {showLineNumbers && (
                            <div className="select-none text-right mr-4 pr-2 border-r border-gray-200 dark:border-gray-700 text-gray-400 min-w-[2rem]">
                              {trimmedLines.map((_, i) => (
                                <div key={i} className="leading-6 px-1 h-6">{i + 1}</div>
                              ))}
                            </div>
                          )}
                          
                          <div className="flex-1 overflow-x-auto whitespace-pre text-gray-800 dark:text-gray-200">
                            {jsonFilter ? (
                              <>
                                {trimmedLines.map((line, i) => {
                                  if (line.toLowerCase().includes(jsonFilter.toLowerCase())) {
                                    return (
                                      <div key={i} className="leading-6 h-6 bg-yellow-50 dark:bg-yellow-900/20" dangerouslySetInnerHTML={{
                                        __html: line
                                          .replace(/("[^"]*"):/g, '<span class="text-purple-600 dark:text-purple-400">$1</span>:')
                                          .replace(/: (".*?")(,?)/g, ': <span class="text-green-600 dark:text-green-400">$1</span>$2')
                                          .replace(/: (true|false|null)(,?)/g, ': <span class="text-blue-600 dark:text-blue-400">$1</span>$2')
                                          .replace(/: (\d+(\.\d+)?)(,?)/g, ': <span class="text-blue-600 dark:text-blue-400">$1</span>$3')
                                          .replace(/([{}\[\],])/g, '<span class="text-gray-500">$1</span>')
                                      }} />
                                    );
                                  }
                                  return (
                                    <div key={i} className="leading-6 h-6" dangerouslySetInnerHTML={{
                                      __html: line
                                        .replace(/("[^"]*"):/g, '<span class="text-purple-600 dark:text-purple-400">$1</span>:')
                                        .replace(/: (".*?")(,?)/g, ': <span class="text-green-600 dark:text-green-400">$1</span>$2')
                                        .replace(/: (true|false|null)(,?)/g, ': <span class="text-blue-600 dark:text-blue-400">$1</span>$2')
                                        .replace(/: (\d+(\.\d+)?)(,?)/g, ': <span class="text-blue-600 dark:text-blue-400">$1</span>$3')
                                        .replace(/([{}\[\],])/g, '<span class="text-gray-500">$1</span>')
                                    }} />
                                  );
                                })}
                              </>
                            ) : (
                              <>
                                {trimmedLines.map((line, i) => (
                                  <div key={i} className="leading-6 h-6" dangerouslySetInnerHTML={{
                                    __html: line
                                      .replace(/("[^"]*"):/g, '<span class="text-purple-600 dark:text-purple-400">$1</span>:')
                                      .replace(/: (".*?")(,?)/g, ': <span class="text-green-600 dark:text-green-400">$1</span>$2')
                                      .replace(/: (true|false|null)(,?)/g, ': <span class="text-blue-600 dark:text-blue-400">$1</span>$2')
                                      .replace(/: (\d+(\.\d+)?)(,?)/g, ': <span class="text-blue-600 dark:text-blue-400">$1</span>$3')
                                      .replace(/([{}\[\],])/g, '<span class="text-gray-500">$1</span>')
                                  }} />
                                ))}
                              </>
                            )}
                          </div>
                        </>
                      );
                    })()}
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === "metadata" && (
            <div className="animate-fade-in space-y-6">
              <Card>
                <CardHeader className="pb-3">
                  <div className="flex items-center">
                    <Database className="h-5 w-5 mr-2 text-blue-500 dark:text-blue-400" />
                    <CardTitle>Analysis Information</CardTitle>
                  </div>
                  <CardDescription>Details about the processing and model used</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-4">
                      <div>
                        <h4 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-1">Model</h4>
                        <div className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded-lg border border-blue-100 dark:border-blue-900/50">
                          <div className="flex items-center">
                            <div className="font-medium text-blue-800 dark:text-blue-300">{getModelName(results.model)}</div>
                          </div>
                          <div className="text-xs text-blue-700 dark:text-blue-400 mt-1">Model ID: <span className="uppercase">{results.model}</span></div>
                        </div>
                      </div>
                      
                      <div>
                        <h4 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-1">Processing Time</h4>
                        <div className="bg-gray-50 dark:bg-gray-800 p-3 rounded-lg border border-gray-200 dark:border-gray-700">
                          <div className="flex items-center">
                            <Clock className="h-4 w-4 mr-2 text-gray-500 dark:text-gray-400" />
                            <div className="font-medium">{results.processingTime}</div>
                          </div>
                        </div>
                      </div>
                    </div>
                    
                    <div className="space-y-4">
                      <div>
                        <h4 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-1">Detection Results</h4>
                        <div className="bg-green-50 dark:bg-green-900/20 p-3 rounded-lg border border-green-100 dark:border-green-900/50">
                          <div className="flex items-center justify-between">
                            <div>
                              <div className="font-medium text-green-800 dark:text-green-300">
                                {results.names.length} Names Detected
                                {results.uniqueNames !== undefined && (
                                  <span className="text-sm font-normal"> ({results.uniqueNames} unique)</span>
                                )}
                              </div>
                              <div className="text-xs text-green-700 dark:text-green-400 mt-1">
                                {model === "all" ? "Combined results from all models" : "Click on the Names tab to view details"}
                              </div>
                            </div>
                            <Button 
                              variant="outline" 
                              size="sm" 
                              className="h-8 bg-white dark:bg-gray-800 border-green-200 dark:border-green-800 text-green-700 dark:text-green-300 hover:bg-green-50 dark:hover:bg-green-900/30"
                              onClick={() => setActiveTab("names")}
                            >
                              <Tag className="h-3.5 w-3.5 mr-1" />
                              View
                            </Button>
                          </div>
                        </div>
                      </div>
                      
                      <div>
                        <h4 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-1">Analysis Date</h4>
                        <div className="bg-gray-50 dark:bg-gray-800 p-3 rounded-lg border border-gray-200 dark:border-gray-700">
                          <div className="flex items-center">
                            <Calendar className="h-4 w-4 mr-2 text-gray-500 dark:text-gray-400" />
                            <div className="font-medium">{new Date(results.analysisDate).toLocaleString()}</div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader className="pb-3">
                  <div className="flex items-center">
                    <Film className="h-5 w-5 mr-2 text-purple-500 dark:text-purple-400" />
                    <CardTitle>Video File Information</CardTitle>
                  </div>
                  <CardDescription>Metadata about the analysed video file</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
                      <h3 className="font-medium mb-3">{enhancedMetadata?.filename}</h3>
                      
                      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
                        <div className="space-y-1">
                          <div className="text-xs text-gray-500 dark:text-gray-400">File Size</div>
                          <div className="flex items-center">
                            <HardDrive className="h-3.5 w-3.5 mr-1.5 text-gray-500 dark:text-gray-400" />
                            <span className="font-medium">{formatFileSize(enhancedMetadata?.size || "Unknown")}</span>
                          </div>
                        </div>
                        
                        <div className="space-y-1">
                          <div className="text-xs text-gray-500 dark:text-gray-400">Duration</div>
                          <div className="flex items-center">
                            <Clock className="h-3.5 w-3.5 mr-1.5 text-gray-500 dark:text-gray-400" />
                            <span className="font-medium font-mono">{enhancedMetadata?.duration}</span>
                          </div>
                        </div>
                        
                        <div className="space-y-1">
                          <div className="text-xs text-gray-500 dark:text-gray-400">File Type</div>
                          <div className="flex items-center">
                            <FileType className="h-3.5 w-3.5 mr-1.5 text-gray-500 dark:text-gray-400" />
                            <span className="font-medium">{enhancedMetadata?.type}</span>
                          </div>
                        </div>
                        
                        {enhancedMetadata?.fps && (
                          <div className="space-y-1">
                            <div className="text-xs text-gray-500 dark:text-gray-400">Frame Rate</div>
                            <div className="flex items-center">
                              <Film className="h-3.5 w-3.5 mr-1.5 text-gray-500 dark:text-gray-400" />
                              <span className="font-medium">{enhancedMetadata.fps} FPS</span>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          )}
        </div>
      </div>

      <FullscreenModal />
    </div>
  );
};

export default ResultsStep;