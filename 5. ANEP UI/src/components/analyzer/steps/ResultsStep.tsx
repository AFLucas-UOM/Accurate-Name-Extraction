import { useState, useEffect } from "react";
import { Download, RefreshCw, Copy, Maximize2, X, Calendar, Clock, FileType, HardDrive, ChevronRight, CheckCircle, ExternalLink, Tag, Search, Filter, Info, Code, Eye, EyeOff, Film, PlusCircle, Database, Trash2, Share2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Input } from "@/components/ui/input";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";

interface NameDetection {
  name: string;
  confidence: number;
  timestamp: string;
  frames?: number[];
  duration?: string;
}

interface VideoMetadata {
  filename: string;
  size: string;
  duration: string;
  resolution: string;
  type: string;
  frameRate?: string;
  url?: string; // Added URL field for video source
}

interface AnalysisResults {
  names: NameDetection[];
  model: string;
  modelName: string;
  processingTime: string;
  videoMetadata: VideoMetadata;
  analysisDate: string;
  videoFile?: File; // Added optional videoFile for direct access
}

interface ResultsStepProps {
  results: AnalysisResults;
  onRestart: () => void;
  className?: string;
}

  // Function to extract metadata from a video file or URL
  const extractVideoMetadata = (source: File | string | null): Promise<Partial<VideoMetadata>> => {
    return new Promise((resolve) => {
      if (!source) {
        resolve({
          duration: "Unknown",
          resolution: "Unknown"
        });
        return;
      }

      const video = document.createElement("video");
      video.preload = "metadata";
      
      // Handle both File objects and URLs
      if (source instanceof File) {
        video.src = URL.createObjectURL(source);
      } else if (typeof source === 'string') {
        video.src = source;
      }
      
      video.onloadedmetadata = () => {
        // Format duration as HH:MM:SS
        const formatDuration = (seconds: number): string => {
          const hrs = Math.floor(seconds / 3600);
          const mins = Math.floor((seconds % 3600) / 60);
          const secs = Math.floor(seconds % 60);
          return `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        };
        
        // Extract metadata
        const metadata: Partial<VideoMetadata> = {
          duration: formatDuration(video.duration),
          resolution: `${video.videoWidth}x${video.videoHeight}`,
        };
        
        // Cleanup
        if (source instanceof File) {
          URL.revokeObjectURL(video.src);
        }
        
        resolve(metadata);
      };
      
      // Handle errors
      video.onerror = () => {
        if (source instanceof File) {
          URL.revokeObjectURL(video.src);
        }
        resolve({
          duration: "Unknown",
          resolution: "Unknown"
        });
      };
      
      // Set a timeout in case metadata loading takes too long
      setTimeout(() => {
        if (source instanceof File) {
          URL.revokeObjectURL(video.src);
        }
        resolve({
          duration: "Unknown",
          resolution: "Unknown"
        });
      }, 5000); // 5 second timeout
    });
  };

const ResultsStep = ({
  results,
  onRestart,
  className = "",
}: ResultsStepProps) => {
  const { toast } = useToast();
  const [activeTab, setActiveTab] = useState("names");
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [selectedName, setSelectedName] = useState<NameDetection | null>(null);
  const [enhancedMetadata, setEnhancedMetadata] = useState<VideoMetadata>(results.videoMetadata);
  const [jsonFilter, setJsonFilter] = useState("");
  const [expandedSections, setExpandedSections] = useState<string[]>(["names", "videoMetadata", "model"]);
  const [showLineNumbers, setShowLineNumbers] = useState(true);
  
  // Extract video metadata on component mount - but NEVER modify the incoming values
  useEffect(() => {
    // Simply use the exact metadata from the backend without any modifications
    setEnhancedMetadata(results.videoMetadata);
  }, [results.videoMetadata]);

  const handleDownload = () => {
    // Create a JSON blob from the results with enhanced metadata
    const resultsToDownload = {
      ...results,
      videoMetadata: enhancedMetadata
    };
    
    const jsonBlob = new Blob([JSON.stringify(resultsToDownload, null, 2)], {
      type: "application/json",
    });
    
    // Create a temporary URL for the blob
    const url = URL.createObjectURL(jsonBlob);
    
    // Create a temporary link element to trigger the download
    const link = document.createElement("a");
    link.href = url;
    link.download = `analysis-results-${new Date().toISOString().slice(0, 10)}.json`;
    
    // Trigger the download
    document.body.appendChild(link);
    link.click();
    
    // Clean up
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    
    toast({
      title: "Results downloaded",
      description: "The results have been saved as a JSON file.",
    });
  };

  const handleCopyToClipboard = () => {
    // Include enhanced metadata in the copied JSON
    const resultsToCopy = {
      ...results,
      videoMetadata: enhancedMetadata
    };
    
    // Format the JSON properly for clipboard
    const formattedJson = JSON.stringify(resultsToCopy, null, 2);
    
    navigator.clipboard.writeText(formattedJson)
      .then(() => {
        toast({
          title: "Copied to clipboard",
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

  const getModelName = (modelId: string): string => {
    const modelMap: Record<string, string> = {
      "anep": "ANEP (Accurate Name Extraction Pipeline)",
      "model1": "Google Cloud Vision & Gemini 1.5 Pro",
      "model2": "Llama 4 Maverick",
      "all": "All Models (Ensemble)",
    };
    
    return modelMap[modelId] || "Unknown Model";
  };

  // Function to format file size
  const formatFileSize = (sizeString: string): string => {
    // If it's already formatted, return as is
    if (typeof sizeString === 'string' && sizeString.includes('MB')) {
      return sizeString;
    }
    
    // Try to parse as number
    const bytes = parseFloat(sizeString);
    if (isNaN(bytes)) return sizeString;
    
    if (bytes < 1024) return bytes + " bytes";
    else if (bytes < 1048576) return (bytes / 1024).toFixed(2) + " KB";
    else return (bytes / 1048576).toFixed(2) + " MB";
  };
  
  // Process video metadata to properly calculate values when needed
  const processVideoMetadata = (metadata: VideoMetadata): VideoMetadata => {
    // Create a new object to avoid modifying the original
    const processed = { ...metadata };
    
    // Calculate duration if needed
    if (processed.duration === "Processed") {
      // Try to get it from the video element if available
      if (results.videoFile) {
        try {
          const video = document.createElement('video');
          video.src = URL.createObjectURL(results.videoFile);
          video.onloadedmetadata = () => {
            const hrs = Math.floor(video.duration / 3600);
            const mins = Math.floor((video.duration % 3600) / 60);
            const secs = Math.floor(video.duration % 60);
            processed.duration = `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
            URL.revokeObjectURL(video.src);
          };
        } catch (e) {
          console.error("Failed to calculate video duration:", e);
        }
      }
    }
    
    // Calculate resolution if needed
    if (processed.resolution === "Extracted") {
      // Try to get it from the video element if available
      if (results.videoFile) {
        try {
          const video = document.createElement('video');
          video.src = URL.createObjectURL(results.videoFile);
          video.onloadedmetadata = () => {
            processed.resolution = `${video.videoWidth}x${video.videoHeight}`;
            URL.revokeObjectURL(video.src);
          };
        } catch (e) {
          console.error("Failed to calculate video resolution:", e);
        }
      }
    }
    
    return processed;
  };

  // Get confidence color based on value
  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.9) return "bg-green-500";
    if (confidence >= 0.7) return "bg-blue-500";
    if (confidence >= 0.5) return "bg-yellow-500";
    return "bg-red-500";
  };

  // Format timestamp for better display
  const formatTimestamp = (timestamp: string) => {
    // If timestamp is in format like "00:01:23"
    if (/^\d{2}:\d{2}:\d{2}$/.test(timestamp)) {
      return timestamp;
    }
    
    // Try to convert from seconds to HH:MM:SS
    const seconds = parseFloat(timestamp);
    if (!isNaN(seconds)) {
      const hrs = Math.floor(seconds / 3600);
      const mins = Math.floor((seconds % 3600) / 60);
      const secs = Math.floor(seconds % 60);
      
      return `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    
    return timestamp;
  };

  // Toggle section expansion in JSON view
  const toggleSection = (section: string) => {
    setExpandedSections((prev) => 
      prev.includes(section) 
        ? prev.filter(s => s !== section) 
        : [...prev, section]
    );
  };

  // Get severity color
  const getSeverityColor = (value: number, type: "bg" | "text" | "border" = "bg") => {
    if (type === "bg") {
      if (value >= 0.9) return "bg-green-500";
      if (value >= 0.7) return "bg-blue-500";
      if (value >= 0.5) return "bg-yellow-500";
      return "bg-red-500";
    } else if (type === "text") {
      if (value >= 0.9) return "text-green-500 dark:text-green-400";
      if (value >= 0.7) return "text-blue-500 dark:text-blue-400";
      if (value >= 0.5) return "text-yellow-500 dark:text-yellow-400";
      return "text-red-500 dark:text-red-400";
    } else {
      if (value >= 0.9) return "border-green-500";
      if (value >= 0.7) return "border-blue-500";
      if (value >= 0.5) return "border-yellow-500";
      return "border-red-500";
    }
  };

  // Create Tab components to make the code more readable
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
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 backdrop-blur-sm overflow-auto p-4">
        <div className="bg-white dark:bg-[#1e293b] rounded-xl shadow-xl w-full max-w-6xl max-h-[90vh] flex flex-col overflow-hidden">
          {/* Modal header */}
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
          
          {/* Modal content */}
          <div className="overflow-auto p-6 flex-1 bg-gray-50 dark:bg-[#111827]">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Names table - takes 2/3 of space */}
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
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
              
              {/* Detail panel - takes 1/3 of space */}
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
                        <span className="text-sm text-gray-700 dark:text-gray-300">Appeared at: <span className="font-mono">{formatTimestamp(selectedName.timestamp)}</span></span>
                      </div>
                      
                      {selectedName.duration && (
                        <div className="flex items-center gap-2">
                          <Calendar className="h-4 w-4 text-gray-500 dark:text-gray-400" />
                          <span className="text-sm text-gray-700 dark:text-gray-300">Duration: {selectedName.duration}</span>
                        </div>
                      )}
                      
                      {selectedName.frames && selectedName.frames.length > 0 && (
                        <div className="pt-1">
                          <div className="flex items-center gap-2 mb-2">
                            <FileType className="h-4 w-4 text-gray-500 dark:text-gray-400" />
                            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Detected in {selectedName.frames.length} frames</span>
                          </div>
                          <div className="grid grid-cols-5 gap-1 max-h-32 overflow-y-auto pl-6">
                            {selectedName.frames.slice(0, 10).map((frame, idx) => (
                              <Badge key={idx} variant="outline" className="justify-center dark:border-gray-600 dark:text-gray-300">
                                {frame}
                              </Badge>
                            ))}
                            {selectedName.frames.length > 10 && (
                              <Badge variant="outline" className="justify-center dark:border-gray-600 dark:text-gray-300">
                                +{selectedName.frames.length - 10} more
                              </Badge>
                            )}
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

            {/* File information */}
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
                    <dd className="text-sm text-gray-700 dark:text-gray-300 truncate">{enhancedMetadata.filename}</dd>
                    
                    <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">File Size:</dt>
                    <dd className="text-sm text-gray-700 dark:text-gray-300">{formatFileSize(enhancedMetadata.size)}</dd>
                    
                    <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">Duration:</dt>
                    <dd className="text-sm text-gray-700 dark:text-gray-300">
                      {processVideoMetadata(enhancedMetadata).duration}
                    </dd>
                    
                    <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">Resolution:</dt>
                    <dd className="text-sm text-gray-700 dark:text-gray-300">
                      {processVideoMetadata(enhancedMetadata).resolution}
                    </dd>
                    
                    <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">File Type:</dt>
                    <dd className="text-sm text-gray-700 dark:text-gray-300">{enhancedMetadata.type}</dd>
                    
                    {enhancedMetadata.frameRate && (
                      <>
                        <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">Frame Rate:</dt>
                        <dd className="text-sm text-gray-700 dark:text-gray-300">{enhancedMetadata.frameRate}</dd>
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
                    
                    <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">Analysed On:</dt>
                    <dd className="text-sm text-gray-700 dark:text-gray-300">{new Date(results.analysisDate).toLocaleString()}</dd>
                  </dl>
                </div>
              </div>
            </div>
          </div>
          
          {/* Modal footer */}
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

  // Main component render
  return (
    <div className={`w-full ${className}`}>
      <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-6">
        <div>
          <h2 className="text-2xl font-bold mb-2">Analysis Results</h2>
          <p className="text-muted-foreground">
            {results.names.length} names detected in {enhancedMetadata.filename}
          </p>
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
                  <span>{results.names.length} Names Detected</span>
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

              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                  <thead className="bg-gray-50 dark:bg-gray-800">
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
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                    {results.names.map((entry, index) => (
                      <tr 
                        key={index} 
                        className="hover:bg-gray-50 dark:hover:bg-gray-750 cursor-pointer transition-colors"
                        onClick={() => {
                          handleRowClick(entry);
                          toggleFullscreen();
                        }}
                      >
                        <td className="px-4 py-3 whitespace-nowrap text-sm font-medium">
                          {entry.name}
                        </td>
                        <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                          {formatTimestamp(entry.timestamp)}
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
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              
              <div className="mt-4 text-center">
                <Button 
                  variant="ghost" 
                  size="sm"
                  onClick={toggleFullscreen}
                  className="text-gray-500 text-sm flex items-center gap-1"
                >
                  <ExternalLink className="h-3 w-3" />
                  <span>Open detailed view for more information</span>
                </Button>
              </div>
            </div>
          )}

          {/* IMPROVED JSON TAB */}
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
                    {JSON.stringify({...results, videoMetadata: enhancedMetadata}).length} bytes
                  </Badge>
                </div>
                
                <div className="overflow-x-auto relative">
                  <div className="p-4 font-mono text-sm flex">
                    {/* Line numbers */}
                    {showLineNumbers && (
                      <div className="select-none text-right mr-4 pr-2 border-r border-gray-200 dark:border-gray-700 text-gray-400 min-w-[2rem]">
                        {JSON.stringify({...results, videoMetadata: enhancedMetadata}, null, 2)
                          .split('\n')
                          .map((_, i) => (
                            <div key={i} className="leading-6 px-1">{i + 1}</div>
                          ))}
                      </div>
                    )}
                    
                    {/* Direct JSON Content Display */}
                    <pre className="flex-1 overflow-x-auto whitespace-pre text-gray-800 dark:text-gray-200">
                      {jsonFilter ? (
                        // If there's a filter, process the JSON and apply highlighting
                        <div dangerouslySetInnerHTML={{
                          __html: JSON.stringify({...results, videoMetadata: enhancedMetadata}, null, 2)
                            .split('\n')
                            .map(line => {
                              if (line.toLowerCase().includes(jsonFilter.toLowerCase())) {
                                // Apply the same syntax highlighting to filtered results
                                return `<div class="bg-yellow-50 dark:bg-yellow-900/20">${
                                  line
                                    .replace(/("[^"]*"):/g, '<span class="text-purple-600 dark:text-purple-400">$1</span>:')
                                    .replace(/: (".*?")(,?)/g, ': <span class="text-green-600 dark:text-green-400">$1</span>$2')
                                    .replace(/: (true|false|null)(,?)/g, ': <span class="text-blue-600 dark:text-blue-400">$1</span>$2')
                                    .replace(/: (\d+(\.\d+)?)(,?)/g, ': <span class="text-blue-600 dark:text-blue-400">$1</span>$3')
                                    .replace(/([{}\[\],])/g, '<span class="text-gray-500">$1</span>')
                                }</div>`;
                              }
                              // Normal syntax highlighting for non-matching lines
                              return `<div>${
                                line
                                  .replace(/("[^"]*"):/g, '<span class="text-purple-600 dark:text-purple-400">$1</span>:')
                                  .replace(/: (".*?")(,?)/g, ': <span class="text-green-600 dark:text-green-400">$1</span>$2')
                                  .replace(/: (true|false|null)(,?)/g, ': <span class="text-blue-600 dark:text-blue-400">$1</span>$2')
                                  .replace(/: (\d+(\.\d+)?)(,?)/g, ': <span class="text-blue-600 dark:text-blue-400">$1</span>$3')
                                  .replace(/([{}\[\],])/g, '<span class="text-gray-500">$1</span>')
                              }</div>`;
                            })
                            .join('')
                        }} />
                      ) : (
                        // Standard syntax highlighting for JSON
                        <div dangerouslySetInnerHTML={{
                          __html: JSON.stringify({...results, videoMetadata: enhancedMetadata}, null, 2)
                            .replace(/("[^"]*"):/g, '<span class="text-purple-600 dark:text-purple-400">$1</span>:')
                            .replace(/: (".*?")(,?)/g, ': <span class="text-green-600 dark:text-green-400">$1</span>$2')
                            .replace(/: (true|false|null)(,?)/g, ': <span class="text-blue-600 dark:text-blue-400">$1</span>$2')
                            .replace(/: (\d+(\.\d+)?)(,?)/g, ': <span class="text-blue-600 dark:text-blue-400">$1</span>$3')
                            .replace(/([{}\[\],])/g, '<span class="text-gray-500">$1</span>')
                            .split('\n')
                            .map(line => `<div>${line}</div>`)
                            .join('')
                        }} />
                      )}
                    </pre>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* IMPROVED METADATA TAB */}
          {activeTab === "metadata" && (
            <div className="animate-fade-in space-y-6">
              {/* Analysis Information */}
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
                          <div className="text-xs text-blue-700 dark:text-blue-400 mt-1">Model ID: {results.model}</div>
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
                              <div className="font-medium text-green-800 dark:text-green-300">{results.names.length} Names Detected</div>
                              <div className="text-xs text-green-700 dark:text-green-400 mt-1">
                                Click on the "Names" tab to view details
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
              
              {/* File Information */}
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
                      <h3 className="font-medium mb-3">{enhancedMetadata.filename}</h3>
                      
                      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
                        <div className="space-y-1">
                          <div className="text-xs text-gray-500 dark:text-gray-400">File Size</div>
                          <div className="flex items-center">
                            <HardDrive className="h-3.5 w-3.5 mr-1.5 text-gray-500 dark:text-gray-400" />
                            <span className="font-medium">{formatFileSize(enhancedMetadata.size)}</span>
                          </div>
                        </div>
                        
                        <div className="space-y-1">
                          <div className="text-xs text-gray-500 dark:text-gray-400">Duration</div>
                          <div className="flex items-center">
                            <Clock className="h-3.5 w-3.5 mr-1.5 text-gray-500 dark:text-gray-400" />
                            <span className="font-medium font-mono">{processVideoMetadata(enhancedMetadata).duration}</span>
                          </div>
                        </div>
                        
                        <div className="space-y-1">
                          <div className="text-xs text-gray-500 dark:text-gray-400">File Type</div>
                          <div className="flex items-center">
                            <FileType className="h-3.5 w-3.5 mr-1.5 text-gray-500 dark:text-gray-400" />
                            <span className="font-medium">{enhancedMetadata.type}</span>
                          </div>
                        </div>
                        
                        <div className="space-y-1">
                          <div className="text-xs text-gray-500 dark:text-gray-400">Resolution</div>
                          <div className="flex items-center">
                            <Maximize2 className="h-3.5 w-3.5 mr-1.5 text-gray-500 dark:text-gray-400" />
                            <span className="font-medium">{processVideoMetadata(enhancedMetadata).resolution}</span>
                          </div>
                        </div>
                        
                        {enhancedMetadata.frameRate && (
                          <div className="space-y-1">
                            <div className="text-xs text-gray-500 dark:text-gray-400">Frame Rate</div>
                            <div className="flex items-center">
                              <Film className="h-3.5 w-3.5 mr-1.5 text-gray-500 dark:text-gray-400" />
                              <span className="font-medium">{enhancedMetadata.frameRate}</span>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </CardContent>
                <CardFooter className="border-t border-gray-200 dark:border-gray-700 pt-4 flex justify-between">
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    Analysis completed on {new Date(results.analysisDate).toLocaleString()}
                  </div>
                  <div className="flex gap-2">
                    {/* Removed buttons as requested */}
                  </div>
                </CardFooter>
              </Card>
            </div>
          )}
        </div>
      </div>

      {/* Render the fullscreen modal */}
      <FullscreenModal />
    </div>
  );
};

export default ResultsStep;