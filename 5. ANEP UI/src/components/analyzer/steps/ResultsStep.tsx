
import { useState } from "react";
import { Download, RefreshCw, Copy, Maximize2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";

interface ResultsStepProps {
  results: any;
  onRestart: () => void;
  className?: string;
}

const ResultsStep = ({
  results,
  onRestart,
  className = "",
}: ResultsStepProps) => {
  const { toast } = useToast();
  const [activeTab, setActiveTab] = useState("names");

  const handleDownload = () => {
    // Create a JSON blob from the results
    const jsonBlob = new Blob([JSON.stringify(results, null, 2)], {
      type: "application/json",
    });
    
    // Create a temporary URL for the blob
    const url = URL.createObjectURL(jsonBlob);
    
    // Create a temporary link element to trigger the download
    const link = document.createElement("a");
    link.href = url;
    link.download = `anep-results-${new Date().toISOString().slice(0, 10)}.json`;
    
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
    navigator.clipboard.writeText(JSON.stringify(results, null, 2))
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

  const getModelName = (modelId: string): string => {
    switch (modelId) {
      case "anep":
        return "ANEP";
      case "model1":
        return "MODEL #1";
      case "model2":
        return "MODEL #2";
      case "all":
        return "All Models";
      default:
        return "Unknown Model";
    }
  };

  // Function to format file size
  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return bytes + " bytes";
    else if (bytes < 1048576) return (bytes / 1024).toFixed(2) + " KB";
    else return (bytes / 1048576).toFixed(2) + " MB";
  };

  return (
    <div className={`w-full ${className}`}>
      <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-6">
        <div>
          <h2 className="text-2xl font-bold mb-2">Analysis Results</h2>
          <p className="text-muted-foreground">
            Names detected in {results.videoMetadata.filename}
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
            <button
              className={`px-4 py-3 text-sm font-medium ${
                activeTab === "names"
                  ? "text-primary border-b-2 border-primary"
                  : "text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
              }`}
              onClick={() => setActiveTab("names")}
            >
              Names
            </button>
            <button
              className={`px-4 py-3 text-sm font-medium ${
                activeTab === "json"
                  ? "text-primary border-b-2 border-primary"
                  : "text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
              }`}
              onClick={() => setActiveTab("json")}
            >
              JSON Data
            </button>
            <button
              className={`px-4 py-3 text-sm font-medium ${
                activeTab === "metadata"
                  ? "text-primary border-b-2 border-primary"
                  : "text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
              }`}
              onClick={() => setActiveTab("metadata")}
            >
              Metadata
            </button>
          </div>
        </div>

        <div className="p-4">
          {activeTab === "names" && (
            <div className="animate-fade-in">
              <div className="mb-4 flex items-center justify-between">
                <h3 className="text-lg font-medium">
                  {results.names.length} Names Detected
                </h3>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    // Expand view functionality can be implemented here
                    toast({
                      title: "Fullscreen view",
                      description: "This feature will be implemented in a future update.",
                    });
                  }}
                >
                  <Maximize2 className="h-4 w-4 mr-2" />
                  <span>Expand</span>
                </Button>
              </div>

              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                  <thead>
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
                    {results.names.map((entry: any, index: number) => (
                      <tr key={index}>
                        <td className="px-4 py-3 whitespace-nowrap text-sm font-medium">
                          {entry.name}
                        </td>
                        <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                          {entry.timestamp}
                        </td>
                        <td className="px-4 py-3 whitespace-nowrap text-sm">
                          <div className="flex items-center">
                            <div className="w-16 bg-gray-200 dark:bg-gray-700 rounded-full h-2 mr-2">
                              <div
                                className="bg-blue-500 h-2 rounded-full"
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
            </div>
          )}

          {activeTab === "json" && (
            <div className="animate-fade-in">
              <div className="mb-4 flex justify-end">
                <Button
                  variant="outline"
                  size="sm"
                  className="flex items-center gap-2"
                  onClick={handleCopyToClipboard}
                >
                  <Copy className="h-4 w-4" />
                  <span>Copy</span>
                </Button>
              </div>
              <pre className="bg-gray-50 dark:bg-gray-900 p-4 rounded-md overflow-x-auto text-sm">
                {JSON.stringify(results, null, 2)}
              </pre>
            </div>
          )}

          {activeTab === "metadata" && (
            <div className="animate-fade-in">
              <div className="space-y-6">
                <div>
                  <h3 className="text-lg font-medium mb-3">Analysis Information</h3>
                  <dl className="grid grid-cols-1 md:grid-cols-2 gap-x-4 gap-y-2">
                    <div className="flex">
                      <dt className="text-sm font-medium text-gray-500 dark:text-gray-400 w-32">Model Used:</dt>
                      <dd className="text-sm">{getModelName(results.model)}</dd>
                    </div>
                    <div className="flex">
                      <dt className="text-sm font-medium text-gray-500 dark:text-gray-400 w-32">Processing Time:</dt>
                      <dd className="text-sm">{results.processingTime}</dd>
                    </div>
                    <div className="flex">
                      <dt className="text-sm font-medium text-gray-500 dark:text-gray-400 w-32">Names Found:</dt>
                      <dd className="text-sm">{results.names.length}</dd>
                    </div>
                    <div className="flex">
                      <dt className="text-sm font-medium text-gray-500 dark:text-gray-400 w-32">Analyzed On:</dt>
                      <dd className="text-sm">{new Date().toLocaleString()}</dd>
                    </div>
                  </dl>
                </div>

                <div>
                  <h3 className="text-lg font-medium mb-3">File Information</h3>
                  <dl className="grid grid-cols-1 md:grid-cols-2 gap-x-4 gap-y-2">
                    <div className="flex">
                      <dt className="text-sm font-medium text-gray-500 dark:text-gray-400 w-32">Filename:</dt>
                      <dd className="text-sm truncate max-w-[200px]">{results.videoMetadata.filename}</dd>
                    </div>
                    <div className="flex">
                      <dt className="text-sm font-medium text-gray-500 dark:text-gray-400 w-32">File Size:</dt>
                      <dd className="text-sm">{formatFileSize(results.videoMetadata.size)}</dd>
                    </div>
                    <div className="flex">
                      <dt className="text-sm font-medium text-gray-500 dark:text-gray-400 w-32">File Type:</dt>
                      <dd className="text-sm">{results.videoMetadata.type}</dd>
                    </div>
                  </dl>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ResultsStep;
