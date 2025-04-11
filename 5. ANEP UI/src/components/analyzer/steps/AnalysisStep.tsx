
import { useState, useEffect } from "react";
import { BarChart3, AlertTriangle } from "lucide-react";

interface AnalysisStepProps {
  videoFile: File;
  selectedModel: string;
  onAnalysisComplete: (results: any) => void;
  className?: string;
}

const AnalysisStep = ({
  videoFile,
  selectedModel,
  onAnalysisComplete,
  className = "",
}: AnalysisStepProps) => {
  const [progress, setProgress] = useState(0);
  const [currentStage, setCurrentStage] = useState("Initializing...");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const analyzeVideo = async () => {
      try {
        // Create form data to send to the backend
        const formData = new FormData();
        formData.append("video", videoFile);
        formData.append("model", selectedModel);

        // Create artificial stages for better UX
        const stages = [
          "Loading video content...", 
          "Preparing for analysis...",
          "Detecting frames with text...",
          "Recognizing name patterns...",
          "Processing extracted names...",
          "Finalizing results...",
        ];

        // Artificial progress simulation (will be replaced by real backend)
        let currentProgress = 0;
        
        for (let i = 0; i < stages.length - 1; i++) {
          setCurrentStage(stages[i]);
          
          // Update progress in smaller increments to look smoother
          const targetProgress = ((i + 1) / stages.length) * 100;
          const progressStep = (targetProgress - currentProgress) / 10;
          
          for (let j = 0; j < 10; j++) {
            await new Promise(resolve => setTimeout(resolve, 500));
            currentProgress += progressStep;
            setProgress(Math.round(currentProgress));
          }
        }
        
        // Set final stage
        setCurrentStage(stages[stages.length - 1]);
        
        // Send actual request to backend (simulated for now)
        // In a real app, you'd make the actual API call here
        // For now, we'll just simulate the response
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        // Simulate a backend response
        const dummyResults = {
          names: [
            { name: "Donald Trump", confidence: 0.95, timestamp: "00:01:23" },
            { name: "Boris Johnson", confidence: 0.87, timestamp: "00:03:45" },
            { name: "Angela Merkel", confidence: 0.92, timestamp: "00:05:12" }
          ],
          model: selectedModel,
          processingTime: "00:02:35",
          videoMetadata: {
            filename: videoFile.name,
            size: videoFile.size,
            type: videoFile.type
          }
        };

        setProgress(100);
        // Wait a bit before completing to show 100% state
        await new Promise(resolve => setTimeout(resolve, 1000));
        onAnalysisComplete(dummyResults);
      } catch (err) {
        console.error("Analysis error:", err);
        setError("An error occurred during analysis. Please try again.");
      }
    };

    analyzeVideo();
  }, [videoFile, selectedModel, onAnalysisComplete]);

  const getModelName = (modelId: string): string => {
    switch (modelId) {
      case "anep":
        return "ANEP";
      case "model1":
        return "Google Vision OCR";
      case "model2":
        return "Llama 3.2 Vision";
      case "all":
        return "All Models";
      default:
        return "Unknown Model";
    }
  };
  

  return (
    <div className={`w-full ${className}`}>
      <h2 className="text-2xl font-bold mb-4">Analysing Video</h2>
      {!error ? (
        <div className="animate-fade-in">
          <p className="text-muted-foreground mb-6">
            Extracting names using {getModelName(selectedModel)} model
          </p>

          <div className="bg-secondary p-8 rounded-lg text-center">
            <div className="flex flex-col items-center">
              {progress < 100 ? (
                <div className="w-24 h-24 rounded-full border-4 border-t-primary border-r-primary border-b-gray-200 border-l-gray-200 animate-spin mb-6" />
              ) : (
                <div className="w-24 h-24 rounded-full bg-green-100 flex items-center justify-center mb-6">
                  <BarChart3 className="h-10 w-10 text-green-600" />
                </div>
              )}
              
              <h3 className="text-xl font-medium mb-2">
                {progress < 100 ? "Processing" : "Analysis Complete"}
              </h3>
              
              <p className="text-gray-500 dark:text-gray-400 mb-6">
                {currentStage}
              </p>

              <div className="w-full max-w-md mb-2">
                <div className="progress-wrapper">
                  <div 
                    className="progress-inner" 
                    style={{ width: `${progress}%` }}
                  />
                </div>
              </div>
              
              <p className="text-sm text-gray-500">{progress}% complete</p>
            </div>
          </div>

          <div className="mt-6">
            <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg flex">
              <div className="mr-4 text-primary">ℹ️</div>
              <div>
                <p className="text-sm">
                  Analysis may take several minutes depending on the video length 
                  and selected model. You can continue working in other tabs 
                  while we process your video.
                </p>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="bg-red-50 dark:bg-red-900/20 p-6 rounded-lg flex items-start">
          <AlertTriangle className="h-6 w-6 text-red-500 mr-4 mt-0.5" />
          <div>
            <h3 className="text-lg font-medium text-red-800 dark:text-red-300 mb-2">
              Analysis Failed
            </h3>
            <p className="text-red-700 dark:text-red-400">
              {error}
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default AnalysisStep;
