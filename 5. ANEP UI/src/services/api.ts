
/**
 * API service for ANEP
 * This is a simulated API service that would typically communicate with a Flask backend.
 * In a production environment, replace these functions with actual API calls.
 */

// Simulate video analysis
export const analyzeVideo = async (
  video: File,
  model: string
): Promise<any> => {
  // Simulate API delay
  await new Promise((resolve) => setTimeout(resolve, 5000));
  
  // Simulate successful response
  return {
    names: [
      { name: "Donald Trump", confidence: 0.95, timestamp: "00:01:23" },
      { name: "Boris Johnson", confidence: 0.87, timestamp: "00:03:45" },
      { name: "Angela Merkel", confidence: 0.92, timestamp: "00:05:12" }
    ],
    model: model,
    processingTime: "00:02:35",
    videoMetadata: {
      filename: video.name,
      size: video.size,
      type: video.type
    }
  };
};

// For simulating errors (uncomment to test error handling)
/*
export const analyzeVideoWithError = async (
  video: File,
  model: string
): Promise<any> => {
  // Simulate API delay
  await new Promise((resolve) => setTimeout(resolve, 3000));
  
  // Simulate error response
  throw new Error("Analysis failed: Server error");
};
*/
