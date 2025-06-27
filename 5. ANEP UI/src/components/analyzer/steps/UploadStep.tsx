import { useState, useCallback, useRef, useEffect } from "react";
import {
  Upload,
  X,
  FileVideo,
  Clock,
  HardDrive,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";

interface UploadStepProps {
  onVideoUploaded: (file: File | null) => void;
  className?: string;
  initialFile?: File | null;
  initialURL?: string | null;
  initialMetadata?: { duration: number; type: string } | null;
}

const UploadStep = ({
  onVideoUploaded,
  className = "",
  initialFile = null,
  initialURL = null,
  initialMetadata = null,
}: UploadStepProps) => {
  const { toast, dismiss } = useToast();
  const [isDragging, setIsDragging] = useState(false);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoURL, setVideoURL] = useState<string | null>(null);
  const [videoMetadata, setVideoMetadata] = useState<{ duration: number; type: string } | null>(null);
  const [isVideoLoading, setIsVideoLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const toastIdRef = useRef<string | undefined>(undefined);

  // Clear vid_dur from localStorage on component mount
  useEffect(() => {
    localStorage.removeItem("CurrentVideoName");
    localStorage.removeItem("vid_dur");
    localStorage.removeItem("vid_siz");
  }, []);

  useEffect(() => {
    if (initialFile && initialURL && initialMetadata) {
      setVideoFile(initialFile);
      setVideoURL(initialURL);
      setVideoMetadata(initialMetadata);
      // Save initial duration to localStorage if available
      if (initialMetadata.duration) {
        localStorage.setItem("vid_dur", initialMetadata.duration.toString());
      }
    }
  }, [initialFile, initialURL, initialMetadata]);

  const extractVideoMetadata = (file: File) => {
    return new Promise((resolve) => {
      const video = document.createElement("video");
      video.preload = "auto";
      video.muted = true; // Required for autoplay in many browsers
      video.playsInline = true; // Required for mobile Safari
      
      const cleanup = () => {
        if (video.src && video.src.startsWith('blob:')) {
          URL.revokeObjectURL(video.src);
        }
      };

      const handleMetadata = () => {
        const metadata = {
          duration: video.duration || 0,
          type: file.type,
        };
        setVideoMetadata(metadata);
        
        // Save duration to localStorage
        localStorage.setItem("vid_dur", metadata.duration.toString());
        
        cleanup();
        resolve(metadata);
      };

      const handleError = (e) => {
        console.error("Error loading video metadata:", e);
        console.error("File type:", file.type);
        console.error("File size:", file.size);
        cleanup();
        
        // Still set basic metadata even if video fails to load
        const basicMetadata = {
          duration: 0,
          type: file.type,
        };
        setVideoMetadata(basicMetadata);
        resolve(basicMetadata);
      };

      video.addEventListener('loadedmetadata', handleMetadata);
      video.addEventListener('error', handleError);
      video.addEventListener('abort', handleError);
      
      // Create object URL and set as source
      const objectURL = URL.createObjectURL(file);
      video.src = objectURL;
      
      // Fallback timeout
      setTimeout(() => {
        if (video.readyState === 0) {
          console.warn('Video metadata loading timeout');
          handleError(new Error('Timeout'));
        }
      }, 5000);
    });
  };

  const uploadToServer = async (file: File) => {
    const formData = new FormData();
    formData.append("video", file);
  
    try {
      const response = await fetch("http://localhost:5050/api/upload", {
        method: "POST",
        body: formData,
      });
  
      if (!response.ok) throw new Error("Upload failed");
  
      console.log("✅ Server upload complete");
  
      const res = await fetch("http://localhost:5050/api/latest-upload");
      const data = await res.json();
  
      if (data.latest) {
        console.log("💾 Latest uploaded video:", data.latest);
        localStorage.setItem("CurrentVideoName", data.latest);
      } else {
        console.log("⚠️ No uploads yet");
      }
  
    } catch (err) {
      console.error("❌ Server upload error or fetch failed:", err);
      toast({
        title: "Upload failed",
        description: "There was an issue uploading your video to the server.",
        variant: "destructive",
      });
    }
  };
  
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const processVideoFile = async (file: File) => {
    setIsVideoLoading(true);
    const url = URL.createObjectURL(file);
    setVideoFile(file);
    setVideoURL(url);
    
    // Extract metadata
    await extractVideoMetadata(file);
    
    onVideoUploaded(file);
    uploadToServer(file);
    
    localStorage.setItem("vid_siz", (file.size / (1024 * 1024)).toFixed(2)); // MB

    const toastData = toast({
      title: "Video uploaded 🎉",
      description: `${file.name} has successfully uploaded!`,
    });

    toastIdRef.current = toastData?.id as string;
    setIsVideoLoading(false);
  };

  const handleDrop = useCallback(
    async (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);

      const files = Array.from(e.dataTransfer.files);
      const file = files.find(f => f.type.startsWith("video/"));

      if (file) {
        await processVideoFile(file);
      } else {
        toast({
          title: "Invalid file",
          description: "Please upload a video file.",
          variant: "destructive",
        });
      }
    },
    [onVideoUploaded, toast]
  );

  const handleFileInputChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];

      if (file.type.startsWith("video/")) {
        await processVideoFile(file);
      } else {
        toast({
          title: "Invalid file",
          description: "Please upload a video file.",
          variant: "destructive",
        });
      }
    }
  };

  const clearSelectedFile = () => {
    if (videoURL) URL.revokeObjectURL(videoURL);
    setVideoFile(null);
    setVideoURL(null);
    setVideoMetadata(null);
    setIsVideoLoading(false);
    onVideoUploaded(null);

    // Clear duration from localStorage when file is removed
    localStorage.removeItem("vid_dur");
    localStorage.removeItem("vid_siz"); 
    localStorage.removeItem("CurrentVideoName");

    if (toastIdRef.current) {
      dismiss(toastIdRef.current);
      toastIdRef.current = undefined;
    }
  };

  const formatDuration = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    return `${hours.toString().padStart(2, "0")}:${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  };

  return (
    <div className={`w-full ${className}`}>
      <h2 className="text-2xl font-bold mb-4">Upload Video</h2>
      <p className="text-muted-foreground mb-6">
        Upload a news video to extract names from on-screen graphics
      </p>

      {!videoFile ? (
        <div
          className={`
            file-drop-area ${isDragging ? "dragging" : ""} cursor-pointer
            border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg
            transition-all duration-300
            hover:border-[#2463EB] hover:bg-[#f5faff] hover:shadow-md
            dark:hover:border-[#2463EB] dark:hover:bg-[#1a2d4a]
          `}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <div className="flex flex-col items-center justify-center text-center p-6">
            <div className="w-14 h-14 bg-blue-50 dark:bg-blue-900/20 rounded-full flex items-center justify-center mb-4">
              <Upload className="h-7 w-7 text-primary" />
            </div>
            <p className="mb-2 text-lg font-medium">Drag and drop your video file</p>
            <p className="mb-4 text-sm text-gray-500 dark:text-gray-400">
              Or click anywhere to browse from your device
            </p>
            <input
              type="file"
              ref={fileInputRef}
              className="hidden"
              accept="video/*"
              onChange={handleFileInputChange}
            />
          </div>
        </div>
      ) : (
        <div className="bg-secondary rounded-lg p-6 animate-fade-in">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 bg-blue-100 dark:bg-blue-900/30 rounded flex items-center justify-center">
                <FileVideo className="h-6 w-6 text-primary" />
              </div>
              <div>
                <p className="font-medium truncate max-w-[200px] sm:max-w-[300px] md:max-w-md">
                  {videoFile.name}
                </p>
                <div className="text-sm text-muted-foreground flex flex-col sm:flex-row sm:items-center gap-1 sm:gap-3 mt-1">
                  <span className="flex items-center gap-1">
                    <HardDrive className="w-4 h-4" />
                    {(videoFile.size / (1024 * 1024)).toFixed(2)} MB
                  </span>
                  {videoMetadata && (
                    <>
                      <span className="flex items-center gap-1">
                        <Clock className="w-4 h-4" />
                        {formatDuration(videoMetadata.duration)}
                      </span>
                      <span className="flex items-center gap-1">
                        <FileVideo className="w-4 h-4" />
                        {videoMetadata.type}
                      </span>
                    </>
                  )}
                </div>
              </div>
            </div>
            <Button
              variant="ghost"
              size="icon"
              className="text-red-500 hover:text-red-600"
              onClick={clearSelectedFile}
              aria-label="Remove file"
            >
              <X className="h-5 w-5" />
            </Button>
          </div>

          {isVideoLoading ? (
            <div className="flex items-center justify-center h-[280px] bg-gray-100 dark:bg-gray-800 rounded-md">
              <div className="text-center">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-2"></div>
                <p className="text-sm text-muted-foreground">Loading video preview...</p>
              </div>
            </div>
          ) : videoURL ? (
            <div className="rounded-md overflow-hidden border border-gray-300 dark:border-gray-700 bg-black">
              <video
                key={videoFile?.name || 'video'} // Force re-render on file change
                controls
                preload="auto"
                muted
                playsInline
                className="w-full h-[280px] object-contain bg-black"
                onError={(e) => {
                  console.error('Video error:', e);
                  console.error('Video error details:', e.currentTarget.error);
                }}
                onEmptied={() => {
                  console.log('Video emptied');
                }}
                onStalled={() => {
                  console.log('Video stalled');
                }}
              >
                <source src={videoURL} type={videoFile?.type || 'video/mp4'} />
                <p className="text-white p-4 text-center">
                  Your browser doesn't support this video format.<br/>
                </p>
              </video>
            </div>
          ) : (
            // Show sample video if no file uploaded
            <div className="rounded-md overflow-hidden border border-gray-300 dark:border-gray-700 bg-black">
              <div className="flex items-center justify-center h-[280px] bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-800 dark:to-gray-900">
                <div className="text-center">
                  <FileVideo className="h-16 w-16 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-600 dark:text-gray-400 font-medium">Video preview will appear here</p>
                  <p className="text-sm text-gray-500 dark:text-gray-500 mt-2">Upload a video to see the preview</p>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      <div className="mt-8">
        <h3 className="text-lg font-medium mb-3">Supported file formats:</h3>
        <div className="flex flex-wrap gap-2">
          {[".mp4", ".avi", ".mov", ".mkv"].map((format) => (
            <div
              key={format}
              className="flex items-center gap-2 bg-[#e9f1ff] text-[#2463eb] dark:bg-[#1a2d4a] dark:text-[#93b7ff] px-3 py-1 rounded text-sm font-medium"
            >
              <FileVideo className="w-4 h-4" />
              {format}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default UploadStep;