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
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const toastIdRef = useRef<string | undefined>(undefined); // ✅ TS-safe

  // ✅ Restore state from parent when coming back
  useEffect(() => {
    if (initialFile && initialURL && initialMetadata) {
      setVideoFile(initialFile);
      setVideoURL(initialURL);
      setVideoMetadata(initialMetadata);
    }
  }, [initialFile, initialURL, initialMetadata]);

  const extractVideoMetadata = (file: File) => {
    const video = document.createElement("video");
    video.preload = "metadata";
    video.src = URL.createObjectURL(file);
    video.onloadedmetadata = () => {
      setVideoMetadata({
        duration: video.duration,
        type: file.type,
      });
      URL.revokeObjectURL(video.src);
    };
  };

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);

      const files = Array.from(e.dataTransfer.files);
      const file = files.find(f => f.type.startsWith("video/"));

      if (file) {
        const url = URL.createObjectURL(file);
        setVideoFile(file);
        setVideoURL(url);
        extractVideoMetadata(file);
        onVideoUploaded(file);

        const toastData = toast({
          title: "Video uploaded",
          description: `${file.name} has been successfully uploaded.`,
        });

        toastIdRef.current = toastData?.id as string;
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

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];

      if (file.type.startsWith("video/")) {
        const url = URL.createObjectURL(file);
        setVideoFile(file);
        setVideoURL(url);
        extractVideoMetadata(file);
        onVideoUploaded(file);

        const toastData = toast({
          title: "Video uploaded",
          description: `${file.name} has been successfully uploaded.`,
        });

        toastIdRef.current = toastData?.id as string;
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
    onVideoUploaded(null);

    if (toastIdRef.current) {
      dismiss(toastIdRef.current);
      toastIdRef.current = undefined;
    }
  };

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
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

          {videoURL && (
            <div className="rounded overflow-hidden border border-gray-300 dark:border-gray-700">
              <video
                src={videoURL}
                controls
                className="w-full h-[280px] object-contain rounded-md"
              />
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
