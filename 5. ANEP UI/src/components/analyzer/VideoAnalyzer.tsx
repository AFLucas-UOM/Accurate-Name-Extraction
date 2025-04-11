import React, { useState } from "react";

// Import step components
import UploadStep from "./steps/UploadStep";
import ModelSelectionStep from "./steps/ModelSelectionStep";
import ConfirmationStep from "./steps/ConfirmationStep";
import AnalysisStep from "./steps/AnalysisStep";
import ResultsStep from "./steps/ResultsStep";

// Import refactored components
import StepIndicator from "./step-indicator/StepIndicator";
import MobileStepIndicator from "./step-indicator/MobileStepIndicator";
import NavigationButtons from "./navigation/NavigationButtons";
import { defineSteps } from "./config/stepsConfig";

const VideoAnalyzer = () => {
  const [currentStep, setCurrentStep] = useState(1);

  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoURL, setVideoURL] = useState<string | null>(null);
  const [videoMetadata, setVideoMetadata] = useState<{ duration: number; type: string } | null>(null);

  const [selectedModel, setSelectedModel] = useState("anep");
  const [results, setResults] = useState<any>(null);

  const steps = defineSteps();

  const goToNextStep = () => {
    setCurrentStep((prevStep) => Math.min(prevStep + 1, 5));
  };

  const goToPreviousStep = () => {
    setCurrentStep((prevStep) => Math.max(prevStep - 1, 1));
  };

  const handleRestart = () => {
    setVideoFile(null);
    setVideoURL(null);
    setVideoMetadata(null);
    setSelectedModel("anep");
    setResults(null);
    setCurrentStep(1);
  };

  const handleVideoUploaded = (file: File | null) => {
    if (!file) {
      setVideoFile(null);
      setVideoURL(null);
      setVideoMetadata(null);
      return;
    }

    const url = URL.createObjectURL(file);
    const video = document.createElement("video");
    video.preload = "metadata";
    video.src = url;

    video.onloadedmetadata = () => {
      setVideoFile(file);
      setVideoURL(url);
      setVideoMetadata({
        duration: video.duration,
        type: file.type,
      });
      URL.revokeObjectURL(video.src); // avoid memory leak
    };
  };

  const handleModelSelected = (modelId: string) => {
    setSelectedModel(modelId);
  };

  const handleAnalysisComplete = (results: any) => {
    setResults(results);
    goToNextStep();
  };

  const isNextDisabled =
    (currentStep === 1 && !videoFile) ||
    (currentStep === 2 && !selectedModel);

  return (
    <div className="flex flex-col space-y-8">
      {/* Stepper header - Desktop */}
      <StepIndicator steps={steps} currentStep={currentStep} />

      {/* Mobile stepper indicator */}
      <MobileStepIndicator steps={steps} currentStep={currentStep} />

      {/* Step content */}
      <div className="px-4 flex-1 max-w-4xl mx-auto w-full">
        {currentStep === 1 && (
          <UploadStep
            onVideoUploaded={handleVideoUploaded}
            initialFile={videoFile}
            initialURL={videoURL}
            initialMetadata={videoMetadata}
            className="animate-fade-in"
          />
        )}

        {currentStep === 2 && (
          <ModelSelectionStep
            onModelSelected={handleModelSelected}
            selectedModel={selectedModel}
            className="animate-fade-in"
          />
        )}

        {currentStep === 3 && videoFile && (
          <ConfirmationStep
            videoFile={videoFile}
            selectedModel={selectedModel}
            className="animate-fade-in"
          />
        )}

        {currentStep === 4 && videoFile && (
          <AnalysisStep
            videoFile={videoFile}
            selectedModel={selectedModel}
            onAnalysisComplete={handleAnalysisComplete}
            className="animate-fade-in"
          />
        )}

        {currentStep === 5 && results && (
          <ResultsStep
            results={results}
            onRestart={handleRestart}
            className="animate-fade-in"
          />
        )}
      </div>

      {/* Navigation buttons */}
      {currentStep !== 4 && currentStep !== 5 && (
        <NavigationButtons
          currentStep={currentStep}
          goToNextStep={goToNextStep}
          goToPreviousStep={goToPreviousStep}
          isNextDisabled={isNextDisabled}
        />
      )}
    </div>
  );
};

export default VideoAnalyzer;
