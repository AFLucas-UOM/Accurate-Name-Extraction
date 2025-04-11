import React from "react";
import { Button } from "@/components/ui/button";
import { ArrowLeft, ArrowRight } from "lucide-react";

interface NavigationButtonsProps {
  currentStep: number;
  goToPreviousStep: () => void;
  goToNextStep: () => void;
  isNextDisabled: boolean;
}

const NavigationButtons = ({
  currentStep,
  goToPreviousStep,
  goToNextStep,
  isNextDisabled,
}: NavigationButtonsProps) => {
  return (
    <div className="px-4 flex justify-between max-w-4xl mx-auto w-full py-4">
      <Button
        variant="ghost"
        onClick={goToPreviousStep}
        disabled={currentStep === 1}
        className={currentStep === 1 ? "invisible" : ""}
      >
        <ArrowLeft className="h-4 w-4 mr-2" />
        Back
      </Button>

      <Button
        onClick={goToNextStep}
        disabled={isNextDisabled}
        className={isNextDisabled ? "cursor-not-allowed opacity-50" : ""}
      >
        Next
        <ArrowRight className="h-4 w-4 ml-2" />
      </Button>
    </div>
  );
};

export default NavigationButtons;
