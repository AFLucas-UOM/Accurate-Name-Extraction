
import React from "react";
import { Step } from "./StepIndicator";

interface MobileStepIndicatorProps {
  steps: Step[];
  currentStep: number;
}

const MobileStepIndicator = ({ steps, currentStep }: MobileStepIndicatorProps) => {
  return (
    <div className="sm:hidden px-4">
      <div className="text-sm text-muted-foreground mb-1">
        Step {currentStep} of {steps.length}
      </div>
      <div className="text-lg font-medium flex items-center">
        {steps[currentStep - 1].icon}
        <span className="ml-2">{steps[currentStep - 1].title}</span>
      </div>
      <div className="w-full h-1 bg-gray-200 dark:bg-gray-700 mt-4 rounded-full overflow-hidden">
        <div
          className="h-full bg-primary transition-all duration-300"
          style={{ width: `${(currentStep / steps.length) * 100}%` }}
        />
      </div>
    </div>
  );
};

export default MobileStepIndicator;
