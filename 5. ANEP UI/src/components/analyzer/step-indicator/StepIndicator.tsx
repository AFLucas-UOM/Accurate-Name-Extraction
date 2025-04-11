
import React from "react";
import { CheckCircle } from "lucide-react";

export interface Step {
  number: number;
  title: string;
  icon: React.ReactNode;
}

interface StepIndicatorProps {
  steps: Step[];
  currentStep: number;
}

const StepIndicator = ({ steps, currentStep }: StepIndicatorProps) => {
  return (
    <div className="hidden sm:flex justify-center pt-8">
      <div className="flex items-center space-x-8">
        {steps.map((step, index) => (
          <div key={step.number} className="flex flex-col items-center">
            <div
              className={`flex flex-col items-center relative ${
                currentStep >= step.number
                  ? "text-primary"
                  : "text-gray-400"
              }`}
            >
              <div
                className={`w-14 h-14 rounded-full flex items-center justify-center mb-2 ${
                  currentStep > step.number
                    ? "bg-primary text-white"
                    : currentStep === step.number
                    ? "border-2 border-primary bg-white dark:bg-gray-800"
                    : "border-2 border-gray-300 dark:border-gray-600"
                }`}
              >
                {currentStep > step.number ? (
                  <CheckCircle className="h-6 w-6" />
                ) : (
                  React.cloneElement(step.icon as React.ReactElement, { className: "h-6 w-6" })
                )}
              </div>
              <div className="text-xs font-medium">{step.title}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default StepIndicator;
