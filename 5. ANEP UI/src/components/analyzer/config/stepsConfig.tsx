
import React from "react";
import { Upload, Layers, CheckCircle, BarChart, FileText } from "lucide-react";

export const defineSteps = () => [
  {
    number: 1,
    title: "Upload",
    icon: <Upload className="h-5 w-5" />,
  },
  {
    number: 2,
    title: "Model",
    icon: <Layers className="h-5 w-5" />,
  },
  {
    number: 3,
    title: "Confirm",
    icon: <CheckCircle className="h-5 w-5" />,
  },
  {
    number: 4,
    title: "Analyse",
    icon: <BarChart className="h-5 w-5" />,
  },
  {
    number: 5,
    title: "Results",
    icon: <FileText className="h-5 w-5" />,
  },
];
