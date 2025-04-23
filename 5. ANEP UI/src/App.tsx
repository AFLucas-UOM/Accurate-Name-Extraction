import { Toaster as ShadcnToaster } from "@/components/ui/toaster";
import { Toaster as SonnerToaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";

import { useTheme } from "./hooks/use-theme";
import Index from "./pages/Index";
import NotFound from "./pages/NotFound";
import SurveyDashboard from "./pages/SurveyVisualizerCore";

const queryClient = new QueryClient();

// Theme initializer
const ThemeInitializer = ({ children }: { children: React.ReactNode }) => {
  useTheme(); // Applies theme preferences
  return <>{children}</>;
};

const App = () => (
  <QueryClientProvider client={queryClient}>
    <ThemeInitializer>
      <TooltipProvider>
        <ShadcnToaster />
        <SonnerToaster />
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<Index />} />
            <Route path="/survey" element={<SurveyDashboard />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </BrowserRouter>
      </TooltipProvider>
    </ThemeInitializer>
  </QueryClientProvider>
);

export default App;
