import React, { useCallback, useEffect, useMemo, useState, Suspense } from "react";
import Papa from "papaparse";
import {
  BarChart as BarChartIcon,
  PieChart as PieChartIcon,
  LineChart as LineChartIcon,
  Loader2,
  Download,
  Filter,
  Layers,
  UserCircle2,
  X,
  Copy,
  Check,
  Info,
  BarChart2,
  ChevronDown,
  ChevronRight,
  MoreHorizontal,
  Share
} from "lucide-react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip as RechartTooltip,
  Legend,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
} from "recharts";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import NavBar from "@/components/layout/NavBar";
import Footer from "@/components/layout/Footer";

// Types
export type SurveyRow = Record<string, string | number | undefined>;
interface DashboardProps {
  csvUrl?: string;     
  title?: string;
  customPalette?: string[];
}

// Accessible palette (WCAG-AA compliant)
const DEFAULT_PALETTE = [
  "#2563eb", // Blue
  "#10b981", // Green
  "#f59e0b", // Amber
  "#ef4444", // Red
  "#8b5cf6", // Purple
  "#ec4899", // Pink
  "#14b8a6", // Teal
  "#f97316", // Orange
  "#22c55e", // Green
  "#6366f1", // Indigo
];

// Helper – persisted state
function useLocalState<T>(key: string, initial: T) {
  const [state, setState] = useState<T>(() => {
    try {
      const cached = localStorage.getItem(key);
      return cached ? (JSON.parse(cached) as T) : initial;
    } catch {
      return initial;
    }
  });
  const setPersisted = useCallback(
    (val: T) => {
      setState(val);
      try {
        localStorage.setItem(key, JSON.stringify(val));
      } catch {
        /* ignore quota errors */
      }
    },
    [key],
  );
  return [state, setPersisted] as const;
}

// Split headers into demographic and opinion questions
function splitHeaders(headers: string[]) {
  // Common demographic keywords
  const demographicKeywords = ["age", "gender", "reside", "location", "education", "income", "occupation", "ethnicity", "country"];

  // Phrases to exclude if they appear anywhere in the header (lowercase match)
  const excludeKeywords = [
    "confirmation of consent",
    "would you use such",
    "current employment status",
    "misinformation",
    "fake news"
  ];

  const filteredHeaders = headers.filter((h) => {
    const hLower = h.toLowerCase().trim();

    // EXCLUDE questions that match any phrase
    const isExcluded = excludeKeywords.some((kw) => hLower.includes(kw));

    const isJunk = (
      hLower.includes("timestamp") ||
      hLower.includes("time") ||
      hLower.includes("date")
    );

    return !isExcluded && !isJunk;
  });

  const demographics = filteredHeaders.filter((h) =>
    demographicKeywords.some((k) => new RegExp(`\\b${k}\\b`, "i").test(h))
  );

  const opinions = filteredHeaders.filter((h) => !demographics.includes(h));

  return { demographics, opinions } as const;
}

// Format percentages with proper rounding
function formatPercentage(value: number): string {
  return `${value.toFixed(1)}%`;
}

// Get color for accessibility
function getColorWithContrast(index: number, palette: string[]): string {
  return palette[index % palette.length];
}

// Dropdown Menu Component
const DropdownMenu = ({ 
  label, 
  icon,
  children
}: { 
  label: string; 
  icon?: React.ReactNode;
  children: React.ReactNode;
}) => {
  const [isOpen, setIsOpen] = useState(false);
  
  return (
    <div className="relative inline-block text-left">
      <Button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded-lg shadow-sm transition-colors"
      >
        {icon}
        {label}
        <ChevronDown className={`w-4 h-4 transition-transform ${isOpen ? "transform rotate-180" : ""}`} />
      </Button>
      
      {isOpen && (
        <div 
          className="absolute right-0 mt-2 w-56 rounded-md shadow-lg bg-white dark:bg-gray-800 ring-1 ring-black ring-opacity-5 z-10"
          onMouseLeave={() => setIsOpen(false)}
        >
          <div className="py-1" role="menu" aria-orientation="vertical">
            {children}
          </div>
        </div>
      )}
    </div>
  );
};

// Collapsible Section Component
const CollapsibleSection = ({
  title,
  icon,
  defaultOpen = true,
  children,
  className = "",
  type = "default", // "default", "filter", "insights"
  summary = null, // Optional summary to show when collapsed
}: {
  title: string;
  icon?: React.ReactNode;
  defaultOpen?: boolean;
  children: React.ReactNode;
  className?: string;
  type?: "default" | "filter" | "insights";
  summary?: React.ReactNode;
}) => {
  const [isOpen, setIsOpen] = useState(defaultOpen);
  
  let headerClasses = "w-full flex items-center justify-between p-4 rounded-xl transition-all duration-200";
  let containerClasses = "border rounded-xl shadow-md transition-all duration-300";
  
  // Apply style variants based on type
  if (type === "insights") {
    containerClasses += isOpen 
      ? " border-blue-300 dark:border-blue-700 bg-blue-50/50 dark:bg-blue-900/20 shadow-lg" 
      : " border-gray-200 dark:border-gray-700 hover:border-blue-200 dark:hover:border-blue-800";
      
    headerClasses += isOpen
      ? " bg-blue-100 dark:bg-blue-900/40 text-blue-800 dark:text-blue-300"
      : " bg-gray-50 dark:bg-gray-800 hover:bg-blue-50 dark:hover:bg-blue-900/30 text-gray-800 dark:text-gray-200";
  } else if (type === "filter") {
    containerClasses += isOpen 
      ? " border-blue-300 dark:border-blue-700 shadow-lg" 
      : " border-gray-200 dark:border-gray-700 hover:border-blue-200 dark:hover:border-blue-800";
      
    headerClasses += isOpen
      ? " bg-blue-50 dark:bg-blue-900/20 text-blue-800 dark:text-blue-300"
      : " bg-gray-50 dark:bg-gray-800 hover:bg-blue-50 dark:hover:bg-blue-900/20 text-gray-800 dark:text-gray-200";
  } else {
    containerClasses += " border-gray-200 dark:border-gray-700";
    headerClasses += " bg-gray-50 dark:bg-gray-800 hover:bg-gray-100 dark:hover:bg-gray-700";
  }
  
  return (
    <div className={`${containerClasses} ${className}`}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={headerClasses}
        aria-expanded={isOpen}
      >
        <div className="flex items-center gap-2 font-semibold text-lg">
          <div className={`${isOpen ? "bg-white/80 dark:bg-gray-800/50" : "bg-blue-100 dark:bg-blue-900/30"} p-2 rounded-full transition-colors duration-300`}>
            {icon}
          </div>
          {title}
          
          {/* Badge for insights section */}
          {type === "insights" && !isOpen && (
            <span className="ml-2 bg-blue-100 dark:bg-blue-900/50 text-blue-700 dark:text-blue-300 text-xs font-medium px-2.5 py-0.5 rounded-full">
              3 insights
            </span>
          )}
        </div>
        
        <div className="flex items-center gap-3">
          {/* Summary when collapsed */}
          {!isOpen && summary}
          
          <div className={`p-1 rounded-full ${isOpen ? "bg-white/80 dark:bg-gray-800/50" : "bg-blue-100 dark:bg-blue-900/30"} transition-colors duration-300`}>
            {isOpen ? (
              <ChevronDown className="w-5 h-5" />
            ) : (
              <ChevronRight className="w-5 h-5" />
            )}
          </div>
        </div>
      </button>
      
      <div 
        className="overflow-hidden transition-all duration-500 ease-in-out"
        style={{ 
          maxHeight: isOpen ? '2000px' : '0px',
          opacity: isOpen ? 1 : 0
        }}
      >
        <div className={`p-4 ${isOpen ? "animate-fadeIn" : ""}`}>
          {children}
        </div>
      </div>
    </div>
  );
};

// Error boundary component
class ErrorBoundary extends React.Component<
  { children: React.ReactNode; fallback: React.ReactNode },
  { hasError: boolean }
> {
  constructor(props: { children: React.ReactNode; fallback: React.ReactNode }) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback;
    }
    return this.props.children;
  }
}

// Loading component
const LoadingComponent = () => (
  <>
    <NavBar />
    <div className="flex flex-col items-center justify-center h-[60vh] text-muted-foreground text-center">
      <Loader2 className="w-10 h-10 animate-spin mb-4 text-blue-600" /> 
      <p className="text-xl font-medium">Loading survey data...</p>
      <p className="text-sm mt-2 text-gray-500">This may take a moment</p>
    </div>
    <Footer />
  </>
);

// Error component
const ErrorComponent = ({ error, onRetry }: { error: string; onRetry: () => void }) => (
  <>
    <NavBar />
    <div className="p-8 text-center">
      <div className="inline-flex items-center justify-center p-4 bg-red-50 rounded-full mb-4">
        <X className="w-8 h-8 text-red-600" />
      </div>
      <p className="text-xl font-semibold mb-2 text-red-600">Error Loading Survey Data</p>
      <p className="text-gray-600 max-w-lg mx-auto mb-6">{error}</p>
      <Button
        onClick={onRetry}
        className="bg-red-600 hover:bg-red-700 text-white"
      >
        Try Again
      </Button>
    </div>
    <Footer />
  </>
);

// Placeholder component when no question is selected
const NoQuestionSelectedPlaceholder = () => (
  <div className="h-64 flex flex-col items-center justify-center text-muted-foreground space-y-4">
    <Filter className="w-12 h-12 text-blue-300" />
    <p className="text-xl">Please select a question to visualize</p>
    <p className="text-sm max-w-md text-center">
      Choose a question from the dropdown above to see the distribution of responses
    </p>
  </div>
);

// Main Component
// Add keyframes for animations
const fadeInAnimation = `
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}
.animate-fadeIn {
  animation: fadeIn 0.3s ease-out forwards;
}

@keyframes expand {
  from { width: 0; }
  to { width: 100%; }
}
.animate-expand {
  animation: expand 0.8s ease-out forwards;
}
`;

export default function SurveyDashboard({ 
  csvUrl = "/ANEP.csv", 
  title = "Survey Results Dashboard",
  customPalette
}: DashboardProps) {
  // Raw data + header meta
  const [rows, setRows] = useState<SurveyRow[]>([]);
  const [headers, setHeaders] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  
  const [exportMenuOpen, setExportMenuOpen] = useState(false);
  const [controlsExpanded, setControlsExpanded] = useState(true);
  
  // Use custom palette if provided, otherwise use default
  const colorPalette = useMemo(() => customPalette || DEFAULT_PALETTE, [customPalette]);

  // UI state – persisted between sessions
  const [primaryQ, setPrimaryQ] = useLocalState<string>("sd_primaryQ", "");
  const [chartType, setChartType] = useLocalState<"bar" | "pie" | "line">("sd_chartType", "bar");
  const [filters, setFilters] = useLocalState<{ q: string; a: string }[]>("sd_filters", []);
  const [sortOrder, setSortOrder] = useLocalState<"asc" | "desc" | "none">("sd_sortOrder", "none");
  const [viewMode, setViewMode] = useLocalState<"chart" | "table">("sd_viewMode", "chart");
  const [demographicsCollapsed, setDemographicsCollapsed] = useState(false);

  // Fetch & parse CSV
  useEffect(() => {
    setLoading(true);
    Papa.parse(csvUrl, {
      download: true,
      header: true,
      skipEmptyLines: true,
      dynamicTyping: true,
      complete: (res) => {
        if (res.errors.length) {
          setError(res.errors[0].message);
        } else {
          setRows(res.data as SurveyRow[]);
          setHeaders(res.meta.fields || []);
          if (!primaryQ && res.meta.fields?.length) setPrimaryQ(res.meta.fields[0]);
        }
        setLoading(false);
      },
      error: (err) => {
        setError(err.message);
        setLoading(false);
      },
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [csvUrl]);

  const { demographics, opinions } = useMemo(() => splitHeaders(headers), [headers]);

  // Improved filtering
  const addFilter = (q: string, a: string) => {
    // Allow multiple filters for the same demographic question
    setFilters([...filters, { q, a }]);
  };
  
  const removeFilter = (idx: number) => setFilters(filters.filter((_, i) => i !== idx));
  const clearFilters = () => setFilters([]);

  // Filtered rows with improved OR logic for same-question filters
  const filteredRows = useMemo(() => {
    // Group filters by question
    const filtersByQuestion: Record<string, string[]> = {};
    
    filters.forEach(f => {
      if (!filtersByQuestion[f.q]) {
        filtersByQuestion[f.q] = [];
      }
      filtersByQuestion[f.q].push(f.a);
    });
    
    // Apply filters with OR logic within same question, AND between questions
    return rows.filter(row => {
      // Check if this row matches all filter groups
      return Object.entries(filtersByQuestion).every(([question, answers]) => {
        // For each question, check if row matches ANY of its filters (OR logic)
        return answers.some(answer => row[question] === answer);
      });
    });
  }, [rows, filters]);

  // Chart data with improved handling of complex responses (including brackets, etc.)
  const chartData = useMemo(() => {
    if (!primaryQ) return [];

    const counts: Record<string, number> = {};
    
// Safely split options by commas **outside parentheses**
function splitTopLevelAnswers(response: string): string[] {
  const parts: string[] = [];
  let current = "";
  let depth = 0;

  for (let i = 0; i < response.length; i++) {
    const char = response[i];

    if (char === "," && depth === 0) {
      parts.push(current.trim());
      current = "";
    } else {
      if (char === "(") depth++;
      if (char === ")") depth = Math.max(0, depth - 1);
      current += char;
    }
  }

  if (current) parts.push(current.trim());
  return parts.filter(Boolean);
}

// Determine if this answer should be split
const shouldSplitAnswer = (answer: string): boolean => {
  const alwaysSplitQuestions = [
    "Where do you mostly get your news from?",
    "Select all that apply"
  ];
  const isForceSplit = alwaysSplitQuestions.some(q => primaryQ.toLowerCase().includes(q.toLowerCase()));
  return isForceSplit && /[,;]/.test(answer); // Only split if it's one of those questions and there's a comma
};

// Count responses
filteredRows.forEach((r) => {
  const ans = r[primaryQ];

  if (ans !== undefined && ans !== "") {
    const answerString = String(ans).trim();

    if (shouldSplitAnswer(answerString)) {
      const individualAnswers = splitTopLevelAnswers(answerString);
      individualAnswers.forEach(answer => {
        counts[answer] = (counts[answer] || 0) + 1;
      });
    } else {
      counts[answerString] = (counts[answerString] || 0) + 1;
    }
  }
});


    const total = Object.values(counts).reduce((s, v) => s + v, 0);

    let result = Object.entries(counts).map(([name, value]) => ({
      name,
      value,
      percentage: total > 0 ? (value / total) * 100 : 0,
    }));

    // Apply sorting if requested
    if (sortOrder !== "none") {
      result = [...result].sort((a, b) => {
        if (sortOrder === "asc") {
          return a.value - b.value;
        } else {
          return b.value - a.value;
        }
      });
    }

    return result;
  }, [filteredRows, primaryQ, sortOrder]);

  // Export CSV with headers
  const exportCsv = () => {
    if (!chartData.length) return;
    const csv = ["Response,Count,Percentage", ...chartData.map((d) => `"${d.name.replace(/"/g, '""')}",${d.value},${d.percentage.toFixed(2)}`)].join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${primaryQ.replace(/[^a-z0-9]/gi, "_")}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };
  
  // Copy chart as image (enhanced version)
  const copyChart = () => {
    const chartElement = document.querySelector(".recharts-wrapper");
    if (!chartElement) {
      console.warn("Chart element not found");
      return;
    }

    const svgElement = chartElement.querySelector("svg");
    if (!svgElement) {
      console.warn("SVG element not found inside chart");
      return;
    }

    // Trigger copied indicator
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);

    // Get chart dimensions
    const { width, height } = chartElement.getBoundingClientRect();

    // Serialize SVG
    const svgData = new XMLSerializer().serializeToString(svgElement);
    const svgBlob = new Blob([svgData], { type: "image/svg+xml;charset=utf-8" });
    const svgUrl = URL.createObjectURL(svgBlob);

    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      // Create canvas
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;

      const ctx = canvas.getContext("2d");
      if (!ctx) {
        console.error("Failed to get canvas context");
        return;
      }

      // Draw white background
      ctx.fillStyle = "white";
      ctx.fillRect(0, 0, width, height);
      ctx.drawImage(img, 0, 0, width, height);

      // Convert to blob and copy to clipboard
      canvas.toBlob((blob) => {
        if (!blob) {
          console.error("Canvas toBlob failed");
          return;
        }

        // Copy to clipboard if supported
        if (navigator.clipboard && window.ClipboardItem) {
          const item = new ClipboardItem({ "image/png": blob });
          navigator.clipboard.write([item]).catch((err) => {
            console.error("Clipboard write failed:", err);
          });
        } else {
          // Fallback: prompt download
          const fallbackUrl = URL.createObjectURL(blob);
          const link = document.createElement("a");
          link.href = fallbackUrl;
          link.download = "chart.png";
          link.click();
          URL.revokeObjectURL(fallbackUrl);
        }
      });

      URL.revokeObjectURL(svgUrl);
    };

    img.onerror = () => {
      console.error("Failed to load SVG as image");
      URL.revokeObjectURL(svgUrl);
    };

    img.src = svgUrl;
  };

  // Short circuit to loading and error states
  if (loading) return <LoadingComponent />;
  if (error) return <ErrorComponent error={error} onRetry={() => window.location.reload()} />;

  // Derived values
  const hasQuestions = opinions.length > 0;
  const hasChartData = chartData.length > 0;

  return (
    <>
      <NavBar />
      <div className="mx-auto max-w-7xl px-4 py-8 space-y-8 bg-gray-50 dark:bg-[#0F1729] min-h-screen">
        {/* Header */}
        <header className="text-center space-y-3 mb-6 bg-blue-50 dark:bg-[#0F1729] p-6 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700">
          <h1 className="text-4xl font-bold flex justify-center items-center gap-3 text-blue-800 dark:text-white">
            <Layers className="w-10 h-10 text-blue-700 dark:text-blue-400" /> {title}
          </h1>
          <p className="text-muted-foreground text-base max-w-2xl mx-auto">
            Explore survey results with interactive charts and demographic filters
          </p>
          <div className="h-1 w-20 bg-blue-600 dark:bg-blue-600 mx-auto rounded-full mt-2"></div>
        </header>
        
        {/* Primary Question Selection */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md border border-gray-200 dark:border-gray-700 p-4">
          <label className="block mb-2 text-lg font-medium flex items-center gap-2 text-gray-800 dark:text-gray-200">
            <Filter className="w-5 h-5 text-blue-600" /> 
            <span>Select Question to Visualize</span>
          </label>
          
          <select
            className="w-full rounded-lg shadow-sm border border-gray-300 dark:border-gray-600 p-3 bg-white dark:bg-[#1E293B] text-base dark:text-gray-100"
            value={primaryQ}
            onChange={(e) => setPrimaryQ(e.target.value)}
          >
            <option value="" disabled>-- Select a question --</option>
            {opinions.map((q) => (
              <option key={q} value={q}>{q}</option>
            ))}
          </select>
          
          {!primaryQ && (
            <p className="text-sm text-blue-600 mt-1">Please select a question to visualize the results</p>
          )}
        </div>

        {/* Improved Controls Section */}
        <section className="bg-white dark:bg-gray-800 rounded-lg shadow-md border border-gray-200 dark:border-gray-700 mb-5">
          <div className="p-4 flex justify-between items-center border-b border-gray-200 dark:border-gray-700">
            <h2 className="text-lg font-medium text-gray-800 dark:text-gray-200 flex items-center gap-2">
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-blue-600 dark:text-blue-400">
                <path d="M12 20a8 8 0 1 0 0-16 8 8 0 0 0 0 16Z"></path>
                <path d="M12 14a2 2 0 1 0 0-4 2 2 0 0 0 0 4Z"></path>
                <path d="M12 2v2"></path>
                <path d="M12 22v-2"></path>
                <path d="m17 20.66-1-1.73"></path>
                <path d="M11 10.27 7 3.34"></path>
                <path d="m20.66 17-1.73-1"></path>
                <path d="m3.34 7 1.73 1"></path>
                <path d="M14 12h8"></path>
                <path d="M2 12h2"></path>
                <path d="m20.66 7-1.73 1"></path>
                <path d="m3.34 17 1.73-1"></path>
                <path d="m17 3.34-1 1.73"></path>
                <path d="m7 20.66 1-1.73"></path>
              </svg>
              Visualization Controls
            </h2>
            <button
              onClick={() => setControlsExpanded(!controlsExpanded)}
              className="text-gray-500 hover:text-blue-600 dark:text-gray-400 dark:hover:text-blue-400 transition-colors"
            >
              {controlsExpanded ? (
                <ChevronDown className="w-5 h-5" />
              ) : (
                <ChevronRight className="w-5 h-5" />
              )}
            </button>
          </div>
          
          {controlsExpanded && (
            <div className="p-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Display Mode */}
                <div>
                  <label className="block mb-2 text-sm font-medium text-gray-700 dark:text-gray-300">
                    Display Mode
                  </label>
                  <div className="flex gap-3">
                    <button 
                      onClick={() => setViewMode("chart")}
                      className={`relative overflow-hidden flex-1 p-3 rounded-lg flex justify-center items-center gap-2 transition-all ${
                        viewMode === "chart" 
                          ? "bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300 ring-2 ring-blue-400 dark:ring-blue-600 font-medium" 
                          : "bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700"
                      }`}
                    >
                      <BarChart2 className="w-5 h-5" />
                      <span>Chart</span>
                      {viewMode === "chart" && (
                        <span className="absolute inset-x-0 bottom-0 h-0.5 bg-blue-500 dark:bg-blue-600"></span>
                      )}
                    </button>
                    <button 
                      onClick={() => setViewMode("table")}
                      className={`relative overflow-hidden flex-1 p-3 rounded-lg flex justify-center items-center gap-2 transition-all ${
                        viewMode === "table" 
                          ? "bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300 ring-2 ring-blue-400 dark:ring-blue-600 font-medium" 
                          : "bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700"
                      }`}
                    >
                      <Layers className="w-5 h-5" />
                      <span>Table</span>
                      {viewMode === "table" && (
                        <span className="absolute inset-x-0 bottom-0 h-0.5 bg-blue-500 dark:bg-blue-600"></span>
                      )}
                    </button>
                  </div>
                </div>
                
                {/* Chart Type Options - Only show when Chart view is selected */}
                {viewMode === "chart" && (
                  <div>
                    <label className="block mb-2 text-sm font-medium text-gray-700 dark:text-gray-300">
                      Chart Type
                    </label>
                    <div className="flex gap-3">
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <button 
                              onClick={() => setChartType("bar")}
                              className={`relative overflow-hidden flex-1 p-3 rounded-lg flex justify-center items-center gap-2 transition-all ${
                                chartType === "bar" 
                                  ? "bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300 ring-2 ring-blue-400 dark:ring-blue-600 font-medium" 
                                  : "bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700"
                              }`}
                            >
                              <BarChartIcon className="w-5 h-5" />
                              <span>Bar</span>
                              {chartType === "bar" && (
                                <span className="absolute inset-x-0 bottom-0 h-0.5 bg-blue-500 dark:bg-blue-600"></span>
                              )}
                            </button>
                          </TooltipTrigger>
                          <TooltipContent>
                            <p>Bar chart - good for comparing values</p>
                          </TooltipContent>
                        </Tooltip>

                        <Tooltip>
                          <TooltipTrigger asChild>
                            <button 
                              onClick={() => setChartType("pie")}
                              className={`relative overflow-hidden flex-1 p-3 rounded-lg flex justify-center items-center gap-2 transition-all ${
                                chartType === "pie" 
                                  ? "bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300 ring-2 ring-blue-400 dark:ring-blue-600 font-medium" 
                                  : "bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700"
                              }`}
                            >
                              <PieChartIcon className="w-5 h-5" />
                              <span>Pie</span>
                              {chartType === "pie" && (
                                <span className="absolute inset-x-0 bottom-0 h-0.5 bg-blue-500 dark:bg-blue-600"></span>
                              )}
                            </button>
                          </TooltipTrigger>
                          <TooltipContent>
                            <p>Pie chart - good for showing proportions</p>
                          </TooltipContent>
                        </Tooltip>

                        <Tooltip>
                          <TooltipTrigger asChild>
                            <button 
                              onClick={() => setChartType("line")}
                              className={`relative overflow-hidden flex-1 p-3 rounded-lg flex justify-center items-center gap-2 transition-all ${
                                chartType === "line" 
                                  ? "bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300 ring-2 ring-blue-400 dark:ring-blue-600 font-medium"
                                  : "bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700"
                              }`}
                            >
                              <LineChartIcon className="w-5 h-5" />
                              <span>Line</span>
                              {chartType === "line" && (
                                <span className="absolute inset-x-0 bottom-0 h-0.5 bg-blue-500 dark:bg-blue-600"></span>
                              )}
                            </button>
                          </TooltipTrigger>
                          <TooltipContent>
                            <p>Line chart - good for showing trends</p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    </div>
                  </div>
                )}
                
                {/* Sort Order */}
                <div className="md:col-span-2">
                  <label className="block mb-2 text-sm font-medium text-gray-700 dark:text-gray-300">
                    Sort Order
                  </label>
                  <div className="flex flex-wrap gap-3">
                    <button 
                      onClick={() => setSortOrder("none")}
                      className={`relative overflow-hidden flex-1 p-3 rounded-lg flex justify-center items-center gap-2 transition-all ${
                        sortOrder === "none" 
                          ? "bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300 ring-2 ring-blue-400 dark:ring-blue-600 font-medium" 
                          : "bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700"
                      }`}
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <rect width="21" height="6" x="2" y="3" rx="2"></rect>
                        <rect width="15" height="6" x="2" y="9" rx="2"></rect>
                        <rect width="17" height="6" x="2" y="15" rx="2"></rect>
                      </svg>
                      <span>Original Order</span>
                      {sortOrder === "none" && (
                        <span className="absolute inset-x-0 bottom-0 h-0.5 bg-blue-500 dark:bg-blue-600"></span>
                      )}
                    </button>
                    <button 
                      onClick={() => setSortOrder("asc")}
                      className={`relative overflow-hidden flex-1 p-3 rounded-lg flex justify-center items-center gap-2 transition-all ${
                        sortOrder === "asc" 
                          ? "bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300 ring-2 ring-blue-400 dark:ring-blue-600 font-medium" 
                          : "bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700"
                      }`}
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="m3 8 4-4 4 4"></path>
                        <path d="M7 4v16"></path>
                        <path d="M11 12h4"></path>
                        <path d="M11 16h7"></path>
                        <path d="M11 20h10"></path>
                      </svg>
                      <span>Ascending (Low to High)</span>
                      {sortOrder === "asc" && (
                        <span className="absolute inset-x-0 bottom-0 h-0.5 bg-blue-500 dark:bg-blue-600"></span>
                      )}
                    </button>
                    <button 
                      onClick={() => setSortOrder("desc")}
                      className={`relative overflow-hidden flex-1 p-3 rounded-lg flex justify-center items-center gap-2 transition-all ${
                        sortOrder === "desc" 
                          ? "bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300 ring-2 ring-blue-400 dark:ring-blue-600 font-medium" 
                          : "bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700"
                      }`}
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M3 16l4 4 4-4"></path>
                        <path d="M7 20V4"></path>
                        <path d="M11 4h10"></path>
                        <path d="M11 8h7"></path>
                        <path d="M11 12h4"></path>
                      </svg>
                      <span>Descending (High to Low)</span>
                      {sortOrder === "desc" && (
                        <span className="absolute inset-x-0 bottom-0 h-0.5 bg-blue-500 dark:bg-blue-600"></span>
                      )}
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}
        </section>

        {/* Demographic Filters - Less Boxy Design */}
        {demographics.length > 0 && (
          <CollapsibleSection 
            title="Filter by Demographics" 
            icon={<UserCircle2 className="w-5 h-5 text-blue-600" />}
            defaultOpen={true}
            type="filter"
            summary={
              filters.length > 0 ? (
                <div className="flex items-center">
                  <span className="text-xs font-medium bg-blue-100 dark:bg-blue-900/50 text-blue-700 dark:text-blue-300 px-2.5 py-1 rounded-full">
                    {filters.length} active filter{filters.length !== 1 ? 's' : ''}
                  </span>
                </div>
              ) : null
            }
          >
            {/* Active Filters */}
            {filters.length > 0 && (
              <div className="mb-4 rounded-lg overflow-hidden">
                <div className="flex items-center justify-between px-4 py-2 bg-blue-50 dark:bg-blue-900/30 border-b border-blue-100 dark:border-blue-800">
                  <span className="text-sm font-medium text-blue-700 dark:text-blue-300 flex items-center gap-1">
                    <Filter className="w-4 h-4" /> Active filters
                  </span>
                  <Button 
                    variant="outline" 
                    size="sm" 
                    onClick={clearFilters}
                    className="flex items-center gap-1 bg-transparent hover:bg-red-50 dark:hover:bg-red-900/30 text-red-600 dark:text-red-400 hover:text-red-700 dark:hover:text-red-300 border-red-200 dark:border-red-800 hover:border-red-300 dark:hover:border-red-700 transition-all"
                  >
                    <X className="w-4 h-4" /> Clear All
                  </Button>
                </div>
                
                <div className="flex flex-wrap gap-2 p-3 bg-white dark:bg-gray-800">
                  {filters.map((f, i) => (
                    <span key={i} className="flex items-center gap-1 bg-blue-50 dark:bg-blue-900/30 text-blue-800 dark:text-blue-200 px-3 py-1 rounded-full text-sm border border-blue-100 dark:border-blue-800/50 transition-all hover:bg-blue-100 dark:hover:bg-blue-900/50">
                      <span className="font-medium">{f.q}:</span> {f.a}
                      <button 
                        onClick={() => removeFilter(i)}
                        className="ml-1 hover:bg-blue-200 dark:hover:bg-blue-800 rounded-full p-1 transition-colors"
                        aria-label={`Remove ${f.q} filter`}
                      >
                        <X className="w-3 h-3" />
                      </button>
                    </span>
                  ))}
                </div>
              </div>
            )}
            
            {/* Filter Selectors */}
            <div className="px-1">
              {/* Group filters by 3 in a row */}
              {demographics.reduce((rows, q, index) => {
                if (index % 3 === 0) rows.push([]);
                rows[rows.length - 1].push(q);
                return rows;
              }, [] as string[][]).map((row, rowIndex) => (
                <div key={rowIndex} className="mb-4 flex flex-col md:flex-row gap-4">
                  {row.map(q => {
                    const options = Array.from(new Set(rows.map((r) => String(r[q] ?? "")).filter(Boolean)));
                    const activeFiltersCount = filters.filter(f => f.q === q).length;
                    
                    return (
                      <div key={q} className="flex-1">
                        <div className="mb-1 flex items-center gap-1">
                          <label className={`text-sm font-medium ${activeFiltersCount > 0 ? "text-blue-700 dark:text-blue-400" : "text-gray-700 dark:text-gray-300"}`}>
                            {q} 
                            {activeFiltersCount > 0 && (
                              <span className="text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/30 px-1.5 py-0.5 rounded-full text-xs ml-1">
                                {activeFiltersCount}
                              </span>
                            )}
                          </label>
                          
                          <TooltipProvider>
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <div>
                                  <Info className="w-3 h-3 text-gray-400 hover:text-blue-500 cursor-help ml-1" />
                                </div>
                              </TooltipTrigger>
                              <TooltipContent side="top">
                                <p className="text-xs">You can select multiple values from this filter</p>
                              </TooltipContent>
                            </Tooltip>
                          </TooltipProvider>
                        </div>
                        
                        <select
                          className={`w-full border rounded-md p-2 text-sm bg-white dark:bg-[#1E293B] shadow-sm focus:ring focus:ring-opacity-50 dark:text-gray-200 ${
                            activeFiltersCount > 0
                              ? "border-blue-300 dark:border-blue-700 focus:border-blue-500 focus:ring-blue-200"
                              : "border-gray-300 dark:border-gray-600 focus:border-blue-500 focus:ring-blue-200"
                          }`}
                          onChange={(e) => e.target.value && addFilter(q, e.target.value)}
                          value=""
                          aria-label={`Filter by ${q}`}
                        >
                          <option value="">All options...</option>
                          {options.map((opt) => {
                            const isAlreadyFiltered = filters.some(f => f.q === q && f.a === opt);
                            return (
                              <option 
                                key={opt} 
                                value={opt}
                                style={isAlreadyFiltered ? { color: '#3b82f6', fontWeight: 'bold' } : {}}
                              >
                                {opt} {isAlreadyFiltered ? '(active)' : ''}
                              </option>
                            );
                          })}
                        </select>
                      </div>
                    );
                  })}
                </div>
              ))}
            </div>
            
            {/* Stats Footer */}
            <div className="flex justify-between items-center mt-2 px-3 py-2 bg-gray-50 dark:bg-gray-800/30 rounded-lg">
              <p className="text-sm flex items-center gap-2">
                <UserCircle2 className="w-4 h-4 text-blue-600 dark:text-blue-400" /> 
                <span>
                  <strong>{filteredRows.length}</strong> of <strong>{rows.length}</strong> responses 
                  <span className="text-gray-500 dark:text-gray-400 ml-1">
                    ({rows.length > 0 ? ((filteredRows.length / rows.length) * 100).toFixed(1) : 0}%)
                  </span>
                </span>
              </p>
              
              {filters.length > 0 && (
                <p className="text-xs bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400 px-2 py-1 rounded-full">
                  Filtered results
                </p>
              )}
            </div>
          </CollapsibleSection>
        )}

        {/* Action Row with Export & Share Dropdown */}
        <div className="flex flex-wrap gap-3 justify-between items-center mt-6">
          <p className="text-sm text-gray-600 dark:text-gray-300">
            {filteredRows.length > 0
              ? `Showing results from ${filteredRows.length} respondents`
              : "No data with current filters. Try adjusting your filters."
            }
          </p>

          <div className="flex gap-2 flex-wrap">
          {/* Improved Export & Share Dropdown */}
          <div className="relative">
            <Button
              onClick={() => setExportMenuOpen(!exportMenuOpen)}
              className="flex items-center gap-2 bg-gradient-to-br from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white py-2 px-4 rounded-lg shadow-sm transition-all"
              aria-expanded={exportMenuOpen}
            >
              <Share className="w-4 h-4" />
              <span>Export & Share</span>
              <ChevronDown className={`w-4 h-4 transition-transform ${exportMenuOpen ? "transform rotate-180" : ""}`} />
            </Button>
            
            {exportMenuOpen && (
              <div 
                className="absolute right-0 mt-2 w-64 rounded-lg shadow-lg bg-white dark:bg-gray-800 ring-1 ring-black ring-opacity-5 z-10 divide-y divide-gray-200 dark:divide-gray-700 animate-fadeIn"
                onMouseLeave={() => setExportMenuOpen(false)}
              >
                <div className="p-2 bg-blue-50 dark:bg-blue-900/30 rounded-t-lg">
                  <h3 className="text-sm font-medium text-blue-700 dark:text-blue-300 px-2 py-1">Data Export Options</h3>
                </div>
                <div className="py-1">
                  {/* Copy Chart Button - Only show when chart is visible */}
                  <button
                    onClick={copyChart}
                    className={`w-full text-left px-4 py-2.5 text-sm hover:bg-blue-50 dark:hover:bg-blue-900/20 flex items-center gap-2 transition-colors ${
                      chartData.length === 0 || viewMode === "table" 
                        ? 'text-gray-400 dark:text-gray-500 cursor-not-allowed' 
                        : 'text-gray-700 dark:text-gray-200'
                    }`}
                    disabled={chartData.length === 0 || viewMode === "table"}
                  >
                    {copied ? <Check className="w-4 h-4 text-green-500" /> : <Copy className="w-4 h-4" />}
                    <div>
                      <div className="font-medium">{copied ? "Chart Copied!" : "Copy Chart as Image"}</div>
                      <div className="text-xs text-gray-500 dark:text-gray-400">Add to clipboard for pasting elsewhere</div>
                    </div>
                  </button>
                </div>
                
                <div className="py-1">
                  {/* Export Filtered Data */}
                  <button
                    onClick={exportCsv}
                    className={`w-full text-left px-4 py-2.5 text-sm hover:bg-blue-50 dark:hover:bg-blue-900/20 flex items-center gap-2 transition-colors ${
                      chartData.length === 0 
                        ? 'text-gray-400 dark:text-gray-500 cursor-not-allowed' 
                        : 'text-gray-700 dark:text-gray-200'
                    }`}
                    disabled={chartData.length === 0}
                  >
                    <Download className="w-4 h-4" />
                    <div>
                      <div className="font-medium">Export Current View</div>
                      <div className="text-xs text-gray-500 dark:text-gray-400">Download visible data as CSV</div>
                    </div>
                  </button>

                  {/* Export Full Dataset */}
                  <button
                    onClick={() => {
                      const a = document.createElement("a");
                      a.href = csvUrl;
                      a.download = csvUrl.split('/').pop() || "survey_data.csv";
                      a.click();
                    }}
                    className="w-full text-left px-4 py-2.5 text-sm text-gray-700 dark:text-gray-200 hover:bg-blue-50 dark:hover:bg-blue-900/20 flex items-center gap-2 transition-colors"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                      <polyline points="7 10 12 15 17 10"></polyline>
                      <line x1="12" y1="15" x2="12" y2="3"></line>
                    </svg>
                    <div>
                      <div className="font-medium">Download Full Dataset</div>
                      <div className="text-xs text-gray-500 dark:text-gray-400">Get the complete survey data</div>
                    </div>
                  </button>
                </div>
              </div>
            )}
          </div>
          </div>
        </div>

        {/* Main Visualization Area */}
        <section className="bg-white dark:bg-[#1E293B] p-6 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 transition-all hover:shadow-xl">
          {primaryQ ? (
            <>
              <h2 className="text-xl font-semibold mb-6 text-center flex items-center justify-center gap-2 bg-blue-50 dark:bg-blue-900/30 p-3 rounded-lg shadow-sm">
                <div className="bg-blue-100 dark:bg-blue-800 p-2 rounded-full shadow-inner">
                  {viewMode === "chart" ? (
                    <>
                      {chartType === "bar" && <BarChartIcon className="w-6 h-6 text-blue-600 dark:text-blue-400" />}
                      {chartType === "pie" && <PieChartIcon className="w-6 h-6 text-blue-600 dark:text-blue-400" />}
                      {chartType === "line" && <LineChartIcon className="w-6 h-6 text-blue-600 dark:text-blue-400" />}
                    </>
                  ) : (
                    <Layers className="w-6 h-6 text-blue-600 dark:text-blue-400" />
                  )}
                </div>
                <span className="text-blue-800 dark:text-blue-300 truncate max-w-lg">{primaryQ}</span>
              </h2>
              
              {chartData.length === 0 ? (
                <div className="h-64 flex items-center justify-center text-muted-foreground">
                  No data available for this question with current filters
                </div>
              ) : viewMode === "table" ? (
                /* Table View */
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                    <thead className="bg-gray-50 dark:bg-gray-800">
                      <tr>
                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                          Response
                        </th>
                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                          Count
                        </th>
                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                          Percentage
                        </th>
                      </tr>
                    </thead>
                    <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700">
                      {chartData.map((item, index) => (
                        <tr key={index} className={index % 2 === 0 ? 'bg-white dark:bg-gray-900' : 'bg-gray-50 dark:bg-gray-800'}>
                          <td className="px-6 py-4 whitespace-normal text-sm text-gray-900 dark:text-gray-100 break-words max-w-xs">
                            {item.name}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">
                            {item.value}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">
                            {formatPercentage(item.percentage)}
                          </td>
                        </tr>
                      ))}
                      {/* Summary row */}
                      <tr className="bg-blue-50 dark:bg-blue-900/30 font-medium">
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100">
                          Total
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100">
                          {chartData.reduce((sum, item) => sum + item.value, 0)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100">
                          100%
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              ) : (
                /* Chart View */
                <ErrorBoundary
                  fallback={
                    <div className="h-64 flex items-center justify-center text-red-500">
                      <p>An error occurred while rendering the chart.</p>
                    </div>
                  }
                >
                  <Suspense fallback={<div className="h-64 flex items-center justify-center"><Loader2 className="w-10 h-10 animate-spin" /></div>}>
                    <div className="h-[500px]">
                    {chartType === "bar" && (
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart 
                          data={chartData} 
                          margin={{ top: 20, right: 30, left: 20, bottom: 90 }}
                          barCategoryGap="20%"
                        >
                          <CartesianGrid strokeDasharray="3 3" stroke="#ccc" />
                          <XAxis 
                            dataKey="name" 
                            angle={-45} 
                            textAnchor="end" 
                            height={90} 
                            interval={0}
                            tick={{ fontSize: 12 }}
                            tickMargin={20}
                          />
                          <YAxis 
                            allowDecimals={false} 
                            tickFormatter={(value) => value.toLocaleString()}
                          />
                          <RechartTooltip 
                            formatter={(value, name) => [
                              `${value.toLocaleString()} responses (${chartData.find(d => d.value === value)?.percentage.toFixed(1)}%)`, 
                              "Count"
                            ]}
                            contentStyle={{ 
                              backgroundColor: 'rgba(255, 255, 255, 0.95)',
                              border: '1px solid #ccc',
                              borderRadius: '4px',
                              padding: '10px',
                              boxShadow: '0 2px 5px rgba(0,0,0,0.15)',
                              fontWeight: 500,
                              color: '#333'
                            }}
                            labelStyle={{ fontWeight: 'bold', color: '#000' }}
                            itemStyle={{ color: '#333' }}
                            cursor={{ fill: 'rgba(200, 200, 200, 0.2)' }}
                          />
                          <Legend wrapperStyle={{ paddingTop: 20 }} />
                          <Bar 
                            dataKey="value" 
                            name="Count" 
                            fill={colorPalette[0]} 
                            barSize={60} 
                            animationDuration={1000}
                          />
                        </BarChart>
                      </ResponsiveContainer>
                    )}

                    {chartType === "pie" && (
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                          <Pie 
                            data={chartData} 
                            dataKey="value" 
                            nameKey="name" 
                            cx="50%" 
                            cy="50%" 
                            outerRadius={180} 
                            labelLine={true}
                            label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(1)}%`}
                            animationDuration={1000}
                          >
                            {chartData.map((_, idx) => (
                              <Cell key={`cell-${idx}`} fill={getColorWithContrast(idx, colorPalette)} />
                            ))}
                          </Pie>
                          <Legend 
                            layout="horizontal" 
                            verticalAlign="bottom" 
                            align="center"
                            formatter={(value) => {
                              const entry = chartData.find(d => d.name === value);
                              return `${value} (${entry?.value.toLocaleString()} responses)`;
                            }}
                            wrapperStyle={{ paddingTop: 20 }}
                          />
                          <RechartTooltip 
                            formatter={(value) => `${value.toLocaleString()} responses`} 
                            contentStyle={{ 
                              backgroundColor: 'rgba(255, 255, 255, 0.95)',
                              border: '1px solid #ccc', 
                              borderRadius: '4px',
                              padding: '10px',
                              boxShadow: '0 2px 5px rgba(0,0,0,0.15)',
                              fontWeight: 500,
                              color: '#333'
                            }}
                            labelStyle={{ fontWeight: 'bold', color: '#000' }}
                            itemStyle={{ color: '#333' }}
                          />
                        </PieChart>
                      </ResponsiveContainer>
                    )}

                    {chartType === "line" && (
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 90 }}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis 
                            dataKey="name" 
                            angle={-45} 
                            textAnchor="end" 
                            height={90} 
                            interval={0}
                            tick={{ fontSize: 12 }}
                            tickMargin={20}
                          />
                          <YAxis 
                            allowDecimals={false}
                            tickFormatter={(value) => value.toLocaleString()} 
                          />
                          <RechartTooltip 
                            formatter={(value) => [`${value.toLocaleString()} responses`, "Count"]} 
                            contentStyle={{ 
                              backgroundColor: 'rgba(255, 255, 255, 0.95)',
                              border: '1px solid #ccc', 
                              borderRadius: '4px',
                              padding: '10px',
                              boxShadow: '0 2px 5px rgba(0,0,0,0.15)',
                              fontWeight: 500,
                              color: '#333'
                            }}
                            labelStyle={{ fontWeight: 'bold', color: '#000' }}
                            itemStyle={{ color: '#333' }}
                          />
                          <Legend wrapperStyle={{ paddingTop: 20 }} />
                          <Line 
                            type="monotone" 
                            dataKey="value" 
                            stroke={colorPalette[0]} 
                            strokeWidth={3}
                            activeDot={{ r: 8 }}
                            animationDuration={1000}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    )}
                    </div>
                  </Suspense>
                </ErrorBoundary>
              )}
            </>
          ) : (
            <NoQuestionSelectedPlaceholder />
          )}
        </section>

        {/* Key Insights Section - Redesigned for clarity */}
        {hasQuestions && hasChartData && (
          <CollapsibleSection 
            title="Key Insights" 
            icon={<Info className="w-5 h-5 text-blue-600" />}
            defaultOpen={false}
            type="insights"
            summary={
              <div className="flex items-center">
                <span className="text-sm flex items-center gap-1.5 text-blue-600 dark:text-blue-400">
                  <span className="hidden md:inline">Top answer:</span>
                  <span className="font-medium">{[...chartData].sort((a, b) => b.value - a.value)[0].name.length > 25 ? [...chartData].sort((a, b) => b.value - a.value)[0].name.substring(0, 25) + '...' : [...chartData].sort((a, b) => b.value - a.value)[0].name}</span>
                  <span className="text-xs bg-blue-100 dark:bg-blue-900/40 px-1.5 py-0.5 rounded text-blue-700 dark:text-blue-300">
                    {formatPercentage([...chartData].sort((a, b) => b.value - a.value)[0].percentage)}
                  </span>
                </span>
              </div>
            }
          >
            <div className="space-y-6 mb-2">
              <div className="flex items-center justify-between border-b border-gray-200 dark:border-gray-700 pb-3">
                <h3 className="text-lg font-medium text-gray-800 dark:text-gray-200 flex items-center gap-2">
                  <span className="bg-blue-100 dark:bg-blue-900/50 p-1.5 rounded-full">
                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-blue-600 dark:text-blue-400">
                      <path d="M2.97 12.92A2 2 0 0 0 2 14.63v3.24a2 2 0 0 0 .97 1.71l3 1.8a2 2 0 0 0 2.06 0L12 19v-5.5l-5-3-4.03 2.42Z"></path>
                      <path d="m7 16.5-4.74-2.85"></path>
                      <path d="m7 16.5 5-3"></path>
                      <path d="M7 16.5v5.17"></path>
                      <path d="M12 13.5V19l3.97 2.38a2 2 0 0 0 2.06 0l3-1.8a2 2 0 0 0 .97-1.71v-3.24a2 2 0 0 0-.97-1.71L17 10.5l-5 3Z"></path>
                      <path d="m17 16.5-5-3"></path>
                      <path d="m17 16.5 4.74-2.85"></path>
                      <path d="M17 16.5v5.17"></path>
                      <path d="M7.97 4.42A2 2 0 0 0 7 6.13v4.37l5 3 5-3V6.13a2 2 0 0 0-.97-1.71l-3-1.8a2 2 0 0 0-2.06 0l-3 1.8Z"></path>
                      <path d="M12 8 7.26 5.15"></path>
                      <path d="m12 8 4.74-2.85"></path>
                      <path d="M12 13.5V8"></path>
                    </svg>
                  </span>
                  Analysis of "{primaryQ.length > 40 ? primaryQ.substring(0, 40) + '...' : primaryQ}"
                </h3>
                
                <div className="flex items-center gap-2">
                  <span className="text-sm text-gray-600 dark:text-gray-400">
                    {chartData.reduce((sum, item) => sum + item.value, 0).toLocaleString()} responses
                  </span>
                </div>
              </div>
              
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Distribution Overview */}
                <div className="lg:col-span-2 bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
                  <div className="p-4 border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/80">
                    <h4 className="font-medium text-gray-800 dark:text-gray-200 flex items-center gap-2">
                      <BarChartIcon className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                      Response Distribution
                    </h4>
                  </div>
                  
                  <div className="p-4">
                    {/* Top 5 responses with horizontal bars */}
                    <div className="space-y-4">
                      {[...chartData]
                        .sort((a, b) => b.value - a.value)
                        .slice(0, 5)
                        .map((item, index) => (
                          <div key={index} className="space-y-1">
                            <div className="flex justify-between text-sm">
                              <span className="font-medium text-gray-700 dark:text-gray-300 truncate max-w-[60%]" title={item.name}>
                                {item.name.length > 40 ? item.name.substring(0, 40) + '...' : item.name}
                              </span>
                              <span className="text-gray-600 dark:text-gray-400">
                                {item.value.toLocaleString()} ({formatPercentage(item.percentage)})
                              </span>
                            </div>
                            <div className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                              <div 
                                className="h-full rounded-full transition-all duration-500 animate-expand"
                                style={{ 
                                  width: `${item.percentage}%`,
                                  backgroundColor: getColorWithContrast(index, colorPalette)
                                }}
                              />
                            </div>
                          </div>
                        ))
                      }
                    </div>
                    
                    {/* Other responses */}
                    {chartData.length > 5 && (
                      <div className="mt-4 pt-3 border-t border-gray-200 dark:border-gray-700">
                        <details>
                          <summary className="text-sm text-blue-600 dark:text-blue-400 cursor-pointer">
                            {chartData.length - 5} more responses...
                          </summary>
                          <div className="mt-3 space-y-3 pl-2">
                            {[...chartData]
                              .sort((a, b) => b.value - a.value)
                              .slice(5)
                              .map((item, index) => (
                                <div key={index} className="flex justify-between text-sm">
                                  <span className="text-gray-700 dark:text-gray-300 truncate max-w-[60%]" title={item.name}>
                                    {item.name.length > 40 ? item.name.substring(0, 40) + '...' : item.name}
                                  </span>
                                  <span className="text-gray-600 dark:text-gray-400">
                                    {item.value.toLocaleString()} ({formatPercentage(item.percentage)})
                                  </span>
                                </div>
                              ))
                            }
                          </div>
                        </details>
                      </div>
                    )}
                  </div>
                </div>
                
                {/* Key Metrics */}
                <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
                  <div className="p-4 border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/80">
                    <h4 className="font-medium text-gray-800 dark:text-gray-200 flex items-center gap-2">
                      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-blue-600 dark:text-blue-400">
                        <path d="M10 2v4c0 1.1-.9 2-2 2H4"></path>
                        <path d="M4 22V10c0-1.1.9-2 2-2h8m2-2v4c0 1.1.9 2 2 2h4M22 22V10c0-1.1-.9-2-2-2h-8"></path>
                        <path d="M8 16h8"></path>
                        <path d="M8 19h8"></path>
                      </svg>
                      Key Metrics
                    </h4>
                  </div>
                  
                  <div className="p-4 divide-y divide-gray-200 dark:divide-gray-700">
                    {/* Most Common */}
                    <div className="pb-3 space-y-1">
                      <div className="text-sm text-gray-600 dark:text-gray-400">Most Common</div>
                      <div className="font-medium text-gray-900 dark:text-gray-100">
                        {chartData.length > 0 ? (
                          <div className="flex items-center gap-2">
                            <span className="text-base truncate max-w-full" title={[...chartData].sort((a, b) => b.value - a.value)[0].name}>
                              {[...chartData].sort((a, b) => b.value - a.value)[0].name}
                            </span>
                          </div>
                        ) : "No data"}
                      </div>
                      {chartData.length > 0 && (
                        <div className="flex items-center gap-1 text-sm">
                          <span className="font-medium text-blue-600 dark:text-blue-400">
                            {[...chartData].sort((a, b) => b.value - a.value)[0].value.toLocaleString()}
                          </span>
                          <span className="text-gray-500 dark:text-gray-400">
                            ({formatPercentage([...chartData].sort((a, b) => b.value - a.value)[0].percentage)})
                          </span>
                        </div>
                      )}
                    </div>
                    
                    {/* Least Common */}
                    <div className="py-3 space-y-1">
                      <div className="text-sm text-gray-600 dark:text-gray-400">Least Common</div>
                      <div className="font-medium text-gray-900 dark:text-gray-100">
                        {chartData.length > 0 ? (
                          <div className="flex items-center gap-2">
                            <span className="text-base truncate max-w-full" title={[...chartData].sort((a, b) => a.value - b.value)[0].name}>
                              {[...chartData].sort((a, b) => a.value - b.value)[0].name}
                            </span>
                          </div>
                        ) : "No data"}
                      </div>
                      {chartData.length > 0 && (
                        <div className="flex items-center gap-1 text-sm">
                          <span className="font-medium text-blue-600 dark:text-blue-400">
                            {[...chartData].sort((a, b) => a.value - b.value)[0].value.toLocaleString()}
                          </span>
                          <span className="text-gray-500 dark:text-gray-400">
                            ({formatPercentage([...chartData].sort((a, b) => a.value - b.value)[0].percentage)})
                          </span>
                        </div>
                      )}
                    </div>
                    
                    {/* Distribution Stats */}
                    <div className="pt-3 space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-gray-600 dark:text-gray-400">Total Responses:</span>
                        <span className="font-medium text-gray-900 dark:text-gray-100">
                          {chartData.reduce((sum, item) => sum + item.value, 0).toLocaleString()}
                        </span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-gray-600 dark:text-gray-400">Unique Answers:</span>
                        <span className="font-medium text-gray-900 dark:text-gray-100">
                          {chartData.length}
                        </span>
                      </div>
                      
                      {/* Response spread */}
                      <div className="mt-3 pt-2 border-t border-gray-200 dark:border-gray-700">
                        <div className="text-xs text-center mb-1 text-gray-500 dark:text-gray-400">Answer Diversity</div>
                        <div className="flex gap-0.5 h-2">
                          {chartData.length > 0 && [...Array(Math.min(20, chartData.length))].map((_, i) => {
                            const ratio = (chartData[i]?.value || 0) / chartData[0].value;
                            return (
                              <div 
                                key={i}
                                className="h-full rounded-sm flex-1"
                                style={{ 
                                  backgroundColor: getColorWithContrast(0, colorPalette),
                                  opacity: Math.max(0.1, ratio)
                                }}
                              />
                            );
                          })}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </CollapsibleSection>
        )}
      </div>
      <Footer />
    </>
  );
}