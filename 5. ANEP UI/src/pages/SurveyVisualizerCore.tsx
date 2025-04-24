import React, { useCallback, useEffect, useMemo, useState, Suspense } from "react";
import Papa from "papaparse";
import {
  BarChart,
  PieChart,
  LineChart,
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
  Share,
  Sliders,
  Settings,
  ArrowDownAZ,
  ArrowUpZA,
  LayoutGrid
} from "lucide-react";
import {
  ResponsiveContainer,
  BarChart as RechartBarChart,
  Bar,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip as RechartTooltip,
  Legend,
  PieChart as RechartPieChart,
  Pie,
  Cell,
  LineChart as RechartLineChart,
  Line,
} from "recharts";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import NavBar from "@/components/layout/NavBar";
import Footer from "@/components/layout/Footer";

import {List, SortAsc, SortDesc } from "lucide-react";

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
      : " bg-gray-50 dark:bg-gray-800 hover:bg-blue-50 dark:hover:bg-[#121E3C] text-gray-800 dark:text-gray-200";
  } else {
    containerClasses += " border-gray-200 dark:border-gray-700";
    headerClasses += isOpen
      ? " bg-gray-50 dark:bg-gray-800"
      : " bg-gray-50 dark:bg-gray-800 hover:bg-[#121E3C]";
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

// Control option component - reusable component for control options
const ControlOption = ({ 
  active, 
  onClick, 
  icon, 
  label, 
  description = null 
}: { 
  active: boolean; 
  onClick: () => void; 
  icon: React.ReactNode; 
  label: string;
  description?: React.ReactNode;
}) => (
  <button 
    onClick={onClick}
    className={`relative overflow-hidden p-3 rounded-lg flex items-center justify-center gap-3 transition-all
      ${active 
        ? "bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300 ring-2 ring-blue-400 dark:ring-blue-600 font-medium w-full" 
        : "bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700 w-full"
      }
      ${description ? 'flex-col items-start' : ''}
    `}
  >
    <div className="flex items-center gap-2">
      <div className={`p-1.5 rounded-full ${active ? "bg-white/90 dark:bg-gray-800/70" : "bg-blue-50 dark:bg-blue-900/20"}`}>
        {icon}
      </div>
      <span>{label}</span>
    </div>
    
    {description && (
      <div className="text-xs text-gray-500 dark:text-gray-400 ml-9 -mt-1">
        {description}
      </div>
    )}
    
    {active && (
      <span className="absolute inset-x-0 bottom-0 h-0.5 bg-blue-500 dark:bg-blue-600"></span>
    )}
  </button>
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
  
  // New UI state additions
  const [showPercentages, setShowPercentages] = useState(true);
  const [chartHeight, setChartHeight] = useState(500);
  const [gridLines, setGridLines] = useState(true);

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

        {/* Visualisation Controls using CollapsibleSection */}
        <CollapsibleSection 
          title="Visualisation Controls" 
          icon={<Sliders className="w-5 h-5 text-blue-600" />}
          defaultOpen={true}
          type="filter"
          className="bg-white dark:bg-gray-800 rounded-lg shadow-md border border-gray-200 dark:border-gray-700 mb-6"
        >
          {/* Basic and Advanced Tabs */}
          <div className="flex border-b border-gray-200 dark:border-gray-700 mb-4">
            <button 
              className={`py-2 px-4 font-medium text-sm relative ${
                !viewMode || viewMode === "chart" 
                  ? "text-blue-600 dark:text-blue-400" 
                  : "text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300"
              }`}
              onClick={() => viewMode !== "chart" && setViewMode("chart")}
            >
              Chart View
              {(!viewMode || viewMode === "chart") && (
                <span className="absolute bottom-0 left-0 w-full h-0.5 bg-blue-600 dark:bg-blue-400"></span>
              )}
            </button>
            <button 
              className={`py-2 px-4 font-medium text-sm relative ${
                viewMode === "table" 
                  ? "text-blue-600 dark:text-blue-400" 
                  : "text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300"
              }`}
              onClick={() => viewMode !== "table" && setViewMode("table")}
            >
              Table View
              {viewMode === "table" && (
                <span className="absolute bottom-0 left-0 w-full h-0.5 bg-blue-600 dark:bg-blue-400"></span>
              )}
            </button>
          </div>
          
          {/* Control Groups in Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Chart Type - Only show when Chart view is selected */}
            {viewMode === "chart" && (
              <div className="bg-gray-50 dark:bg-gray-800/50 rounded-lg p-3 border border-gray-200 dark:border-gray-700">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="font-medium text-gray-800 dark:text-gray-200 flex items-center gap-2">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-blue-600">
                      <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                      <line x1="3" y1="9" x2="21" y2="9"></line>
                      <line x1="3" y1="15" x2="21" y2="15"></line>
                      <line x1="9" y1="3" x2="9" y2="21"></line>
                      <line x1="15" y1="3" x2="15" y2="21"></line>
                    </svg>
                    Chart Type
                  </h3>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Info className="w-4 h-4 text-gray-400 hover:text-blue-500 cursor-help" />
                      </TooltipTrigger>
                      <TooltipContent side="top">
                        <p className="text-xs">Each chart type visualizes data differently</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
                
                <div className="grid grid-cols-3 gap-2">
                  <button 
                    onClick={() => setChartType("bar")}
                    className={`flex flex-col items-center justify-center p-2 rounded-md border ${
                      chartType === "bar" 
                        ? "bg-blue-100 dark:bg-blue-900/40 border-blue-300 dark:border-blue-700 text-blue-700 dark:text-blue-300" 
                        : "bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700"
                    }`}
                  >
                    <BarChart className={`w-6 h-6 mb-1 ${chartType === "bar" ? "text-blue-600 dark:text-blue-400" : ""}`} />
                    <span className="text-xs font-medium">Bar</span>
                  </button>
                  
                  <button 
                    onClick={() => setChartType("pie")}
                    className={`flex flex-col items-center justify-center p-2 rounded-md border ${
                      chartType === "pie" 
                        ? "bg-blue-100 dark:bg-blue-900/40 border-blue-300 dark:border-blue-700 text-blue-700 dark:text-blue-300" 
                        : "bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700"
                    }`}
                  >
                    <PieChart className={`w-6 h-6 mb-1 ${chartType === "pie" ? "text-blue-600 dark:text-blue-400" : ""}`} />
                    <span className="text-xs font-medium">Pie</span>
                  </button>
                  
                  <button 
                    onClick={() => setChartType("line")}
                    className={`flex flex-col items-center justify-center p-2 rounded-md border ${
                      chartType === "line" 
                        ? "bg-blue-100 dark:bg-blue-900/40 border-blue-300 dark:border-blue-700 text-blue-700 dark:text-blue-300" 
                        : "bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700"
                    }`}
                  >
                    <LineChart className={`w-6 h-6 mb-1 ${chartType === "line" ? "text-blue-600 dark:text-blue-400" : ""}`} />
                    <span className="text-xs font-medium">Line</span>
                  </button>
                </div>
              </div>
            )}
            
            {/* Sort Order */}
            <div className="bg-gray-50 dark:bg-gray-800/50 rounded-lg p-3 border border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between mb-2">
                <h3 className="font-medium text-gray-800 dark:text-gray-200 flex items-center gap-2">
                  <List className="w-4 h-4 text-blue-600" />
                  Sort Order
                </h3>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Info className="w-4 h-4 text-gray-400 hover:text-blue-500 cursor-help" />
                    </TooltipTrigger>
                    <TooltipContent side="top">
                      <p className="text-xs">Arrange responses by frequency</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>

              <div className="grid grid-cols-3 gap-2">
                <button
                  onClick={() => setSortOrder("none")}
                  className={`flex flex-col items-center justify-center p-2 rounded-md border ${
                    sortOrder === "none"
                      ? "bg-blue-100 dark:bg-blue-900/40 border-blue-300 dark:border-blue-700 text-blue-700 dark:text-blue-300"
                      : "bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700"
                  }`}
                >
                  <List className={`mb-1 h-5 w-5 ${sortOrder === "none" ? "text-blue-600 dark:text-blue-400" : ""}`} />
                  <span className="text-xs font-medium">Original</span>
                </button>

                <button
                  onClick={() => setSortOrder("asc")}
                  className={`flex flex-col items-center justify-center p-2 rounded-md border ${
                    sortOrder === "asc"
                      ? "bg-blue-100 dark:bg-blue-900/40 border-blue-300 dark:border-blue-700 text-blue-700 dark:text-blue-300"
                      : "bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700"
                  }`}
                >
                  <SortAsc className={`mb-1 h-5 w-5 ${sortOrder === "asc" ? "text-blue-600 dark:text-blue-400" : ""}`} />
                  <span className="text-xs font-medium">Low-to-High</span>
                </button>

                <button
                  onClick={() => setSortOrder("desc")}
                  className={`flex flex-col items-center justify-center p-2 rounded-md border ${
                    sortOrder === "desc"
                      ? "bg-blue-100 dark:bg-blue-900/40 border-blue-300 dark:border-blue-700 text-blue-700 dark:text-blue-300"
                      : "bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700"
                  }`}
                >
                  <SortDesc className={`mb-1 h-5 w-5 ${sortOrder === "desc" ? "text-blue-600 dark:text-blue-400" : ""}`} />
                  <span className="text-xs font-medium">High-to-Low</span>
                </button>
              </div>
            </div>
            
            {/* Advanced Controls - Only show for chart view */}
            {viewMode === "chart" && (
              <>
                {/* Chart Height */}
                <div className="bg-gray-50 dark:bg-gray-800/50 rounded-lg p-3 border border-gray-200 dark:border-gray-700">
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="font-medium text-gray-800 dark:text-gray-200 flex items-center gap-2">
                      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-blue-600">
                        <rect width="18" height="18" x="3" y="3" rx="2"></rect>
                        <line x1="3" y1="8" x2="21" y2="8"></line>
                        <line x1="3" y1="16" x2="21" y2="16"></line>
                      </svg>
                      Chart Height
                    </h3>
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Info className="w-4 h-4 text-gray-400 hover:text-blue-500 cursor-help" />
                        </TooltipTrigger>
                        <TooltipContent side="top">
                          <p className="text-xs">Adjust the height of the chart</p>
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  </div>
                  
                  <div className="grid grid-cols-3 gap-2">
                    <button 
                      onClick={() => setChartHeight(400)}
                      className={`flex flex-col items-center justify-center p-2 rounded-md border ${
                        chartHeight === 400 
                          ? "bg-blue-100 dark:bg-blue-900/40 border-blue-300 dark:border-blue-700 text-blue-700 dark:text-blue-300" 
                          : "bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700"
                      }`}
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={`mb-1 ${chartHeight === 400 ? "text-blue-600 dark:text-blue-400" : ""}`}>
                        <rect width="18" height="8" x="3" y="10" rx="2"></rect>
                      </svg>
                      <span className="text-xs font-medium">Compact</span>
                    </button>
                    
                    <button 
                      onClick={() => setChartHeight(500)}
                      className={`flex flex-col items-center justify-center p-2 rounded-md border ${
                        chartHeight === 500 
                          ? "bg-blue-100 dark:bg-blue-900/40 border-blue-300 dark:border-blue-700 text-blue-700 dark:text-blue-300" 
                          : "bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700"
                      }`}
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={`mb-1 ${chartHeight === 500 ? "text-blue-600 dark:text-blue-400" : ""}`}>
                        <rect width="18" height="12" x="3" y="6" rx="2"></rect>
                      </svg>
                      <span className="text-xs font-medium">Standard</span>
                    </button>
                    
                    <button 
                      onClick={() => setChartHeight(600)}
                      className={`flex flex-col items-center justify-center p-2 rounded-md border ${
                        chartHeight === 600 
                          ? "bg-blue-100 dark:bg-blue-900/40 border-blue-300 dark:border-blue-700 text-blue-700 dark:text-blue-300" 
                          : "bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700"
                      }`}
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={`mb-1 ${chartHeight === 600 ? "text-blue-600 dark:text-blue-400" : ""}`}>
                        <rect width="18" height="16" x="3" y="4" rx="2"></rect>
                      </svg>
                      <span className="text-xs font-medium">Large</span>
                    </button>
                  </div>
                </div>
                
                {/* Display Options */}
                <div className="bg-gray-50 dark:bg-gray-800/50 rounded-lg p-3 border border-gray-200 dark:border-gray-700">
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="font-medium text-gray-800 dark:text-gray-200 flex items-center gap-2">
                      <Settings className="w-4 h-4 text-blue-600" />
                      Display Options
                    </h3>
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Info className="w-4 h-4 text-gray-400 hover:text-blue-500 cursor-help" />
                        </TooltipTrigger>
                        <TooltipContent side="top">
                          <p className="text-xs">Configure chart display features</p>
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-3">
                    {/* Percentages Toggle */}
                    <div className="flex items-center justify-between bg-white dark:bg-gray-800 p-2 rounded-md border border-gray-200 dark:border-gray-700">
                      <span className="text-xs font-medium text-gray-700 dark:text-gray-300">Show Percentages</span>
                      <button
                        onClick={() => setShowPercentages(!showPercentages)}
                        className={`relative inline-flex h-5 w-10 items-center rounded-full transition-colors ${
                          showPercentages ? 'bg-blue-600' : 'bg-gray-200 dark:bg-gray-700'
                        }`}
                      >
                        <span
                          className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                            showPercentages ? 'translate-x-5' : 'translate-x-1'
                          }`}
                        />
                      </button>
                    </div>
                    
                    {/* Grid Lines Toggle */}
                    <div className="flex items-center justify-between bg-white dark:bg-gray-800 p-2 rounded-md border border-gray-200 dark:border-gray-700">
                      <span className="text-xs font-medium text-gray-700 dark:text-gray-300">Grid Lines</span>
                      <button
                        onClick={() => setGridLines(!gridLines)}
                        className={`relative inline-flex h-5 w-10 items-center rounded-full transition-colors ${
                          gridLines ? 'bg-blue-600' : 'bg-gray-200 dark:bg-gray-700'
                        }`}
                      >
                        <span
                          className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                            gridLines ? 'translate-x-5' : 'translate-x-1'
                          }`}
                        />
                      </button>
                    </div>
                  </div>
                </div>
              </>
            )}
          </div>
        </CollapsibleSection>

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

        {/* Main Visualisation Area */}
        <section className="bg-white dark:bg-[#1E293B] p-6 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 transition-all hover:shadow-xl">
          {primaryQ ? (
            <>
              <h2 className="text-xl font-semibold mb-6 text-center flex items-center justify-center gap-2 bg-blue-50 dark:bg-blue-900/30 p-3 rounded-lg shadow-sm">
                <div className="bg-blue-100 dark:bg-blue-800 p-2 rounded-full shadow-inner">
                  {viewMode === "chart" ? (
                    <>
                      {chartType === "bar" && <BarChart className="w-6 h-6 text-blue-600 dark:text-blue-400" />}
                      {chartType === "pie" && <PieChart className="w-6 h-6 text-blue-600 dark:text-blue-400" />}
                      {chartType === "line" && <LineChart className="w-6 h-6 text-blue-600 dark:text-blue-400" />}
                    </>
                  ) : (
                    <Layers className="w-6 h-6 text-blue-600 dark:text-blue-400" />
                  )}
                </div>
                <span className="text-blue-800 dark:text-blue-300 break-words max-w-full">{primaryQ}</span>
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
                    <div className="h-[500px]" style={{ height: `${chartHeight}px` }}>
                    {chartType === "bar" && (
                      <ResponsiveContainer width="100%" height="100%">
                        <RechartBarChart 
                          data={chartData} 
                          margin={{ top: 20, right: 30, left: 20, bottom: 90 }}
                          barCategoryGap="20%"
                        >
                          {gridLines && <CartesianGrid strokeDasharray="3 3" stroke="#ccc" />}
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
                              `${value.toLocaleString()} responses ${showPercentages ? `(${chartData.find(d => d.value === value)?.percentage.toFixed(1)}%)` : ''}`, 
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
                            label={showPercentages ? {
                              position: 'top',
                              formatter: (value) => `${(chartData.find(d => d.value === value)?.percentage.toFixed(1) || 0)}%`,
                              fill: '#666',
                              fontSize: 12
                            } : null}
                          />
                        </RechartBarChart>
                      </ResponsiveContainer>
                    )}

                    {chartType === "pie" && (
                      <ResponsiveContainer width="100%" height="100%">
                        <RechartPieChart margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                          <Pie 
                            data={chartData} 
                            dataKey="value" 
                            nameKey="name" 
                            cx="50%" 
                            cy="50%" 
                            outerRadius={180} 
                            labelLine={true}
                            label={showPercentages ? ({ name, percent }) => `${name}: ${(percent * 100).toFixed(1)}%` : 
                              ({ name }) => `${name}`}
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
                              return `${value} (${entry?.value.toLocaleString()} responses${showPercentages ? `, ${entry?.percentage.toFixed(1)}%` : ''})`;
                            }}
                            wrapperStyle={{ paddingTop: 20 }}
                          />
                          <RechartTooltip 
                            formatter={(value) => `${value.toLocaleString()} responses${showPercentages ? ` (${chartData.find(d => d.value === value)?.percentage.toFixed(1) || 0}%)` : ''}`} 
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
                        </RechartPieChart>
                      </ResponsiveContainer>
                    )}

                    {chartType === "line" && (
                      <ResponsiveContainer width="100%" height="100%">
                        <RechartLineChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 90 }}>
                          {gridLines && <CartesianGrid strokeDasharray="3 3" />}
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
                            formatter={(value) => [`${value.toLocaleString()} responses${showPercentages ? ` (${chartData.find(d => d.value === value)?.percentage.toFixed(1) || 0}%)` : ''}`, "Count"]} 
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
                            label={showPercentages ? {
                              position: 'top',
                              formatter: (value) => `${(chartData.find(d => d.value === value)?.percentage.toFixed(1) || 0)}%`,
                              fill: '#666',
                              fontSize: 12
                            } : null}
                          />
                        </RechartLineChart>
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
                  Analysis of "{primaryQ}"
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
                      <BarChart className="w-4 h-4 text-blue-600 dark:text-blue-400" />
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
                              <span className="font-medium text-gray-700 dark:text-gray-300 break-words max-w-[60%]" title={item.name}>
                                {item.name}
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
                                  <span className="text-gray-700 dark:text-gray-300 break-words max-w-[60%]" title={item.name}>
                                    {item.name}
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
                            <span className="text-base break-words max-w-full" title={[...chartData].sort((a, b) => b.value - a.value)[0].name}>
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
                            <span className="text-base break-words max-w-full" title={[...chartData].sort((a, b) => a.value - b.value)[0].name}>
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