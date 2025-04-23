import React, { useCallback, useEffect, useMemo, useState } from "react";
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
import NavBar from "@/components/layout/NavBar";
import Footer from "@/components/layout/Footer";

// Types
export type SurveyRow = Record<string, string | number | undefined>;
interface DashboardProps {
  csvUrl?: string;     
  title?: string;      
}

// Accessible palette (WCAG-AA compliant)
const PALETTE = [
  "#2563eb",
  "#10b981",
  "#f59e0b",
  "#ef4444",
  "#8b5cf6",
  "#ec4899",
  "#14b8a6",
  "#f97316",
  "#22c55e",
  "#6366f1",
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

function splitHeaders(headers: string[]) {
  const kws = ["age", "gender", "reside"];

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
    kws.some((k) => new RegExp(`\\b${k}\\b`, "i").test(h))
  );

  const opinions = filteredHeaders.filter((h) => !demographics.includes(h));

  return { demographics, opinions } as const;
}


// Main Component
export default function SurveyDashboard({ csvUrl = "/ANEP.csv", title = "Survey Results Dashboard" }: DashboardProps) {
  // Raw data + header meta
  const [rows, setRows] = useState<SurveyRow[]>([]);
  const [headers, setHeaders] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  // UI state – persisted between sessions
  const [primaryQ, setPrimaryQ] = useLocalState<string>("sd_primaryQ", "");
  const [chartType, setChartType] = useLocalState<"bar" | "pie" | "line">("sd_chartType", "bar");
  const [filters, setFilters] = useLocalState<{ q: string; a: string }[]>("sd_filters", []);
  const [sortOrder, setSortOrder] = useLocalState<"asc" | "desc" | "none">("sd_sortOrder", "none");

  // Fetch & parse CSV
  useEffect(() => {
    setLoading(true);
    Papa.parse(csvUrl, {
      download: true,
      header: true,
      skipEmptyLines: true,
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

// Chart data
const chartData = useMemo(() => {
  if (!primaryQ) return [];

  const counts: Record<string, number> = {};

  filteredRows.forEach((r) => {
    const ans = r[primaryQ];

    if (ans !== undefined && ans !== "") {
      // Split answers if it's a "select all that apply" style response
      const individualAnswers = String(ans)
        .split(/[,;]\s*/) // Adjust this regex if needed
        .map(a => a.trim())
        .filter(Boolean);

      individualAnswers.forEach(answer => {
        counts[answer] = (counts[answer] || 0) + 1;
      });
    }
  });

  const total = Object.values(counts).reduce((s, v) => s + v, 0);

  let result = Object.entries(counts).map(([name, value]) => ({
    name,
    value,
    percentage: (value / total) * 100,
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


  // Export CSV
  const exportCsv = () => {
    if (!chartData.length) return;
    const csv = ["Response,Count,Percentage", ...chartData.map((d) => `${d.name},${d.value},${d.percentage.toFixed(2)}`)].join("\n");
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

  // Guards
  if (loading)
    return (
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
  if (error)
    return (
      <>
        <NavBar />
        <div className="p-8 text-center">
          <div className="inline-flex items-center justify-center p-4 bg-red-50 rounded-full mb-4">
            <X className="w-8 h-8 text-red-600" />
          </div>
          <p className="text-xl font-semibold mb-2 text-red-600">Error Loading Survey Data</p>
          <p className="text-gray-600 max-w-lg mx-auto mb-6">{error}</p>
          <Button
            onClick={() => window.location.reload()}
            className="bg-red-600 hover:bg-red-700 text-white"
          >
            Try Again
          </Button>
        </div>
        <Footer />
      </>
    );

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

        {/* Controls */}
        <section className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* Primary Question */}
          <div className="lg:col-span-2">
            <label className="block mb-1 text-lg font-medium flex items-center gap-1">
              <Filter className="w-5 h-5 text-blue-600" /> Select Question to Visualize
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
          {/* Chart Type */}
          <div>
            <label className="block mb-1 text-lg font-medium flex items-center gap-1">
              <BarChartIcon className="w-5 h-5 text-blue-600" /> Chart Style
            </label>
            <div className="flex gap-2">
              <button 
                onClick={() => setChartType("bar")}
                className={`flex-1 p-3 rounded-lg border flex justify-center items-center gap-2 ${
                  chartType === "bar" 
                    ? "bg-blue-100 dark:bg-blue-900 border-blue-500 dark:border-blue-700 text-blue-700 dark:text-blue-300 font-medium" 
                    : "bg-white dark:bg-[#1E293B] border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-[#2D3748]"
                }`}
              >
                <BarChartIcon className="w-5 h-5" /> Bar
              </button>
              <button 
                onClick={() => setChartType("pie")}
                className={`flex-1 p-3 rounded-lg border flex justify-center items-center gap-2 ${
                  chartType === "pie" 
                    ? "bg-blue-100 dark:bg-blue-900 border-blue-500 dark:border-blue-700 text-blue-700 dark:text-blue-300 font-medium" 
                    : "bg-white dark:bg-[#1E293B] border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-[#2D3748]"
                }`}
              >
                <PieChartIcon className="w-5 h-5" /> Pie
              </button>
              <button 
                onClick={() => setChartType("line")}
                className={`flex-1 p-3 rounded-lg border flex justify-center items-center gap-2 ${
                  chartType === "line" 
                    ? "bg-blue-100 dark:bg-blue-900 border-blue-500 dark:border-blue-700 text-blue-700 dark:text-blue-300 font-medium" 
                    : "bg-white dark:bg-[#1E293B] border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-[#2D3748]"
                }`}
              >
                <LineChartIcon className="w-5 h-5" /> Line
              </button>
            </div>
          </div>
          {/* Sort Order */}
          <div className="lg:col-span-3">
            <label className="block mb-1 text-lg font-medium flex items-center gap-1">
              <Filter className="w-5 h-5 text-blue-600" /> Sort Order
            </label>
            <div className="flex gap-2 flex-wrap">
            <button 
              onClick={() => setSortOrder("none")}
              className={`flex-1 p-3 rounded-lg border flex justify-center items-center gap-2 transition-all ${
                sortOrder === "none" 
                  ? "bg-blue-100 dark:bg-blue-900 border-blue-500 dark:border-blue-700 text-blue-700 dark:text-blue-300 font-medium shadow-md" 
                  : "bg-white dark:bg-[#1E293B] border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-[#2D3748]"
              }`}
            >
              <Layers className="w-4 h-4" />
              Default Order
            </button>
              <button 
                onClick={() => setSortOrder("asc")}
                className={`flex-1 p-3 rounded-lg border flex justify-center items-center gap-2 transition-all ${
                  sortOrder === "asc" 
                    ? "bg-blue-100 dark:bg-blue-900 border-blue-500 dark:border-blue-700 text-blue-700 dark:text-blue-300 font-medium shadow-md" 
                    : "bg-white dark:bg-[#1E293B] border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-[#2D3748]"
                }`}
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-1">
                  <path d="m3 8 4-4 4 4"/>
                  <path d="M7 4v16"/>
                  <path d="M11 12h4"/>
                  <path d="M11 16h7"/>
                  <path d="M11 20h10"/>
                </svg>
                Ascending (Low to High)
              </button>
              <button 
                onClick={() => setSortOrder("desc")}
                className={`flex-1 p-3 rounded-lg border flex justify-center items-center gap-2 transition-all ${
                  sortOrder === "desc" 
                    ? "bg-blue-100 dark:bg-blue-900 border-blue-500 dark:border-blue-700 text-blue-700 dark:text-blue-300 font-medium shadow-md" 
                    : "bg-white dark:bg-[#1E293B] border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-[#2D3748]"
                }`}
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-1">
                  <path d="M3 16l4 4 4-4"/>
                  <path d="M7 20V4"/>
                  <path d="M11 4h10"/>
                  <path d="M11 8h7"/>
                  <path d="M11 12h4"/>
                </svg>
                Descending (High to Low)
              </button>
            </div>
          </div>
        </section>

        {/* Demographic Filters */}
        {demographics.length > 0 && (
          <section className="bg-gray-50 dark:bg-[#1E293B] p-5 rounded-xl shadow-md border border-gray-200 dark:border-gray-700 space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold flex items-center gap-2">
                <UserCircle2 className="w-5 h-5 text-blue-600" /> Filter by Demographics
              </h2>
              {filters.length > 0 && (
                <Button 
                  variant="outline" 
                  size="sm" 
                  onClick={clearFilters}
                  className="flex items-center gap-1 hover:bg-red-50 text-red-600 hover:text-red-700 border-red-200 hover:border-red-300"
                >
                  <X className="w-4 h-4" /> Clear All Filters
                </Button>
              )}
            </div>
            
            {/* Active Filters Display */}
            {filters.length > 0 && (
              <div className="flex flex-wrap gap-2 mb-3 p-4 bg-blue-50 dark:bg-blue-900/30 rounded-lg shadow-inner border border-blue-100 dark:border-blue-800">
                <span className="text-sm font-medium text-blue-700 dark:text-blue-300 mr-2">Active filters:</span>
                {filters.map((f, i) => (
                  <span key={i} className="flex items-center gap-1 bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 px-3 py-1 rounded-full text-sm shadow-sm border border-blue-200 dark:border-blue-800 transition-all hover:shadow-md">
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
            )}
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {demographics.map((q) => {
                const options = Array.from(new Set(rows.map((r) => String(r[q] ?? "")).filter(Boolean)));
                // Count how many active filters for this question
                const activeFiltersCount = filters.filter(f => f.q === q).length;
                
                return (
                  <div key={q} className="flex flex-col">
                    <label className="text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">
                      {q} {activeFiltersCount > 0 && <span className="text-blue-600">({activeFiltersCount} filter{activeFiltersCount > 1 ? 's' : ''})</span>}
                    </label>
                    <select
                      className="border border-gray-300 dark:border-gray-600 rounded-md p-2 text-sm bg-white dark:bg-[#1E293B] shadow-sm focus:border-blue-500 focus:ring focus:ring-blue-200 focus:ring-opacity-50 dark:text-gray-200"
                      onChange={(e) => e.target.value && addFilter(q, e.target.value)}
                      value=""
                      aria-label={`Filter by ${q}`}
                    >
                      <option value="">All options...</option>
                      {options.map((opt) => {
                        // Check if this option is already in an active filter
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
            
            <div className="flex justify-between items-center pt-2 border-t border-gray-200 dark:border-gray-700">
              <p className="text-sm text-muted-foreground flex items-center gap-1">
                <UserCircle2 className="w-4 h-4" /> 
                <strong>{filteredRows.length}</strong> of <strong>{rows.length}</strong> responses 
                ({((filteredRows.length / rows.length) * 100).toFixed(1)}%)
              </p>
              
              {filters.length > 0 && (
                <p className="text-xs text-blue-600 dark:text-blue-400">
                  Showing filtered results
                </p>
              )}
            </div>
          </section>
        )}

        {/* Action Row */}
        <div className="flex flex-wrap gap-3 justify-between items-center mt-6">
          <p className="text-sm text-gray-600 dark:text-gray-300">
            {filteredRows.length > 0
              ? `Showing results from ${filteredRows.length} respondents`
              : "No data with current filters. Try adjusting your filters."
            }
          </p>

          <div className="flex gap-2 flex-wrap">
            {/* Copy Chart Button */}
            <Button
              onClick={copyChart}
              className="flex items-center gap-2 bg-gray-600 text-white hover:bg-gray-700 py-2 px-4 rounded-lg shadow-sm transition-colors"
              disabled={chartData.length === 0}
            >
              {copied ? <Check className="w-5 h-5" /> : <Copy className="w-5 h-5" />}
              {copied ? "Copied Chart" : "Copy Chart"}
            </Button>

            {/* Export Filtered Data */}
            <Button
              onClick={exportCsv}
              className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded-lg shadow-sm transition-colors"
              disabled={chartData.length === 0}
            >
              <Download className="w-5 h-5" /> Export Data
            </Button>

            {/* Export Full Dataset */}
            <Button
              onClick={() => {
                const a = document.createElement("a");
                a.href = csvUrl;
                a.download = "ANEP.csv";
                a.click();
              }}
              className="flex items-center gap-2 bg-gray-500 hover:bg-gray-600 text-white py-2 px-4 rounded-lg shadow-sm transition-colors"
            >
              <Download className="w-5 h-5" /> Export Full Dataset
            </Button>
          </div>
        </div>

        {/* Main Chart */}
        <section className="bg-white dark:bg-[#1E293B] p-6 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 transition-all hover:shadow-xl">
          {primaryQ ? (
            <>
              <h2 className="text-xl font-semibold mb-6 text-center flex items-center justify-center gap-2 bg-blue-50 dark:bg-blue-900/30 p-3 rounded-lg shadow-sm">
                <div className="bg-blue-100 dark:bg-blue-800 p-2 rounded-full shadow-inner">
                  {chartType === "bar" && <BarChartIcon className="w-6 h-6 text-blue-600 dark:text-blue-400" />}
                  {chartType === "pie" && <PieChartIcon className="w-6 h-6 text-blue-600 dark:text-blue-400" />}
                  {chartType === "line" && <LineChartIcon className="w-6 h-6 text-blue-600 dark:text-blue-400" />}
                </div>
                <span className="text-blue-800 dark:text-blue-300 truncate max-w-lg">{primaryQ}</span>
              </h2>
              
              {chartData.length === 0 ? (
                <div className="h-64 flex items-center justify-center text-muted-foreground">
                  No data available for this question
                </div>
              ) : (
                <div className="h-[500px]">
                  <ResponsiveContainer width="100%" height="100%">
                    {chartType === "bar" && (
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
                        <YAxis allowDecimals={false} />
                        <RechartTooltip 
                          formatter={(value, name) => [
                            `${value} responses (${chartData.find(d => d.value === value)?.percentage.toFixed(1)}%)`, 
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
                        <Bar dataKey="value" name="Count" fill={PALETTE[0]} barSize={60} />
                      </BarChart>
                    )}
                    {chartType === "pie" && (
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
                        >
                          {chartData.map((_, idx) => (
                            <Cell key={`cell-${idx}`} fill={PALETTE[idx % PALETTE.length]} />
                          ))}
                        </Pie>
                        <Legend 
                          layout="horizontal" 
                          verticalAlign="bottom" 
                          align="center"
                          formatter={(value) => {
                            const entry = chartData.find(d => d.name === value);
                            return `${value} (${entry?.value} responses)`;
                          }}
                          wrapperStyle={{ paddingTop: 20 }}
                        />
                        <RechartTooltip 
                          formatter={(value) => `${value} responses`} 
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
                    )}
                    {chartType === "line" && (
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
                        <YAxis allowDecimals={false} />
                        <RechartTooltip 
                          formatter={(value) => [`${value} responses`, "Count"]} 
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
                          stroke={PALETTE[0]} 
                          strokeWidth={3}
                          activeDot={{ r: 8 }} 
                        />
                      </LineChart>
                    )}
                  </ResponsiveContainer>
                </div>
              )}
            </>
          ) : (
            <div className="h-64 flex flex-col items-center justify-center text-muted-foreground space-y-4">
              <Filter className="w-12 h-12 text-blue-300" />
              <p className="text-xl">Please select a question to visualize</p>
              <p className="text-sm max-w-md text-center">
                Choose a question from the dropdown above to see the distribution of responses
              </p>
            </div>
          )}
        </section>


      </div>
      <Footer />
    </>
  );
}