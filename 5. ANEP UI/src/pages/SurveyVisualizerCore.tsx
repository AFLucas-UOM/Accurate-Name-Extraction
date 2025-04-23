import React, { useEffect, useState, useMemo } from "react";
import Papa from "papaparse";
import { FileBarChart2, Filter, Loader2, PieChart as PieIcon, BarChart as BarIcon, 
         Info, Download, ChevronDown, Percent, Calculator, Users, LineChart } from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, 
         ResponsiveContainer, PieChart, Pie, Cell, Legend, 
         ScatterChart, Scatter, ZAxis, LineChart as RechartLineChart, Line } from "recharts";
import { Button } from "@/components/ui/button";
import * as math from 'mathjs';
import NavBar from "@/components/layout/NavBar";
import Footer from "@/components/layout/Footer";


// Extend the Window interface to include the 'fs' property exposed by Electron preload script
declare global {
  interface Window {
    fs: {
      readFile: (path: string, options?: { encoding?: string }) => Promise<string>;
      // Add other fs methods used here if any
    };
  }
}

type SurveyResponse = Record<string, any>;

interface FilterOption {
  question: string;
  answer: string;
}

// Custom tooltip component for charts
const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-white dark:bg-gray-800 p-3 border border-gray-200 dark:border-gray-700 rounded shadow-lg">
        <p className="font-medium">{label}</p>
        <p className="text-blue-600 dark:text-blue-400">
          Count: {payload[0].value}
        </p>
        {payload[0].payload.percentage && (
          <p className="text-green-600 dark:text-green-400">
            Percentage: {payload[0].payload.percentage.toFixed(1)}%
          </p>
        )}
      </div>
    );
  }
  return null;
};

const SurveyVisualizerCore = () => {
  const [data, setData] = useState<SurveyResponse[]>([]);
  const [filteredData, setFilteredData] = useState<SurveyResponse[]>([]);
  const [chartData, setChartData] = useState<any[]>([]);
  const [question, setQuestion] = useState<string>("");
  const [questions, setQuestions] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filters, setFilters] = useState<FilterOption[]>([]);
  const [availableFilters, setAvailableFilters] = useState<Record<string, string[]>>({});
  const [chartType, setChartType] = useState<string>("bar");
  const [showStats, setShowStats] = useState<boolean>(false);
  const [comparisonQuestion, setComparisonQuestion] = useState<string>("");
  const [showComparison, setShowComparison] = useState<boolean>(false);
  const [sortBy, setSortBy] = useState<string>("value");
  const [sortOrder, setSortOrder] = useState<string>("desc");

  const COLORS = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899", 
                  "#06B6D4", "#F43F5E", "#84CC16", "#6366F1", "#14B8A6", "#D946EF"];

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch("/ANEP.csv");
        const text = await response.text();
        Papa.parse(text, {
          header: true,
          dynamicTyping: true,
          skipEmptyLines: true,
          complete: (result) => {
            const parsedData = result.data as SurveyResponse[];
            setData(parsedData);
            setFilteredData(parsedData);
            
            const fields = result.meta.fields || [];
            const opinionQs = fields.filter(q =>
              q.includes("?") &&
              !q.toLowerCase().includes("age") &&
              !q.toLowerCase().includes("gender") &&
              !q.toLowerCase().includes("reside") &&
              !q.toLowerCase().includes("employment")
            );
            
            const demographicQs = fields.filter(q =>
              q.toLowerCase().includes("age") ||
              q.toLowerCase().includes("gender") ||
              q.toLowerCase().includes("reside") ||
              q.toLowerCase().includes("employment")
            );
            
            // Build available filters from demographic questions
            const filterOptions: Record<string, string[]> = {};
            demographicQs.forEach(question => {
              const uniqueAnswers = Array.from(new Set(
                parsedData.map(row => row[question]).filter(Boolean)
              )) as string[];
              filterOptions[question] = uniqueAnswers;
            });
            
            setAvailableFilters(filterOptions);
            setQuestions(opinionQs);
            setQuestion(opinionQs[0]);
            setLoading(false);
          },
          error: (err) => setError(err.message),
        });
      } catch (err: any) {
        setError(err.message);
        setLoading(false);
      }
    };
  
    fetchData();
  }, []);  

  // Apply filters to data
  useEffect(() => {
    if (data.length) {
      let result = [...data];
      
      // Apply each filter
      filters.forEach(filter => {
        result = result.filter(row => row[filter.question] === filter.answer);
      });
      
      setFilteredData(result);
    }
  }, [data, filters]);

  // Update chart data based on selected question and filtered data
  useEffect(() => {
    if (filteredData.length && question) {
      const counts: Record<string, number> = {};
      let total = 0;
      
      filteredData.forEach(row => {
        const res = row[question];
        if (res) {
          counts[res] = (counts[res] || 0) + 1;
          total++;
        }
      });
      
      // Convert to array and add percentage
      let result = Object.entries(counts).map(([name, value]) => ({ 
        name, 
        value,
        percentage: (value / total) * 100
      }));
      
      // Sort data based on user preference
      if (sortBy === "value") {
        result.sort((a, b) => sortOrder === "desc" ? b.value - a.value : a.value - b.value);
      } else if (sortBy === "name") {
        result.sort((a, b) => sortOrder === "desc" ? 
          b.name.localeCompare(a.name) : a.name.localeCompare(b.name));
      }
      
      setChartData(result);
    }
  }, [question, filteredData, sortBy, sortOrder]);

  // Prepare comparison data when two questions are selected for comparison
  const comparisonData = useMemo(() => {
    if (!showComparison || !comparisonQuestion || !question || !filteredData.length) return [];
    
    const result: any[] = [];
    
    filteredData.forEach(row => {
      const xValue = row[question];
      const yValue = row[comparisonQuestion];
      
      if (xValue && yValue) {
        // Find existing entry or create new one
        let entry = result.find(item => item.x === xValue && item.y === yValue);
        
        if (entry) {
          entry.count++;
        } else {
          result.push({
            x: xValue,
            y: yValue,
            count: 1
          });
        }
      }
    });
    
    return result;
  }, [filteredData, question, comparisonQuestion, showComparison]);

  // Calculate statistics
  const statistics = useMemo(() => {
    if (!chartData.length) return null;
    
    const values = chartData.map(item => item.value);
    
    return {
      total: values.reduce((sum, val) => sum + val, 0),
      mean: math.mean(values),
      median: math.median(values),
      mode: math.mode(values),
      stdDev: math.std(values),
      min: math.min(values),
      max: math.max(values)
    };
  }, [chartData]);
  
  // Add a filter
  const addFilter = (question: string, answer: string) => {
    setFilters([...filters, { question, answer }]);
  };
  
  // Remove a filter
  const removeFilter = (index: number) => {
    const newFilters = [...filters];
    newFilters.splice(index, 1);
    setFilters(newFilters);
  };

  // Export data as CSV
  const exportCSV = () => {
    if (!chartData.length) return;
    
    const csvContent = "data:text/csv;charset=utf-8," 
      + "Response,Count,Percentage\n"
      + chartData.map(item => 
          `"${item.name}",${item.value},${item.percentage.toFixed(2)}`
        ).join("\n");
    
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `${question.replace(/[^a-z0-9]/gi, '_')}_results.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-[60vh] text-center">
        <Loader2 className="h-8 w-8 animate-spin text-blue-600 mb-4" />
        <p className="text-muted-foreground">Loading survey responses...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 text-red-600 text-center">
        <p className="font-semibold">Error loading data</p>
        <p className="text-sm">{error}</p>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto px-4 py-6">
      <div className="mb-6">
        <h2 className="text-2xl font-bold mb-1 flex items-center gap-2">
          <FileBarChart2 className="w-6 h-6 text-blue-500" />
          Advanced Survey Visualizer
        </h2>
        <p className="text-muted-foreground">
          Interactive survey analysis with filtering, comparison, and statistics.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 mb-6">
        <div className="lg:col-span-2">
          <label className="block mb-1 font-medium flex items-center gap-1">
            <Filter className="w-4 h-4" /> Select Primary Question
          </label>
          <select
            className="w-full border border-gray-300 dark:border-gray-600 rounded p-2 bg-white dark:bg-gray-800"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
          >
            {questions.map((q) => (
              <option key={q} value={q}>{q}</option>
            ))}
          </select>
        </div>
        
        <div>
          <label className="block mb-1 font-medium flex items-center gap-1">
            <BarIcon className="w-4 h-4" /> Chart Type
          </label>
          <select
            className="w-full border border-gray-300 dark:border-gray-600 rounded p-2 bg-white dark:bg-gray-800"
            value={chartType}
            onChange={(e) => setChartType(e.target.value)}
          >
            <option value="bar">Bar Chart</option>
            <option value="pie">Pie Chart</option>
            <option value="line">Line Chart</option>
          </select>
        </div>

        <div>
          <label className="block mb-1 font-medium flex items-center gap-1">
            <Calculator className="w-4 h-4" /> Sort Results
          </label>
          <div className="flex gap-2">
            <select
              className="flex-1 border border-gray-300 dark:border-gray-600 rounded p-2 bg-white dark:bg-gray-800"
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
            >
              <option value="value">Count</option>
              <option value="name">Name</option>
            </select>
            <select
              className="flex-1 border border-gray-300 dark:border-gray-600 rounded p-2 bg-white dark:bg-gray-800"
              value={sortOrder}
              onChange={(e) => setSortOrder(e.target.value)}
            >
              <option value="desc">Descending</option>
              <option value="asc">Ascending</option>
            </select>
          </div>
        </div>
      </div>

      {/* Filters Section */}
      <div className="mb-6 bg-gray-50 dark:bg-gray-900 p-4 rounded border border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between mb-3">
          <h3 className="font-semibold flex items-center gap-1">
            <Filter className="w-4 h-4" /> Demographic Filters
          </h3>
          {filters.length > 0 && (
            <Button 
              variant="outline" 
              size="sm" 
              onClick={() => setFilters([])}
              className="text-xs"
            >
              Clear All
            </Button>
          )}
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 mb-3">
          {Object.entries(availableFilters).map(([filterQuestion, answers]) => (
            <div key={filterQuestion} className="flex flex-col">
              <label className="text-sm font-medium mb-1">{filterQuestion}</label>
              <select
                className="border border-gray-300 dark:border-gray-600 rounded p-1 text-sm bg-white dark:bg-gray-800"
                onChange={(e) => e.target.value && addFilter(filterQuestion, e.target.value)}
                value=""
              >
                <option value="">Select {filterQuestion}</option>
                {answers.map(answer => (
                  <option 
                    key={answer} 
                    value={answer}
                    disabled={filters.some(f => f.question === filterQuestion && f.answer === answer)}
                  >
                    {answer}
                  </option>
                ))}
              </select>
            </div>
          ))}
        </div>
        
        {filters.length > 0 && (
          <div className="flex flex-wrap gap-2 mt-2">
            {filters.map((filter, index) => (
              <div 
                key={index} 
                className="bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 px-3 py-1 rounded-full text-sm flex items-center gap-1"
              >
                <span className="text-xs font-medium">{filter.question}: {filter.answer}</span>
                <button 
                  onClick={() => removeFilter(index)}
                  className="ml-1 text-blue-600 dark:text-blue-300 hover:text-blue-800 dark:hover:text-blue-100"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        )}
        
        <div className="text-sm text-muted-foreground mt-2">
          <Users className="w-3 h-3 inline-block mr-1" />
          {filteredData.length} of {data.length} responses match your filters 
          ({((filteredData.length / data.length) * 100).toFixed(1)}%)
        </div>
      </div>

      {/* Comparison Controls */}
      <div className="mb-4 flex flex-col md:flex-row gap-3 items-center">
        <Button 
          variant={showStats ? "default" : "outline"}
          onClick={() => setShowStats(!showStats)}
          className="flex items-center gap-1"
        >
          <Calculator className="w-4 h-4" />
          {showStats ? "Hide Statistics" : "Show Statistics"}
        </Button>
        
        <Button 
          variant={showComparison ? "default" : "outline"}
          onClick={() => setShowComparison(!showComparison)}
          className="flex items-center gap-1"
        >
          <LineChart className="w-4 h-4" />
          {showComparison ? "Hide Comparison" : "Compare Questions"}
        </Button>
        
        <Button 
          variant="outline"
          onClick={exportCSV}
          className="flex items-center gap-1 ml-auto"
        >
          <Download className="w-4 h-4" />
          Export Results
        </Button>
      </div>
      
      {/* Comparison Question Selector */}
      {showComparison && (
        <div className="mb-4">
          <label className="block mb-1 font-medium flex items-center gap-1">
            <ChevronDown className="w-4 h-4" /> Select Comparison Question
          </label>
          <select
            className="w-full border border-gray-300 dark:border-gray-600 rounded p-2 bg-white dark:bg-gray-800"
            value={comparisonQuestion}
            onChange={(e) => setComparisonQuestion(e.target.value)}
          >
            <option value="">Select a question to compare with...</option>
            {questions.filter(q => q !== question).map((q) => (
              <option key={q} value={q}>{q}</option>
            ))}
          </select>
        </div>
      )}
      
      {/* Statistics Panel */}
      {showStats && statistics && (
        <div className="mb-6 bg-white dark:bg-gray-900 p-4 rounded shadow border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
            <Calculator className="w-5 h-5 text-blue-500" /> Statistical Analysis
          </h3>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-gray-50 dark:bg-gray-800 p-3 rounded border border-gray-200 dark:border-gray-700">
              <div className="text-sm text-muted-foreground">Total Responses</div>
              <div className="text-xl font-bold">{statistics.total}</div>
            </div>
            
            <div className="bg-gray-50 dark:bg-gray-800 p-3 rounded border border-gray-200 dark:border-gray-700">
              <div className="text-sm text-muted-foreground">Mean (Average)</div>
              <div className="text-xl font-bold">{statistics.mean.toFixed(2)}</div>
            </div>
            
            <div className="bg-gray-50 dark:bg-gray-800 p-3 rounded border border-gray-200 dark:border-gray-700">
              <div className="text-sm text-muted-foreground">Median</div>
              <div className="text-xl font-bold">{statistics.median.toFixed(2)}</div>
            </div>
            
            <div className="bg-gray-50 dark:bg-gray-800 p-3 rounded border border-gray-200 dark:border-gray-700">
              <div className="text-sm text-muted-foreground">Standard Deviation</div>
              <div className="text-xl font-bold">{statistics.stdDev.toFixed(2)}</div>
            </div>
          </div>
          
          <div className="mt-3 grid grid-cols-2 md:grid-cols-3 gap-4">
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium">Mode:</span>
              <span>{Array.isArray(statistics.mode) 
                ? statistics.mode.join(", ") 
                : statistics.mode}</span>
            </div>
            
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium">Min:</span>
              <span>{statistics.min}</span>
            </div>
            
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium">Max:</span>
              <span>{statistics.max}</span>
            </div>
          </div>
        </div>
      )}

      {/* Main Chart Section */}
      {!showComparison ? (
        <div className="grid grid-cols-1 gap-6">
          <div className="bg-white dark:bg-gray-900 p-4 rounded shadow border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-2 text-center">
              {chartType === "bar" && <span className="flex items-center justify-center gap-2"><BarIcon className="w-5 h-5" /> Bar Chart View</span>}
              {chartType === "pie" && <span className="flex items-center justify-center gap-2"><PieIcon className="w-5 h-5" /> Pie Chart View</span>}
              {chartType === "line" && <span className="flex items-center justify-center gap-2"><LineChart className="w-5 h-5" /> Line Chart View</span>}
            </h3>
            
            <div className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                {chartType === "bar" && (
                  <BarChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#ccc" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <RechartsTooltip content={<CustomTooltip />} />
                    <Legend />
                    <Bar dataKey="value" name="Count" fill="#3B82F6" />
                  </BarChart>
                )}
                
                {chartType === "pie" && (
                  <PieChart>
                    <Pie
                      data={chartData}
                      cx="50%"
                      cy="50%"
                      labelLine={true}
                      label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                      outerRadius={130}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {chartData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <RechartsTooltip />
                    <Legend />
                  </PieChart>
                )}
                
                {chartType === "line" && (
                  <RechartLineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <RechartsTooltip content={<CustomTooltip />} />
                    <Legend />
                    <Line 
                      type="monotone" 
                      dataKey="value" 
                      name="Count" 
                      stroke="#3B82F6" 
                      activeDot={{ r: 8 }} 
                    />
                  </RechartLineChart>
                )}
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      ) : (
        // Comparison View
        comparisonQuestion && (
          <div className="grid grid-cols-1 gap-6">
            <div className="bg-white dark:bg-gray-900 p-4 rounded shadow border border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-semibold mb-2 text-center flex items-center justify-center gap-2">
                <LineChart className="w-5 h-5" /> Question Comparison
              </h3>
              
              <div className="h-96">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart>
                    <CartesianGrid />
                    <XAxis 
                      dataKey="x" 
                      name={question} 
                      type="category"
                    />
                    <YAxis 
                      dataKey="y" 
                      name={comparisonQuestion} 
                      type="category"
                    />
                    <ZAxis 
                      dataKey="count" 
                      range={[40, 200]} 
                      name="Count" 
                    />
                    <RechartsTooltip 
                      cursor={{ strokeDasharray: '3 3' }}
                      content={({ active, payload }) => {
                        if (active && payload && payload.length) {
                          return (
                            <div className="bg-white dark:bg-gray-800 p-3 border border-gray-200 dark:border-gray-700 rounded shadow-lg">
                              <p className="font-medium">{question}: {payload[0].value}</p>
                              <p className="font-medium">{comparisonQuestion}: {payload[1].value}</p>
                              <p className="text-blue-600 dark:text-blue-400">
                                Count: {payload[2].value}
                              </p>
                            </div>
                          );
                        }
                        return null;
                      }}
                    />
                    <Scatter 
                      name="Responses" 
                      data={comparisonData} 
                      fill="#3B82F6"
                    />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
              
              <div className="mt-4 text-sm text-center text-muted-foreground">
                The size of each bubble represents the number of respondents who selected both answers
              </div>
            </div>
          </div>
        )
      )}

      <div className="mt-6 text-sm text-muted-foreground text-center">
        Showing results for: <span className="font-medium">{question}</span>
        {filters.length > 0 && (
          <> (filtered by {filters.length} demographic filter{filters.length !== 1 ? 's' : ''})</>
        )}
      </div>
    </div>
  );
};

export { SurveyVisualizerCore };