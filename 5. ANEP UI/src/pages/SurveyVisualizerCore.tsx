import React, { useEffect, useState } from "react";
import Papa from "papaparse";
import { FileBarChart2, Filter, Loader2, PieChart as PieIcon } from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from "recharts";
import { Button } from "@/components/ui/button";

// Extend the Window interface to include the 'fs' property exposed by Electron preload script
declare global {
  interface Window {
    fs: {
      readFile: (path: string, options: { encoding: string }) => Promise<string>;
      // Add other fs methods used here if any
    };
  }
}

const SurveyVisualizerCore = () => {
  const [data, setData] = useState<any[]>([]);
  const [chartData, setChartData] = useState<any[]>([]);
  const [question, setQuestion] = useState<string>("");
  const [questions, setQuestions] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const COLORS = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899"];

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await window.fs.readFile("Accurate Name Extraction from News Video Graphics Responses  Form responses 1.csv", { encoding: "utf8" });
        Papa.parse(response, {
          header: true,
          dynamicTyping: true,
          skipEmptyLines: true,
          complete: (result) => {
            const parsedData = result.data as any[];
            setData(parsedData);
            const fields = result.meta.fields || [];
            const opinionQs = fields.filter(q => q.includes("?") && !q.includes("age") && !q.includes("gender") && !q.includes("reside") && !q.includes("employment"));
            setQuestions(opinionQs);
            setQuestion(opinionQs[0]);
            setLoading(false);
          },
          error: (err) => setError(err.message)
        });
      } catch (err: any) {
        setError(err.message);
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  useEffect(() => {
    if (data.length && question) {
      const counts: Record<string, number> = {};
      data.forEach(row => {
        const res = row[question];
        if (res) counts[res] = (counts[res] || 0) + 1;
      });
      const result = Object.entries(counts).map(([name, value]) => ({ name, value }));
      setChartData(result);
    }
  }, [question, data]);

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
          Survey Chart Visualizer
        </h2>
        <p className="text-muted-foreground">Filter and view survey question results interactively.</p>
      </div>

      <div className="mb-4">
        <label className="block mb-1 font-medium flex items-center gap-1">
          <Filter className="w-4 h-4" /> Select Question
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

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white dark:bg-gray-900 p-4 rounded shadow border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold mb-2 text-center flex items-center justify-center gap-2">
            <FileBarChart2 className="w-5 h-5" /> Bar Chart View
          </h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#ccc" />
                <XAxis dataKey="name" tick={{ fill: "#8884d8" }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="value" fill="#3B82F6" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-900 p-4 rounded shadow border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold mb-2 text-center flex items-center justify-center gap-2">
            <PieIcon className="w-5 h-5" /> Pie Chart View
          </h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={chartData}
                  cx="50%"
                  cy="50%"
                  label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {chartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="mt-6 text-sm text-muted-foreground text-center">
        Showing results for: <span className="font-medium">{question}</span>
      </div>
    </div>
  );
};

export { SurveyVisualizerCore };
