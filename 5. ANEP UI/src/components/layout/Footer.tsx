import { useState, useEffect } from "react";
import {
  Clock,
  Server,
  Code,
  ChevronUp,
  ChevronDown,
  Info,
  Cpu,
  MemoryStick,
  Gauge,
  Laptop
} from "lucide-react";

const Footer = () => {
  const [showDebug, setShowDebug] = useState(false);
  const [buildTime, setBuildTime] = useState("");
  const [currentTime, setCurrentTime] = useState(new Date());

  const [status, setStatus] = useState({
    loading: true,
    ok: false,
    message: "",
    pythonVersion: "",
    flaskVersion: "",
    flaskEnv: "",
    uptimeSeconds: 0,
    serverTime: "",
    hostname: "",
    port: null,
    frontendPort: "",
    memoryUsedMb: 0,
    memoryTotalMb: 0,
    gpuAvailable: false,
    gpuName: "None"
  });

  const formatUptime = (seconds: number): string => {
    if (!seconds) return "0s";
    const d = Math.floor(seconds / 86400);
    const h = Math.floor((seconds % 86400) / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    return [
      d > 0 ? `${d}d` : "",
      h > 0 ? `${h}h` : "",
      m > 0 ? `${m}m` : "",
      s > 0 || (d === 0 && h === 0 && m === 0) ? `${s}s` : ""
    ].filter(Boolean).join(" ");
  };

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await fetch("http://localhost:5050/api/ping");
        if (!res.ok) throw new Error("Failed to reach Flask backend");
        const data = await res.json();
        setStatus({ loading: false, ok: data.status === "Ok", ...data });
      } catch (error: any) {
        setStatus(prev => ({ ...prev, loading: false, ok: false, message: error.message }));
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    setBuildTime(new Date().toLocaleString());
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const memoryUsagePercent = status.memoryTotalMb
    ? Math.round((status.memoryUsedMb / status.memoryTotalMb) * 100)
    : 0;

  return (
    <footer className="py-4 px-4 mt-auto border-t border-gray-200 dark:border-gray-700 bg-gradient-to-r from-slate-50 to-slate-100 dark:from-slate-800 dark:to-slate-900 transition-colors duration-300">
      <div className="max-w-6xl mx-auto">
        {/* Desktop Layout (sm and up) */}
        <div className="hidden sm:flex flex-row justify-between items-center mb-3">
          <div className="text-left">
            <h3 className="text-base font-semibold text-slate-800 dark:text-slate-200">
              Andrea Filiberto Lucas – ANEP
            </h3>
            <p className="text-xs text-slate-600 dark:text-slate-400">
              Accurate Name Extraction from News Video Graphics
            </p>
          </div>

          <button
            className="flex items-center gap-1 text-blue-600 hover:text-blue-500 dark:text-blue-400 dark:hover:text-blue-300 text-sm transition-colors"
            onClick={() => setShowDebug(!showDebug)}
            aria-expanded={showDebug}
          >
            <Info size={14} />
            <span>Developer Info</span>
            {showDebug ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>
        </div>

        {/* Mobile Layout (only visible below sm) */}
        <div className="sm:hidden flex flex-col items-center text-center mb-3 gap-2">
          <div>
            <h3 className="text-base font-semibold text-slate-800 dark:text-slate-200">
              Andrea Filiberto Lucas – ANEP
            </h3>
            <p className="text-xs text-slate-600 dark:text-slate-400">
              Accurate Name Extraction from News Video Graphics
            </p>
          </div>

          <div className="text-xs text-slate-500 dark:text-slate-400">
            © {currentTime.getFullYear()} All rights reserved.
          </div>

          <button
            className="flex items-center gap-1 text-blue-600 hover:text-blue-500 dark:text-blue-400 dark:hover:text-blue-300 text-sm transition-colors"
            onClick={() => setShowDebug(!showDebug)}
            aria-expanded={showDebug}
          >
            <Info size={14} />
            <span>Developer Info</span>
            {showDebug ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>
        </div>

        {/* Desktop-only Copyright */}
        <div className="hidden sm:block text-xs text-slate-500 dark:text-slate-400 text-left">
          © {currentTime.getFullYear()} All rights reserved.
        </div>
      
        {showDebug && (
          <div className="mt-4 mx-auto bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg p-4 text-xs text-slate-700 dark:text-slate-300 shadow-lg transition-all duration-300 animate-fadeIn">
            <h4 className="font-medium text-slate-900 dark:text-slate-100 mb-3 flex items-center gap-2 pb-2 border-b border-slate-200 dark:border-slate-700">
              <Code size={16} className="text-blue-500" />
              System Status
            </h4>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div className="bg-slate-50 dark:bg-slate-900 p-3 rounded border border-slate-200 dark:border-slate-700">
                <div className="flex items-center gap-2 mb-2 pb-1 border-b border-slate-200 dark:border-slate-700">
                  <Server size={14} className={status.ok ? "text-emerald-500" : "text-rose-500"} />
                  <span className="font-medium">Backend Server</span>
                  <span className={`ml-auto px-1.5 py-0.5 rounded-full text-xs ${status.ok ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-200" : "bg-rose-100 text-rose-800 dark:bg-rose-900 dark:text-rose-200"}`}>
                    {status.loading ? "Checking..." : status.ok ? "Online" : "Offline"}
                  </span>
                </div>
                <div className="space-y-1.5">
                <div className="flex justify-between"><span>Call-Back Status:</span><span className="font-mono">{status.message === "Failed to fetch" ? "Pong" : status.message || "..."}</span></div>
                  <div className="flex justify-between"><span>Host Name:</span><span className="font-mono">{status.hostname || "Unknown"}</span></div>
                  <div className="flex justify-between"><span>Frontend Port:</span><span className="font-mono">{status.frontendPort || "Unknown"}</span></div>
                  <div className="flex justify-between"><span>Backend Port:</span><span className="font-mono">{status.port || "Unknown"}</span></div>
                  <div className="flex justify-between"><span>Uptime:</span><span className="font-mono">{formatUptime(status.uptimeSeconds) === "0s" ? "-" : formatUptime(status.uptimeSeconds)}</span></div>
                  <div className="flex justify-between"><span>Server Time:</span><span className="font-mono">{status.serverTime || "Unknown"}</span></div>
                </div>
              </div>

              <div className="bg-slate-50 dark:bg-slate-900 p-3 rounded border border-slate-200 dark:border-slate-700">
                <div className="flex items-center gap-2 mb-2 pb-1 border-b border-slate-200 dark:border-slate-700">
                  <Laptop size={14} className="text-indigo-500" />
                  <span className="font-medium">Environment</span>
                </div>
                <div className="space-y-1.5">
                  <div className="flex justify-between"><span>Python:</span><span className="font-mono">{status.pythonVersion || "Unknown"}</span></div>
                  <div className="flex justify-between"><span>Flask:</span><span className="font-mono">{status.flaskVersion || "Unknown"}</span></div>
                  <div className="flex justify-between"><span>Frontend Mode:</span><span className="font-mono">{import.meta.env.MODE.charAt(0).toUpperCase() + import.meta.env.MODE.slice(1)}</span></div>
                  <div className="flex justify-between"><span>Backend Mode:</span><span className="font-mono">{status.flaskEnv || "Unknown"}</span></div>
                  <div className="flex justify-between"><span>Frontend Build Time:</span><span className="font-mono truncate" title={buildTime}>{buildTime}</span></div>
                  <div className="flex justify-between"><span>Current Time:</span><span className="font-mono truncate" title={currentTime.toLocaleString()}>{currentTime.toLocaleString()}</span></div>
                </div>
              </div>

              <div className="bg-slate-50 dark:bg-slate-900 p-3 rounded border border-slate-200 dark:border-slate-700">
                <div className="flex items-center gap-2 mb-2 pb-1 border-b border-slate-200 dark:border-slate-700">
                  <Gauge size={14} className="text-amber-500" />
                  <span className="font-medium">System Resources</span>
                </div>
                <div className="space-y-3">
                  <div>
                    <div className="flex justify-between mb-1">
                      <span>Memory Usage:</span>
                      <span className="font-mono">{memoryUsagePercent}%</span>
                    </div>
                    <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-1.5">
                      <div className={`h-1.5 rounded-full ${memoryUsagePercent > 90 ? 'bg-rose-500' : memoryUsagePercent > 70 ? 'bg-amber-500' : 'bg-emerald-500'}`} style={{ width: `${memoryUsagePercent}%` }} />
                    </div>
                    <div className="flex justify-between text-xs mt-1 text-slate-500 dark:text-slate-400">
                      <span>{status.memoryUsedMb} MB used</span>
                      <span>{status.memoryTotalMb} MB total</span>
                    </div>
                  </div>

                  <div className="flex justify-between items-center">
                    <span>GPU:</span>
                    <span className="font-mono text-xs px-2 py-0.5 rounded bg-slate-100 dark:bg-slate-700 dark:text-slate-200">
                      {status.gpuAvailable ? status.gpuName : "None"}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            <div className="mt-4 text-center text-xs text-slate-500 dark:text-slate-400 pt-2 border-t border-slate-200 dark:border-slate-700">
              This information is intended for development and debugging purposes only.
            </div>
          </div>
        )}
      </div>
    </footer>
  );
};

export default Footer;
