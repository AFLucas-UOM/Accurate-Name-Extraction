import { useLocation, Link } from "react-router-dom";
import { useEffect } from "react";
import { Home } from "lucide-react";
import NavBar from "@/components/layout/NavBar";
import Footer from "@/components/layout/Footer";

const NotFound = () => {
  const location = useLocation();

  useEffect(() => {
    console.error("404 Error: User attempted to access non-existent route:", location.pathname);
  }, [location.pathname]);

  return (
    <div className="flex flex-col min-h-screen bg-gradient-to-b from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
      <NavBar />

      <main className="flex-grow flex items-center justify-center p-4">
        <div className="max-w-md w-full bg-white dark:bg-slate-800 rounded-lg shadow-lg overflow-hidden flex flex-col items-center text-center">
          <div className="bg-red-500 dark:bg-red-600 w-full p-6 flex justify-center">
            <div className="text-9xl font-bold text-red-200 opacity-30">404</div>
          </div>

          <div className="p-6 flex flex-col items-center">
            <h1 className="text-2xl font-bold text-slate-800 dark:text-slate-200 mb-2">
              Page Not Found
            </h1>

            <p className="text-slate-600 dark:text-slate-400 mb-6">
              The page{" "}
              <code className="bg-slate-100 dark:bg-slate-700 px-2 py-1 rounded font-mono text-sm">
                {location.pathname}
              </code>{" "}
              doesn’t exist. ANEP probably ignored it as noise.
            </p>
            <Link
              to="/"
              className="flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded-md transition-colors"
            >
              <Home size={18} />
              <span>Go to Home</span>
            </Link>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
};

export default NotFound;
