import { Toaster as ShadcnToaster } from "@/components/ui/toaster";
import { Toaster as SonnerToaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { useState, useEffect } from "react";
import { Eye, EyeOff, Lock } from "lucide-react";

import { useTheme } from "./hooks/use-theme";
import SurveyDashboard from "./pages/SurveyVisualizerCore";

const queryClient = new QueryClient();

// Password Protection Component
const PasswordProtection = ({ children }: { children: React.ReactNode }) => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [failedAttempts, setFailedAttempts] = useState(0);
  const [isBlocked, setIsBlocked] = useState(false);
  const [showBackdoor, setShowBackdoor] = useState(false);
  const [konamiSequence, setKonamiSequence] = useState<string[]>([]);

  // Check for existing authentication on mount
  useEffect(() => {
    // Check localStorage first (normal password login)
    const localAuthStatus = localStorage.getItem('survey_auth');
    const localAuthTime = localStorage.getItem('survey_auth_time');
    
    // Check sessionStorage second (backup password login)
    const sessionAuthStatus = sessionStorage.getItem('survey_auth');
    const sessionAuthTime = sessionStorage.getItem('survey_auth_time');
    
    const currentTime = Date.now();
    
    // Session expires after 24 hours
    if (localAuthStatus === 'authenticated' && localAuthTime && (currentTime - parseInt(localAuthTime)) < 24 * 60 * 60 * 1000) {
      setIsAuthenticated(true);
    } else if (sessionAuthStatus === 'authenticated' && sessionAuthTime && (currentTime - parseInt(sessionAuthTime)) < 24 * 60 * 60 * 1000) {
      setIsAuthenticated(true);
    } else {
      // Clear expired sessions
      if (localAuthStatus === 'authenticated' && localAuthTime) {
        localStorage.removeItem('survey_auth');
        localStorage.removeItem('survey_auth_time');
      }
      if (sessionAuthStatus === 'authenticated' && sessionAuthTime) {
        sessionStorage.removeItem('survey_auth');
        sessionStorage.removeItem('survey_auth_time');
      }
    }

    // Also check for failed attempts in sessionStorage to persist across page refreshes
    const storedFailedAttempts = sessionStorage.getItem('survey_failed_attempts');
    const attemptsTime = sessionStorage.getItem('survey_attempts_time');
    
    if (storedFailedAttempts && attemptsTime) {
      const attemptCount = parseInt(storedFailedAttempts);
      const attemptTime = parseInt(attemptsTime);
      
      // Reset failed attempts after 1 hour
      if (currentTime - attemptTime > 60 * 60 * 1000) {
        sessionStorage.removeItem('survey_failed_attempts');
        sessionStorage.removeItem('survey_attempts_time');
        setFailedAttempts(0);
        setIsBlocked(false);
      } else {
        setFailedAttempts(attemptCount);
        if (attemptCount >= 5) {
          setIsBlocked(true);
        }
      }
    }

    // Konami Code sequence: up up down down left right left right B A
    const konamiCode = ['ArrowUp', 'ArrowUp', 'ArrowDown', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'ArrowLeft', 'ArrowRight', 'KeyB', 'KeyA'];
    
    const handleKeyPress = (event: KeyboardEvent) => {
      // Escape key to cancel backdoor mode
      if (event.code === 'Escape' && showBackdoor) {
        setShowBackdoor(false);
        setError("");
        setPassword("");
        return;
      }
      
      setKonamiSequence(currentSequence => {
        const newSequence = [...currentSequence, event.code];
        
        // Check if the current sequence matches the beginning of the Konami code
        const isValid = konamiCode.slice(0, newSequence.length).every((key, index) => key === newSequence[index]);
        
        if (!isValid) {
          // Reset sequence if it doesn't match
          return event.code === konamiCode[0] ? [event.code] : [];
        }
        
        // Check if we've completed the Konami code
        if (newSequence.length === konamiCode.length) {
          setShowBackdoor(true);
          setError("");
          return [];
        }
        
        return newSequence;
      });
    };

    document.addEventListener('keydown', handleKeyPress);
    
    return () => {
      document.removeEventListener('keydown', handleKeyPress);
    };
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (isBlocked && !showBackdoor) return;
    
    setIsLoading(true);
    setError("");

    // Simulate processing time for security
    await new Promise(resolve => setTimeout(resolve, 500 + (failedAttempts * 1000)));

    const correctPassword = "pswd_admin_2608";
    const backupPassword = ",{MbcU2_yrZLl7Vj9d35yg:1(";
    
    if (password === correctPassword || password === backupPassword) {
      const currentTime = Date.now().toString();
      
      if (password === correctPassword) {
        // Normal password - save in localStorage for 24-hour cross-tab persistence
        localStorage.setItem('survey_auth', 'authenticated');
        localStorage.setItem('survey_auth_time', currentTime);
      } else {
        // Backup password - save in sessionStorage for tab-only session
        sessionStorage.setItem('survey_auth', 'authenticated');
        sessionStorage.setItem('survey_auth_time', currentTime);
      }
      
      setIsAuthenticated(true);
      setError("");
      setFailedAttempts(0);
      setIsBlocked(false);
      setShowBackdoor(false);
      
      // Clear failed attempts on successful login
      sessionStorage.removeItem('survey_failed_attempts');
      sessionStorage.removeItem('survey_attempts_time');
      
      // Clear password from memory
      setPassword("");
    } else {
      if (!showBackdoor) {
        const newFailedAttempts = failedAttempts + 1;
        setFailedAttempts(newFailedAttempts);
        setError(`Incorrect password (${newFailedAttempts}/5 attempts)`);
        setPassword("");
        
        // Store failed attempts in sessionStorage
        sessionStorage.setItem('survey_failed_attempts', newFailedAttempts.toString());
        sessionStorage.setItem('survey_attempts_time', Date.now().toString());
        
        // Block after 5 failed attempts
        if (newFailedAttempts >= 5) {
          setIsBlocked(true);
          setError("Too many failed attempts - Please try again in 1 hour");
        }
      } else {
        // Backdoor mode - wrong backup password
        setError("Incorrect backup password");
        setPassword("");
      }
    }
    
    setIsLoading(false);
  };

  if (isAuthenticated) {
    return <>{children}</>;
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-background to-muted/20 p-4">
      <div className="w-full max-w-md">
        <div className="bg-card border border-border rounded-lg shadow-lg p-8 space-y-6">
          <div className="text-center space-y-2">
            <div className="mx-auto w-12 h-12 bg-primary/10 rounded-full flex items-center justify-center mb-4">
              <Lock className="w-6 h-6 text-primary" />
            </div>
            <h1 className="text-2xl font-semibold tracking-tight">Secure Access</h1>
            <p className="text-sm text-muted-foreground">
              Enter password to access dashboard
            </p>
          </div>
          
          <form onSubmit={handleSubmit} className="space-y-4">
                          <div className="space-y-2">
              <label htmlFor="password" className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                {showBackdoor ? "Backup Password" : "Password"}
              </label>
              <div className="relative">
                <input
                  id="password"
                  type={showPassword ? "text" : "password"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder={showBackdoor ? "Enter backup password" : "Enter administrator password"}
                  className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 pr-10 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                  autoFocus
                  disabled={isLoading || (isBlocked && !showBackdoor)}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 transform -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                  disabled={isLoading || (isBlocked && !showBackdoor)}
                >
                  {showPassword ? (
                    <EyeOff className="w-4 h-4" />
                  ) : (
                    <Eye className="w-4 h-4" />
                  )}
                </button>
              </div>
              
              {error && (
                <div className="flex items-center space-x-2 text-sm text-red-600 animate-in slide-in-from-left-1 duration-300">
                  <span className="font-medium">{error}</span>
                </div>
              )}

              {showBackdoor && (
                <div className="mt-2 p-3 bg-muted/50 rounded-md border border-border/50">
                  <div className="flex items-center justify-between">
                    <p className="text-xs text-muted-foreground" style={{ textAlign: "center" }}>
                      Backup code activated - Emergency access mode enabled
                    </p>
                    <button
                      type="button"
                      onClick={() => {
                        setShowBackdoor(false);
                        setError("");
                        setPassword("");
                      }}
                      className="text-xs text-muted-foreground hover:text-foreground ml-2"
                      title="Cancel (or press Escape)"
                    >
                      ✕
                    </button>
                  </div>
                </div>
              )}
            </div>
            
            <button
              type="submit"
              disabled={isLoading || (isBlocked && !showBackdoor) || !password.trim()}
              className="inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 bg-primary text-primary-foreground hover:bg-primary/90 h-10 px-4 py-2 w-full"
            >
              {isLoading ? (
                <div className="flex items-center space-x-2">
                  <div className="w-4 h-4 border-2 border-primary-foreground/30 border-t-primary-foreground rounded-full animate-spin"></div>
                  <span>Verifying...</span>
                </div>
              ) : (
                showBackdoor ? "Access with Backup" : "Access Dashboard"
              )}
            </button>
          </form>
          
          <div className="text-xs text-muted-foreground text-center">
            Session will expire after 24 hours
          </div>
        </div>
      </div>
    </div>
  );
};

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
            <Route path="/" element={<Navigate to="/ANEP-Survey/" replace />} />
            <Route path="/ANEP-Survey/" element={
              <PasswordProtection>
                <SurveyDashboard />
              </PasswordProtection>
            } />
          </Routes>
        </BrowserRouter>
      </TooltipProvider>
    </ThemeInitializer>
  </QueryClientProvider>
);

export default App;