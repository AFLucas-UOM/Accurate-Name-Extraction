#!/bin/bash

# Full path to your project
PROJECT_DIR="/Volumes/Filis SSD/FYP/Accurate-Name-Extraction/5. ANEP UI"

# Process IDs for tracking
FRONTEND_PID=""
BACKEND_PID=""

# Function to cleanly shutdown all processes
cleanup() {
  echo "Shutting down ANEP development environment..."
  
  # Kill frontend and backend processes if they exist
  if [[ -n "$FRONTEND_PID" ]]; then
    echo "Terminating frontend process..."
    kill -TERM $FRONTEND_PID 2>/dev/null || true
  fi
  
  if [[ -n "$BACKEND_PID" ]]; then
    echo "Terminating backend process..."
    kill -TERM $BACKEND_PID 2>/dev/null || true
  fi
  
  # Additional cleanup to ensure all processes are terminated
  pkill -f "npm run dev" >/dev/null 2>&1
  pkill -f "python flask-app.py" >/dev/null 2>&1
  
  echo "ANEP development environment has been shut down."
  exit 0
}

# Set up trap for clean exit
trap cleanup EXIT INT TERM

# Create logs directory if it doesn't exist
mkdir -p "$PROJECT_DIR/logs"

# Check if directory exists
if [ ! -d "$PROJECT_DIR" ]; then
  echo "Error: Project directory not found"
  exit 1
fi

# Check if conda environment exists
if ! /opt/anaconda3/bin/conda info --envs | grep -q "ANEP"; then
  echo "Error: ANEP conda environment not found"
  exit 1
fi

# Check if required files exist
if [ ! -f "$PROJECT_DIR/flask-app.py" ]; then
  echo "Error: flask-app.py not found"
  exit 1
fi

if [ ! -f "$PROJECT_DIR/package.json" ]; then
  echo "Error: package.json not found"  
  exit 1
fi

# Kill existing processes thoroughly
echo "Terminating any existing processes..."
pkill -f "npm run dev" >/dev/null 2>&1
pkill -f "python flask-app.py" >/dev/null 2>&1
sleep 1 # Give processes time to terminate

# Copy the localhost URL to clipboard (macOS)
echo "http://localhost:8080/" | pbcopy
echo "Development server URL copied to clipboard: http://localhost:8080/"

# Launch backend in a new terminal window and get PID
osascript <<EOF
tell application "Terminal"
  do script "clear && echo 'ANEP BACKEND PROCESS' && echo '-------------------' && echo '' && cd \"$PROJECT_DIR\" && /opt/anaconda3/bin/conda run -n ANEP python flask-app.py | tee logs/backend-\$(date +%F).log; echo 'Backend process terminated. Press any key to close this window.'; read -n 1"
  set custom title of window 1 to "ANEP Backend"
end tell
EOF

# Get backend PID
sleep 2
BACKEND_PID=$(pgrep -f "python flask-app.py" | head -n 1)

# Launch frontend in a new terminal window and get PID
osascript <<EOF
tell application "Terminal"
  do script "clear && echo 'ANEP FRONTEND PROCESS' && echo '--------------------' && echo '' && cd \"$PROJECT_DIR\" && npm run dev | tee logs/frontend-\$(date +%F).log; echo 'Frontend process terminated. Press any key to close this window.'; read -n 1"
  set custom title of window 1 to "ANEP Frontend"
end tell
EOF

# Get frontend PID
sleep 2
FRONTEND_PID=$(pgrep -f "npm run dev" | head -n 1)

# Clear the screen and show control panel
clear
echo ""
echo "ANEP Development Environment Initialized"
echo "----------------------------------------"
echo "Backend: Running in 'ANEP Backend' terminal (PID: $BACKEND_PID)"
echo "Frontend: Running in 'ANEP Frontend' terminal (PID: $FRONTEND_PID)"
echo "Access the application at: http://localhost:8080/"
echo ""
echo "This is the control terminal. Keep it open to maintain the development environment."
echo ""
echo "Press Ctrl+C to gracefully terminate all processes."
echo "If you use Cmd+Q, you will be prompted to confirm before quitting."

# Register a Cmd+Q handler with AppleScript
osascript <<EOF > /dev/null 2>&1 &
tell application "System Events"
  tell process "Terminal"
    set frontmost to true
    tell menu bar 1
      tell menu bar item "Terminal"
        tell menu "Terminal"
          tell menu item "Quit Terminal"
            set enabled to true
          end tell
        end tell
      end tell
    end tell
  end tell
end tell

on quit of application "Terminal"
  display dialog "Are you sure you want to quit Terminal? This will terminate the ANEP development environment." buttons {"Cancel", "Quit"} default button "Cancel"
  if button returned of result is "Quit" then
    do shell script "pkill -f 'npm run dev'; pkill -f 'python flask-app.py'"
    return true
  else
    return false
  end if
end quit
EOF

# Keep script running in foreground
echo ""
echo "Environment is running. Waiting for termination signal..."
# Wait indefinitely, the trap will handle cleanup
while true; do
  sleep 1
done