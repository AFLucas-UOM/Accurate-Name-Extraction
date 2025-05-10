#!/bin/bash

# Full path to your project
PROJECT_DIR="/Volumes/Filis SSD/FYP/Accurate-Name-Extraction/5. ANEP UI"

# Process ID for tracking frontend
FRONTEND_PID=""

# Function to cleanly shutdown frontend process
cleanup() {
  echo "Shutting down ANEP frontend..."

  if [[ -n "$FRONTEND_PID" ]]; then
    echo "Terminating frontend process..."
    kill -TERM $FRONTEND_PID 2>/dev/null || true
  fi

  # Ensure all npm dev processes are killed
  pkill -f "npm run dev" >/dev/null 2>&1

  echo "Frontend has been shut down."
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

# Check if package.json exists
if [ ! -f "$PROJECT_DIR/package.json" ]; then
  echo "Error: package.json not found"
  exit 1
fi

# Kill existing frontend processes
echo "Terminating any existing frontend processes..."
pkill -f "npm run dev" >/dev/null 2>&1
sleep 1

# Copy the localhost URL to clipboard
echo "http://localhost:8080/" | pbcopy
echo "Frontend URL copied to clipboard: http://localhost:8080/"

# Launch frontend in new Terminal window
osascript <<EOF
tell application "Terminal"
  do script "clear && echo 'ANEP FRONTEND PROCESS' && echo '--------------------' && echo '' && cd \"$PROJECT_DIR\" && npm run dev | tee logs/frontend-\$(date +%F).log; echo 'Frontend process terminated. Press any key to close this window.'; read -n 1"
  set custom title of window 1 to "ANEP Frontend"
end tell
EOF

# Get frontend PID
sleep 2
FRONTEND_PID=$(pgrep -f "npm run dev" | head -n 1)

# Control panel
clear
echo ""
echo "ANEP Frontend Initialized"
echo "--------------------------"
echo "Frontend: Running in 'ANEP Frontend' terminal (PID: $FRONTEND_PID)"
echo "Access the application at: http://localhost:8080/"
echo ""
echo "This is the control terminal. Keep it open to maintain the frontend."
echo "Press Ctrl+C to gracefully terminate the frontend process."

# Wait indefinitely
while true; do
  sleep 1
done
