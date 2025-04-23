#!/bin/bash

# Full path to your project
PROJECT_DIR="/Volumes/Filis SSD/FYP/Accurate-Name-Extraction/5. ANEP UI"

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

# Kill existing session if it exists
tmux kill-session -t anep 2>/dev/null

# Copy the localhost URL to clipboard (macOS)
echo "http://localhost:8080/" | pbcopy
echo "Copied http://localhost:8080/ to clipboard"

# Start a new tmux session
tmux new-session -d -s anep "cd \"$PROJECT_DIR\" && npm run dev | tee logs/frontend-\$(date +%F).log || (echo 'Frontend failed to start'; read)"

# Split pane and start Flask backend in the ANEP Conda environment
tmux split-window -v -t anep "cd \"$PROJECT_DIR\" && /opt/anaconda3/bin/conda run -n ANEP python flask-app.py | tee logs/backend-\$(date +%F).log || (echo 'Backend failed to start'; read)"

# Rename windows for better organization
tmux rename-window -t anep 'ANEP-dev'

# Adjust the layout
tmux select-layout -t anep even-vertical

# Configure Ctrl+Q to kill the tmux session
tmux bind-key -n C-q kill-session

# Display helpful message
tmux display-message "Press Ctrl+Q to kill the entire session"

# Attach to the tmux session
tmux attach -t anep