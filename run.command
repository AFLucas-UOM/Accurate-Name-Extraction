# Full path to your project
PROJECT_DIR="/Volumes/Filis SSD/FYP/Accurate-Name-Extraction/5. ANEP UI"

# Start a new tmux session
tmux new-session -d -s anep "cd \"$PROJECT_DIR\" && npm run dev"

# Split pane and start Flask backend in the ANEP Conda environment
tmux split-window -v -t anep "cd \"$PROJECT_DIR\" && /opt/anaconda3/bin/conda run -n ANEP python flask-app.py"

# Attach to the tmux session so you see both
tmux attach -t anep
