from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import time
import platform
import socket
import psutil
import torch
import importlib.metadata
import subprocess
import os
import threading
import logging
import json
import uuid
import queue
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional

# Start timing and configure constants
start_time = time.time()
FRONTEND_ORIGIN = "http://localhost:8080"
FRONTEND_PORT = FRONTEND_ORIGIN.split(":")[-1]
API_PORT = 5050

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('anep_api')

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=[FRONTEND_ORIGIN])

# Configure upload directory
UPLOAD_FOLDER = Path(os.getcwd()) / "5. ANEP UI/Uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# Global state
latest_uploaded_filename = None

# Process tracking and event streaming
active_processes = {}  # Store process objects by ID
process_logs = {}      # Store logs by process ID
clients = {}           # Store client queues by client ID
ensemble_map = {}  # ensemble_id -> list of subprocess_ids

@dataclass
class LogEntry:
    """Represents a single log entry"""
    message: str
    timestamp: str
    type: str = "info"  # info, warning, success, error
    process_id: Optional[str] = None


def get_apple_chip():
    """Detect Apple Silicon chip model"""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            stdout=subprocess.PIPE,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except Exception as e:
        logger.error(f"Apple chip detection error: {e}")
        return "Apple M-series"


def add_log_entry(process_id: str, message: str, log_type: str = "info"):
    """Add a log entry and notify all connected clients"""
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    entry = LogEntry(
        message=message,
        timestamp=timestamp,
        type=log_type,
        process_id=process_id
    )
    
    # Store in process logs
    if process_id not in process_logs:
        process_logs[process_id] = []
    process_logs[process_id].append(entry)
    
    # Notify all connected clients
    for client_queue in clients.values():
        client_queue.put(entry)

def stream_events():
    """Generate SSE stream for client"""
    client_id = str(uuid.uuid4())
    client_queue = queue.Queue()
    clients[client_id] = client_queue
    
    try:
        # Initial connection message
        yield f'data: {json.dumps({"message": "Connected to event stream", "type": "info"})}\n\n'
        
        while True:
            try:
                # Wait for new events with a timeout
                log_entry = client_queue.get(timeout=30)
                event_data = asdict(log_entry)
                yield f'data: {json.dumps(event_data)}\n\n'
                
            except queue.Empty:
                # Send a keep-alive comment to prevent connection timeout
                yield ': keep-alive\n\n'
    finally:
        # Clean up when client disconnects
        if client_id in clients:
            del clients[client_id]

@app.route("/api/events", methods=["GET"])
def events():
    """SSE endpoint to stream logs to clients"""
    return Response(
        stream_events(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable Nginx buffering
        }
    )

@app.route("/api/process/<process_id>/logs", methods=["GET"])
def get_process_logs(process_id):
    """Get all logs for a specific process"""
    if process_id in process_logs:
        return jsonify({
            "logs": [asdict(log) for log in process_logs[process_id]]
        })
    return jsonify({"logs": []})

@app.route("/api/process/<process_id>/status", methods=["GET"])
def get_process_status(process_id):
    """Check if a process is still running"""
    is_active = process_id in active_processes
    return jsonify({
        "active": is_active,
        "log_count": len(process_logs.get(process_id, []))
    })

@app.route("/api/ping", methods=["GET"])
def ping():
    """Health check endpoint with system information"""
    uptime_seconds = int(time.time() - start_time)
    memory_info = psutil.virtual_memory()

    gpu_status = {
        "available": False,
        "name": None
    }

    try:
        if torch.cuda.is_available():
            gpu_status = {
                "available": True,
                "name": torch.cuda.get_device_name(0)
            }
            status_message = "Ping"
        elif torch.backends.mps.is_available():
            gpu_status = {
                "available": True,
                "name": get_apple_chip()
            }
            status_message = "Ping"
        else:
            status_message = "Pong"
    except Exception as e:
        logger.error(f"GPU detection error: {e}")
        status_message = "Pong"

    return jsonify({
        "message": status_message,
        "status": "Ok" if status_message == "Ping" else "Error",
        "pythonVersion": platform.python_version(),
        "flaskVersion": importlib.metadata.version("flask"),
        "flaskEnv": "Debug" if app.debug else "Production",
        "uptimeSeconds": uptime_seconds,
        "serverTime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "hostname": socket.gethostname(),
        "port": API_PORT,
        "frontendPort": FRONTEND_PORT,
        "memoryUsedMb": round(memory_info.used / 1024**2, 2),
        "memoryTotalMb": round(memory_info.total / 1024**2, 2),
        "gpuAvailable": gpu_status["available"],
        "gpuName": gpu_status["name"] or "None"
    })

@app.route("/api/upload", methods=["POST"])
def upload_video():
    """Handle video file uploads"""
    global latest_uploaded_filename

    if "video" not in request.files:
        logger.warning("Upload attempt with no file part")
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["video"]

    if file.filename == "" or not file.content_type.startswith("video/"):
        logger.warning(f"Invalid upload attempt: {file.filename} ({file.content_type})")
        return jsonify({"error": "Invalid file"}), 400

    save_path = UPLOAD_FOLDER / file.filename
    file.save(save_path)

    latest_uploaded_filename = file.filename
    logger.info(f"Successfully uploaded: {file.filename}")
    return jsonify({"message": "File uploaded successfully", "filename": file.filename})

@app.route("/api/latest-upload", methods=["GET"])
def get_latest_upload():
    """Return information about the latest uploaded file"""
    if latest_uploaded_filename:
        return jsonify({"latest": latest_uploaded_filename})
    else:
        return jsonify({"latest": None, "message": "No uploads yet"})

@app.route("/api/process/<process_id>/cancel", methods=["POST"])
def cancel_process(process_id):
    """Cancel and forcefully terminate the running process"""
    process = active_processes.get(process_id)
    if not process:
        add_log_entry(process_id, f"No active process found with ID {process_id}", "error")
        return jsonify({"error": "Process not found"}), 404

    try:
        import psutil
        proc = psutil.Process(process.pid)
        for child in proc.children(recursive=True):
            child.kill()
        proc.kill()

        add_log_entry(process_id, "Process was successfully terminated by user", "warning")

        # Remove from active tracking and logs
        del active_processes[process_id]
        process_logs.pop(process_id, None)

        return jsonify({"message": "Process terminated"}), 200

    except Exception as e:
        error_msg = f"Failed to terminate process: {e}"
        logger.error(error_msg)
        add_log_entry(process_id, error_msg, "error")
        return jsonify({"error": error_msg}), 500

@app.route("/api/process/ensemble/<ensemble_id>/cancel", methods=["POST"])
def cancel_ensemble(ensemble_id):
    """Cancel all subprocesses of an ensemble"""
    subprocess_ids = ensemble_map.get(ensemble_id)
    if not subprocess_ids:
        return jsonify({"error": "No subprocesses found for this ensemble"}), 404

    for pid in subprocess_ids:
        process = active_processes.get(pid)
        if process:
            try:
                proc = psutil.Process(process.pid)
                for child in proc.children(recursive=True):
                    child.kill()
                proc.kill()
                del active_processes[pid]
                process_logs.pop(pid, None)
                add_log_entry(ensemble_id, f"Terminated subprocess {pid}", "warning")
            except Exception as e:
                add_log_entry(ensemble_id, f"Failed to kill subprocess {pid}: {e}", "error")

    add_log_entry(ensemble_id, "All subprocesses terminated", "warning")
    return jsonify({"message": "Ensemble processes terminated"}), 200

def run_script(cmd, process_id=None):
    """Execute a command and stream output live to Flask CLI and connected clients"""
    if process_id is None:
        process_id = str(uuid.uuid4())

    process = subprocess.Popen(
        cmd,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    active_processes[process_id] = process

    try:
        logger.info(f"Executing command: {cmd}")
        add_log_entry(process_id, f"Starting command: {cmd}")

        output_lines = []
        for line in iter(process.stdout.readline, ''):
            if line:
                line = line.strip()
                output_lines.append(line + "\n")
                print(line)  # Live print to Flask console
                
                # Stream to connected clients
                add_log_entry(process_id, line)

        process.stdout.close()
        returncode = process.wait()
        
        if returncode == 0:
            add_log_entry(process_id, "Process completed successfully", "success")
        else:
            add_log_entry(process_id, f"Process failed with exit code {returncode}", "error")
        
        # Cleanup
        if process_id in active_processes:
            del active_processes[process_id]

        return {
            "success": returncode == 0,
            "stdout": ''.join(output_lines),
            "stderr": "",
            "returncode": returncode,
            "process_id": process_id
        }
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Script execution error: {error_msg}")
        add_log_entry(process_id, f"Error: {error_msg}", "error")
        
        # Cleanup
        if process_id in active_processes:
            del active_processes[process_id]
            
        return {
            "success": False,
            "stdout": "",
            "stderr": error_msg,
            "returncode": -1,
            "process_id": process_id
        }

def run_script_async(cmd, process_id=None):
    """Run script in a background thread and return the process ID immediately"""
    if process_id is None:
        process_id = str(uuid.uuid4())
    
    thread = threading.Thread(
        target=run_script,
        args=(cmd, process_id)
    )
    thread.daemon = True
    thread.start()
    
    return process_id

@app.route("/api/run/anep", methods=["POST"])
def run_anep():
    """Run ANEP processing on the latest uploaded video"""
    if not latest_uploaded_filename:
        return jsonify({"error": "No video uploaded yet"}), 400
    
    cmd = f'python "4. ANEP/ANEP.py" --video "5. ANEP UI/Uploads/{latest_uploaded_filename}"'
    process_id = run_script_async(cmd)
    
    return jsonify({
        "success": True,
        "message": "Process started",
        "process_id": process_id
    })


@app.route("/api/run/gcloud", methods=["POST"])
def run_gcloud():
    """Run Google Cloud Vision processing on the latest uploaded video"""
    if not latest_uploaded_filename:
        return jsonify({"error": "No video uploaded yet"}), 400
    
    cmd = f'python "6. GenAI API/GCloudVision.py" "5. ANEP UI/Uploads/{latest_uploaded_filename}" "6. GenAI API/GoogleResults" "6. GenAI API/config.json"'
    process_id = run_script_async(cmd)
    
    return jsonify({
        "success": True,
        "message": "Process started",
        "process_id": process_id
    })


@app.route("/api/run/llama", methods=["POST"])
def run_llama():
    """Run LLaMA processing on the latest uploaded video"""
    if not latest_uploaded_filename:
        return jsonify({"error": "No video uploaded yet"}), 400
    
    cmd = f'python "6. GenAI API/Llama4Maverick.py" "5. ANEP UI/Uploads/{latest_uploaded_filename}" "6. GenAI API/LlamaResults" "6. GenAI API/config.json"'
    process_id = run_script_async(cmd)
    
    return jsonify({
        "success": True,
        "message": "Process started",
        "process_id": process_id
    })

@app.route("/api/run/ensemble", methods=["POST"])
def run_ensemble():
    """Run all processing methods in parallel on the latest uploaded video"""
    if not latest_uploaded_filename:
        return jsonify({"error": "No video uploaded yet"}), 400

    video_path = f'"5. ANEP UI/Uploads/{latest_uploaded_filename}"'
    
    # Create a master process ID for the ensemble
    ensemble_id = str(uuid.uuid4())
    add_log_entry(ensemble_id, "Starting ensemble processing with all models")
    
    # Start each process and collect their IDs
    process_ids = {
        "anep": run_script_async(f'python "4. ANEP/ANEP.py" --video {video_path}'),
        "gcloud": run_script_async(f'python "6. GenAI API/GCloudVision.py" {video_path} "6. GenAI API/GoogleResults" "6. GenAI API/config.json"'),
        "llama": run_script_async(f'python "6. GenAI API/LLaMAOpenRouter.py" {video_path} "6. GenAI API/LlamaResults" "6. GenAI API/config.json"')
    }
    
    ensemble_map[ensemble_id] = list(process_ids.values())
    
    # Store process IDs in the ensemble log
    for model, pid in process_ids.items():
        add_log_entry(ensemble_id, f"Started {model} process with ID: {pid}")
    
    return jsonify({
        "success": True,
        "message": "Ensemble processing started",
        "process_id": ensemble_id,
        "sub_processes": process_ids
    })

if __name__ == "__main__":
    logger.info(f"Starting ANEP API server on port {API_PORT}")
    app.run(debug=True, host="0.0.0.0", port=API_PORT)