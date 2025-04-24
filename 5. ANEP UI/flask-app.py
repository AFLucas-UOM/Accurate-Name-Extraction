from flask import Flask, jsonify, request
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

start_time = time.time()

FRONTEND_ORIGIN = "http://localhost:8080"
FRONTEND_PORT = FRONTEND_ORIGIN.split(":")[-1]

app = Flask(__name__)
CORS(app, origins=[FRONTEND_ORIGIN])

UPLOAD_FOLDER = os.path.join(os.getcwd(), "5. ANEP UI/Uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

latest_uploaded_filename = None  # Global variable to track last upload

def get_apple_chip():
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            stdout=subprocess.PIPE,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except Exception as e:
        print("[Apple chip detection error]", e)
        return "Apple M-series"

@app.route("/api/ping", methods=["GET"])
def ping():
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
        print("[GPU detection error]", e)
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
        "port": 5050,
        "frontendPort": FRONTEND_PORT,
        "memoryUsedMb": round(memory_info.used / 1024**2, 2),
        "memoryTotalMb": round(memory_info.total / 1024**2, 2),
        "gpuAvailable": gpu_status["available"],
        "gpuName": gpu_status["name"] or "None"
    })

@app.route("/api/upload", methods=["POST"])
def upload_video():
    global latest_uploaded_filename

    if "video" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["video"]

    if file.filename == "" or not file.content_type.startswith("video/"):
        return jsonify({"error": "Invalid file"}), 400

    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)

    latest_uploaded_filename = file.filename
    return jsonify({"message": "File uploaded successfully", "filename": file.filename})

@app.route("/api/latest-upload", methods=["GET"])
def get_latest_upload():
    if latest_uploaded_filename:
        return jsonify({"latest": latest_uploaded_filename})
    else:
        return jsonify({"latest": None, "message": "No uploads yet"})

def run_script(cmd):
    try:
        result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
        return {"success": result.returncode == 0, "stdout": result.stdout, "stderr": result.stderr}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.route("/api/run/anep", methods=["POST"])
def run_anep():
    if not latest_uploaded_filename:
        return jsonify({"error": "No video uploaded yet"}), 400
    cmd = f'python "4. ANEP/ANEP.py" --video "5. ANEP UI/Uploads/{latest_uploaded_filename}"'
    result = run_script(cmd)
    return jsonify(result)

@app.route("/api/run/gcloud", methods=["POST"])
def run_gcloud():
    if not latest_uploaded_filename:
        return jsonify({"error": "No video uploaded yet"}), 400
    cmd = f'python "6. GenAI API/GCloudVision.py" "5. ANEP UI/Uploads/{latest_uploaded_filename}" "6. GenAI API/GoogleResults" "6. GenAI API/config.json"'
    result = run_script(cmd)
    return jsonify(result)

@app.route("/api/run/llama", methods=["POST"])
def run_llama():
    if not latest_uploaded_filename:
        return jsonify({"error": "No video uploaded yet"}), 400
    cmd = f'python "6. GenAI API/LLaMAOpenRouter.py" "5. ANEP UI/Uploads/{latest_uploaded_filename}" "6. GenAI API/LlamaResults" "6. GenAI API/config.json"'
    result = run_script(cmd)
    return jsonify(result)

@app.route("/api/run/ensemble", methods=["POST"])
def run_ensemble():
    if not latest_uploaded_filename:
        return jsonify({"error": "No video uploaded yet"}), 400

    results = {}

    def call_and_store(name, cmd):
        results[name] = run_script(cmd)

    threads = [
        threading.Thread(target=call_and_store, args=("anep", f'python "4. ANEP/ANEP.py" --video "5. ANEP UI/Uploads/{latest_uploaded_filename}"')),
        threading.Thread(target=call_and_store, args=("gcloud", f'python "6. GenAI API/GCloudVision.py" "5. ANEP UI/Uploads/{latest_uploaded_filename}" "6. GenAI API/GoogleResults" "6. GenAI API/config.json"')),
        threading.Thread(target=call_and_store, args=("llama", f'python "6. GenAI API/LLaMAOpenRouter.py" "5. ANEP UI/Uploads/{latest_uploaded_filename}" "6. GenAI API/LlamaResults" "6. GenAI API/config.json"')),
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5050)