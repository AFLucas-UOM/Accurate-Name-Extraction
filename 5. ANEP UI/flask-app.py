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

start_time = time.time()

FRONTEND_ORIGIN = "http://localhost:8080"
FRONTEND_PORT = FRONTEND_ORIGIN.split(":")[-1]

app = Flask(__name__)
CORS(app, origins=[FRONTEND_ORIGIN])

UPLOAD_FOLDER = os.path.join(os.getcwd(), "5. ANEP UI/Uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
    print("[INFO] Received file upload request")

    if "video" not in request.files:
        print("[ERROR] No file part in the request")
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["video"]
    print(f"[INFO] File received: {file.filename}")

    if file.filename == "":
        print("[ERROR] No selected file")
        return jsonify({"error": "No selected file"}), 400

    if not file.content_type.startswith("video/"):
        print("[ERROR] File is not a video")
        return jsonify({"error": "Only video files are allowed"}), 400

    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    print(f"[INFO] Saving file to: {save_path}")
    file.save(save_path)

    print("[SUCCESS] File uploaded successfully")
    return jsonify({"message": "File uploaded successfully", "filename": file.filename})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5050)