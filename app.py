import os
import time
import json
import base64
import threading
from pathlib import Path

import requests
from flask import Flask, render_template, send_from_directory, request
from flask_socketio import SocketIO

from serial_recorder import SerialEEGRecorder
from suno_pipeline import run_suno_from_emotion_probs

# ============================================================
# ENVIRONMENT (REQUIRED)
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUNO_API_TOKEN = os.getenv("SUNO_API_TOKEN")
RUNPOD_API_KEY = "rpa_YPK5YV9AL757P9XJEY233AHKCIKKHYHN6G1IB70B1fqowk"

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")
if not SUNO_API_TOKEN:
    raise RuntimeError("SUNO_API_TOKEN not set")
if not RUNPOD_API_KEY:
    raise RuntimeError("RUNPOD_API_KEY not set")

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RECORDINGS_DIR = DATA_DIR / "recordings"
SNAPSHOT_PATH = RECORDINGS_DIR / "capture_snapshot.mat"

RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# RUNPOD ENDPOINTS
# ============================================================

EMOTION_RUNPOD_URL = "https://api.runpod.ai/v2/kw804mmqrwyhzz/runsync"

# ============================================================
# FLASK
# ============================================================

app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static"
)

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading"
)

# ============================================================
# EEG RECORDER
# ============================================================

recorder = SerialEEGRecorder(
    port="COM3",
    baud=115200,
    fs=50,
    snapshot_path=SNAPSHOT_PATH,
    snapshot_interval_s=2.0
)

# ============================================================
# CLIENT STATE
# ============================================================

ACTIVE_CLIENTS = set()
LATEST_EMOTION = {"label": "UNKNOWN", "probs": {}}

# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def root():
    return render_template("index.html")

@app.route("/recordings/<path:filename>")
def download(filename):
    return send_from_directory(RECORDINGS_DIR, filename, as_attachment=True)

# ============================================================
# EEG STREAM
# ============================================================

def eeg_stream_loop(sid):
    try:
        while sid in ACTIVE_CLIENTS:
            sample = recorder.get_latest()
            socketio.emit(
                "eeg_data",
                {
                    "ch1": [float(sample[0])],
                    "ch2": [float(sample[1])],
                    "ch3": [float(sample[2])],
                },
                to=sid
            )
            time.sleep(0.02)
    finally:
        ACTIVE_CLIENTS.discard(sid)

# ============================================================
# RUNPOD EMOTION
# ============================================================

def run_emotion_inference():
    global LATEST_EMOTION

    with open(SNAPSHOT_PATH, "rb") as f:
        mat_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "input": {
            "mat_b64": mat_b64,
            "window_start": 0,
            "window_end": 250,
            "topk": 3,
            "mat_channel_names": ["FP1", "FZ", "FP2"]
        }
    }

    r = requests.post(
        EMOTION_RUNPOD_URL,
        headers={
            "Authorization": f"Bearer {RUNPOD_API_KEY}",
            "Content-Type": "application/json"
        },
        json=payload,
        timeout=120
    )

    r.raise_for_status()
    out = r.json()["output"]

    label = out["topk"][0]["label"]
    probs = dict(zip(out["classes"], out["probs"]))

    LATEST_EMOTION = {"label": label.upper(), "probs": probs}

    socketio.emit("emotion_update", LATEST_EMOTION)

    return probs

# ============================================================
# SOCKET EVENTS
# ============================================================

@socketio.on("connect")
def on_connect():
    print("Socket connected:", request.sid)

@socketio.on("disconnect")
def on_disconnect():
    ACTIVE_CLIENTS.discard(request.sid)

@socketio.on("request_eeg")
def request_eeg():
    sid = request.sid
    if sid in ACTIVE_CLIENTS:
        return
    ACTIVE_CLIENTS.add(sid)
    socketio.start_background_task(eeg_stream_loop, sid)

@socketio.on("start_recording")
def start_recording():
    recorder.start_recording()

@socketio.on("stop_recording")
def stop_recording():
    recorder.stop_recording()

@socketio.on("create_song")
def create_song():
    def job():
        emotion_probs = run_emotion_inference()

        result = run_suno_from_emotion_probs(
            emotion_probs=emotion_probs,
            out_dir=RECORDINGS_DIR,
            cover_clip_id="AUTO_FROM_AUDIO",
            wait_for_complete_final=True
        )

        socketio.emit(
            "song_ready",
            {
                "emotion": LATEST_EMOTION["label"],
                "audio": "/recordings/suno_generated.wav"
            }
        )

    socketio.start_background_task(job)

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    recorder.start()
    socketio.run(app, host="127.0.0.1", port=5000, debug=False)
