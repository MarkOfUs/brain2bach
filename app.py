import os
import time
import json
import base64
import threading
from pathlib import Path

# Load .env before reading env vars (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import requests
from flask import Flask, render_template, send_from_directory, request
from flask_socketio import SocketIO

from suno_pipeline import run_suno_from_emotion_probs, EMOTION_LABELS

# ============================================================
# ENVIRONMENT (REQUIRED)
# ============================================================

OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
SUNO_API_TOKEN = (os.getenv("SUNO_API_TOKEN") or "").strip()
RUNPOD_API_KEY = (os.getenv("RUNPOD_API_KEY") or "").strip()
MOCK_EEG = os.getenv("MOCK_EEG", "0").lower() in ("1", "true", "yes")
SERIAL_PORT = os.getenv("SERIAL_PORT", "COM3")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")
if not SUNO_API_TOKEN:
    raise RuntimeError("SUNO_API_TOKEN not set")
if not RUNPOD_API_KEY:
    raise RuntimeError("RUNPOD_API_KEY not set")

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RECORDINGS_DIR = DATA_DIR / "recordings"
SNAPSHOT_PATH = RECORDINGS_DIR / "capture_snapshot.mat"

RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# RUNPOD ENDPOINTS (from README)
# ============================================================

EMOTION_RUNPOD_URL = "https://api.runpod.ai/v2/kw804mmqrwyhzz/runsync"

# ============================================================
# FLASK
# ============================================================

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading"
)

# ============================================================
# EEG RECORDER
# ============================================================

def _create_recorder():
    from serial_recorder import MockEEGRecorder, SerialEEGRecorder

    if MOCK_EEG:
        return MockEEGRecorder(
            fs=50,
            snapshot_path=SNAPSHOT_PATH,
            snapshot_interval_s=2.0,
        )

    try:
        import serial
        with serial.Serial(SERIAL_PORT, 115200, timeout=0.1) as _:
            pass
    except Exception as e:
        print(f"[EEG] Serial port {SERIAL_PORT} not available ({e}). Using mock recorder. Set MOCK_EEG=1 to suppress.")
        return MockEEGRecorder(
            fs=50,
            snapshot_path=SNAPSHOT_PATH,
            snapshot_interval_s=2.0,
        )

    return SerialEEGRecorder(
        port=SERIAL_PORT,
        baud=115200,
        fs=50,
        snapshot_path=SNAPSHOT_PATH,
        snapshot_interval_s=2.0,
    )

recorder = _create_recorder()

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
    return render_template(
        "index.html",
        mock_eeg=MOCK_EEG,
    )

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
# RUNPOD EMOTION (mat → base64 → emotion probs)
# ============================================================

def run_emotion_inference(sid=None):
    """Send mat snapshot to Emotion RunPod, return emotion_probs dict for Suno."""
    global LATEST_EMOTION

    if not SNAPSHOT_PATH.exists():
        raise FileNotFoundError(
            "No EEG recording available. Start recording, wait a few seconds, then stop before creating a song."
        )

    with open(SNAPSHOT_PATH, "rb") as f:
        mat_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "input": {
            "mat_b64": mat_b64,
            "window_start": 0,
            "window_end": 250,
            "topk": 9,
            "mat_channel_names": ["FP1", "FZ", "FP2"]
        }
    }

    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    }
    r = requests.post(EMOTION_RUNPOD_URL, headers=headers, json=payload, timeout=120)

    if r.status_code == 401:
        raise RuntimeError(
            "RunPod API key rejected (401). Check RUNPOD_API_KEY in .env or environment. "
            "Ensure the key is enabled and has Serverless/AI API permissions at runpod.io/settings."
        )
    r.raise_for_status()
    resp = r.json()
    out = resp.get("output", resp)

    classes = out.get("classes", [])
    probs_list = out.get("probs", [])
    topk = out.get("topk", [])

    if not classes or not probs_list or len(classes) != len(probs_list):
        raise RuntimeError(
            f"RunPod returned invalid format. classes={len(classes)} probs={len(probs_list)}"
        )

    probs_raw = dict(zip(classes, [float(p) for p in probs_list]))
    probs_lower = {str(k).lower().strip(): v for k, v in probs_raw.items()}

    emotion_probs = {}
    for k in EMOTION_LABELS:
        val = probs_lower.get(k.lower(), 0)
        if isinstance(val, (int, float)):
            emotion_probs[k] = float(val)
        else:
            emotion_probs[k] = 0.0

    s = sum(emotion_probs.values())
    if s > 0:
        emotion_probs = {k: v / s for k, v in emotion_probs.items()}
    else:
        emotion_probs = {k: 1.0 / len(EMOTION_LABELS) for k in EMOTION_LABELS}

    label = topk[0]["label"] if topk else (classes[0] if classes else "UNKNOWN")

    LATEST_EMOTION = {"label": str(label).upper(), "probs": emotion_probs}

    socketio.emit("emotion_update", LATEST_EMOTION)

    return emotion_probs


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
    recorder.start_recording(RECORDINGS_DIR / "session.mat")

@socketio.on("stop_recording")
def stop_recording():
    recorder.stop_recording()

@socketio.on("create_song")
def create_song():
    sid = request.sid

    def job():
        def emit_status(msg):
            socketio.emit("song_status", msg, to=sid)

        try:
            emit_status({"status": "emotion", "message": "Analyzing brain signals..."})
            emotion_probs = run_emotion_inference()

            emit_status({"status": "suno", "message": "Generating music with Suno..."})

            def on_audio_url(url: str):
                payload = {"audio_url": url, "message": "Song ready! Click to listen."}
                socketio.emit("song_streaming", payload, to=sid)
                socketio.emit("song_status", {"message": payload["message"]}, to=sid)

            result = run_suno_from_emotion_probs(
                emotion_probs=emotion_probs,
                out_dir=RECORDINGS_DIR,
                cover_clip_id="AUTO_FROM_AUDIO",
                wait_for_complete_final=True,
                on_audio_url=on_audio_url,
            )

            audio_url = result.get("audio_url")
            clip_id = result.get("final_clip_id")
            if not audio_url or "audiopipe.suno.ai" not in str(audio_url):
                if clip_id:
                    audio_url = f"https://audiopipe.suno.ai/?item_id={clip_id}"
            socketio.emit(
                "song_ready",
                {
                    "emotion": LATEST_EMOTION["label"],
                    "audio": "/recordings/suno_generated.wav",
                    "audio_url": audio_url,
                    "emotion_probs": LATEST_EMOTION["probs"],
                },
                to=sid,
            )
        except FileNotFoundError as e:
            socketio.emit("song_error", {"message": str(e)}, to=sid)
        except Exception as e:
            socketio.emit("song_error", {"message": f"Generation failed: {e}"}, to=sid)
            import traceback
            traceback.print_exc()

    socketio.start_background_task(job)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    recorder.start()
    socketio.run(app, host="127.0.0.1", port=5000, debug=False)
