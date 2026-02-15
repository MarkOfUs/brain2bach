import os
import time
import json
import base64
import threading
from pathlib import Path

import requests
from flask import Flask, render_template, send_from_directory, request
from flask_socketio import SocketIO

from suno_pipeline import run_suno_from_emotion_probs, EMOTION_LABELS

# ============================================================
# ENVIRONMENT (REQUIRED)
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUNO_API_TOKEN = os.getenv("SUNO_API_TOKEN")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
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
MUSIC_RUNPOD_URL = "https://api.runpod.ai/v2/4c1o9eu4bhzzz2/runsync"

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
    out = r.json().get("output", r.json())

    label = out["topk"][0]["label"]
    probs = dict(zip(out["classes"], out["probs"]))

    # Ensure all EMOTION_LABELS present (pad missing with 0)
    emotion_probs = {k: float(probs.get(k, 0)) for k in EMOTION_LABELS}
    s = sum(emotion_probs.values())
    if s > 0:
        emotion_probs = {k: v / s for k, v in emotion_probs.items()}

    LATEST_EMOTION = {"label": label.upper(), "probs": emotion_probs}

    socketio.emit("emotion_update", LATEST_EMOTION)

    return emotion_probs


# ============================================================
# RUNPOD MUSIC (mat → base64 → WAV)
# ============================================================

def run_music_inference():
    """Send mat snapshot to Music RunPod, return WAV bytes."""
    if not SNAPSHOT_PATH.exists():
        raise FileNotFoundError(
            "No EEG recording available. Start recording, wait a few seconds, then stop."
        )

    with open(SNAPSHOT_PATH, "rb") as f:
        mat_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "input": {
            "mat_b64": mat_b64,
            "target_t": 2101,
            "griffin_iters": 32,
            "use_refiner": False,
            "mat_key_data": "data",
            "mat_key_time": "t_sec",
            "channel_cols": [1, 2, 3],
        }
    }

    r = requests.post(
        MUSIC_RUNPOD_URL,
        headers={
            "Authorization": f"Bearer {RUNPOD_API_KEY}",
            "Content-Type": "application/json"
        },
        json=payload,
        timeout=180
    )

    r.raise_for_status()
    out = r.json().get("output", r.json())

    if "error" in out:
        raise RuntimeError(out["error"])

    wav_b64 = out.get("wav_b64")
    if not wav_b64:
        raise RuntimeError("Music RunPod did not return wav_b64")

    wav_path = RECORDINGS_DIR / "music_runpod_generated.wav"
    wav_path.write_bytes(base64.b64decode(wav_b64))

    return str(wav_path)


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
                    "audio": "/recordings/suno_generated.wav",
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


@socketio.on("create_song_direct")
def create_song_direct():
    """Use Music RunPod: EEG mat → WAV (no Suno)."""
    sid = request.sid

    def job():
        try:
            socketio.emit("song_status", {"status": "music", "message": "Converting brain signals to music..."}, to=sid)
            wav_path = run_music_inference()
            socketio.emit(
                "song_ready",
                {"emotion": "DIRECT", "audio": "/recordings/music_runpod_generated.wav"},
                to=sid,
            )
        except FileNotFoundError as e:
            socketio.emit("song_error", {"message": str(e)}, to=sid)
        except Exception as e:
            socketio.emit("song_error", {"message": f"Music generation failed: {e}"}, to=sid)
            import traceback
            traceback.print_exc()

    socketio.start_background_task(job)

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    recorder.start()
    socketio.run(app, host="127.0.0.1", port=5000, debug=False)
