import base64
import json
import time
from pathlib import Path
from typing import Dict

import requests

# ============================================================
# RUNPOD CONFIG
# ============================================================

RUNPOD_API_KEY = "rpa_YPK5YV9AL757P9XJEY233AHKCIKKHYHN6G1IB70B1fqowk"

EMOTION_ENDPOINT_ID = "kw804mmqrwyhzz"

EMOTION_URL = f"https://api.runpod.ai/v2/{EMOTION_ENDPOINT_ID}/runsync"

HEADERS = {
    "Authorization": f"Bearer {RUNPOD_API_KEY}",
    "Content-Type": "application/json",
}

# ============================================================
# HELPERS
# ============================================================

def mat_to_base64(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"MAT file not found: {path}")

    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_emotion_runpod(mat_b64: str) -> Dict:
    payload = {
        "input": {
            "mat_b64": mat_b64,
            "topk": 3,
            "window_start": 0,
            "window_end": 250,
            "mat_channel_names": ["FP1", "FZ", "FP2"],
        }
    }

    resp = requests.post(
        EMOTION_URL,
        headers=HEADERS,
        data=json.dumps(payload),
        timeout=120,
    )

    if not resp.ok:
        raise RuntimeError(
            f"RunPod HTTP {resp.status_code}: {resp.text}"
        )

    data = resp.json()

    if data.get("status") != "COMPLETED":
        raise RuntimeError(f"RunPod failed: {data}")

    return data["output"]


# ============================================================
# PUBLIC API (USED BY app.py)
# ============================================================

def run_emotion_from_mat(mat_path: Path) -> Dict[str, float]:
    """
    Called by app.py.
    Returns:
        {
            "label": "joy",
            "prob": 0.86
        }
    """
    print("🔐 Encoding MAT to base64...")
    mat_b64 = mat_to_base64(mat_path)
    print("Base64 length:", len(mat_b64))

    print("🧠 Running emotion inference...")
    output = call_emotion_runpod(mat_b64)

    topk = output.get("topk", [])
    if not topk:
        raise RuntimeError("RunPod returned no topk emotions")

    top = topk[0]

    return {
        "label": top["label"],
        "prob": float(top["prob"]),
    }


# ============================================================
# CLI TEST (OPTIONAL)
# ============================================================

if __name__ == "__main__":
    # Manual test only
    test_mat = Path("../data/recordings/capture_snapshot.mat")
    emotion = run_emotion_from_mat(test_mat)
    print("Emotion:", emotion)
