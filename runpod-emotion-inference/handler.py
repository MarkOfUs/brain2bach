\
import base64
import os
from typing import Any, Dict

import runpod
import requests

from emotion_inference import load_model, predict_from_mat_bytes

CKPT_PATH = os.getenv("CKPT_PATH", "/app/models/cnn_faced_best.pt")

# Load once at cold-start (fast subsequent requests)
MODEL, CLASSES = load_model(CKPT_PATH)


def _get_mat_bytes(job_input: Dict[str, Any]) -> bytes:
    """
    Accept either:
      - mat_b64: base64-encoded bytes of a .mat file
      - mat_url: URL to download a .mat file (publicly accessible)
    """
    if "mat_b64" in job_input and job_input["mat_b64"]:
        b64 = job_input["mat_b64"]
        # Allow both "data:..." and raw base64
        if isinstance(b64, str) and b64.startswith("data:"):
            b64 = b64.split(",", 1)[-1]
        return base64.b64decode(b64)

    if "mat_url" in job_input and job_input["mat_url"]:
        url = job_input["mat_url"]
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        return r.content

    raise ValueError("Provide either 'mat_b64' (base64 .mat) or 'mat_url' (downloadable URL).")


def handler(job):
    """
    RunPod handler:
      job["input"] is your JSON payload from /run or /runsync.
    """
    job_input = job.get("input", {}) or {}

    mat_bytes = _get_mat_bytes(job_input)

    topk = int(job_input.get("topk", 3))
    window_start = int(job_input.get("window_start", 0))
    window_end = int(job_input.get("window_end", 250))
    mat_channel_names = job_input.get("mat_channel_names")  # optional, list[str] length 3

    res = predict_from_mat_bytes(
        mat_bytes,
        model=MODEL,
        classes=CLASSES,
        topk=topk,
        window_start=window_start,
        window_end=window_end,
        mat_channel_names=mat_channel_names,
    )

    return {
        "device": "cuda" if __import__("torch").cuda.is_available() else "cpu",
        "fs_hz": res.fs_hz,
        "window": {"start": res.window_start, "end": res.window_end},
        "mat_channels_used": res.mat_channels_used,
        "topk": res.topk,
        # Full distribution if you want it:
        "classes": res.classes,
        "probs": res.probs,
    }


runpod.serverless.start({"handler": handler})
