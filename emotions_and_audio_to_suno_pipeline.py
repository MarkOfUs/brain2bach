# -*- coding: utf-8 -*-
!pip -q install --upgrade openai requests librosa soundfile numpy

import os

os.environ["OPENAI_API_KEY"] = ""
os.environ["SUNO_API_TOKEN"] = ""

print("OPENAI_API_KEY set:", bool(os.getenv("OPENAI_API_KEY")))
print("SUNO_API_TOKEN set:", bool(os.getenv("SUNO_API_TOKEN")))

import os
import re
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import requests
import soundfile as sf
from openai import OpenAI

SUNO_BASE_URL = "https://studio-api.prod.suno.com/api/v2/external/hackathons"
EMOTION_LABELS = [
    "amusement", "inspiration", "joy", "tenderness",
    "anger", "fear", "disgust", "sadness", "neutral"
]

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")
TRANSCRIBE_MODEL = os.getenv("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-transcribe")
POLL_INTERVAL_SECONDS = 6
MAX_POLL_SECONDS = 240

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
SUNO_API_TOKEN = os.getenv("SUNO_API_TOKEN", "").strip()

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing.")
if not SUNO_API_TOKEN:
    raise ValueError("SUNO_API_TOKEN is missing.")

client = OpenAI(api_key=OPENAI_API_KEY)


########## THIS IS WHERE THE AUDIO FILE AND EMOTION VECTOR GOES IN
AUDIO_FILE_PATH = "/content/Happy_Birthday_KLICKAUD.mp3"
EMOTION_VECTOR = [0.18, 0.14, 0.19, 0.10, 0.08, 0.07, 0.04, 0.06, 0.14]
COVER_CLIP_ID = "AUTO_FROM_AUDIO"

RUN_TRANSCRIPTION = True
WAIT_FOR_COMPLETE_SEED = False
WAIT_FOR_COMPLETE_FINAL = False

SYNTHETIC_DURATION_SECONDS = 24
SYNTHETIC_SR = 22050
SYNTHETIC_SEED = 42

UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)

def normalize_emotion_vector(vec: List[float], labels: List[str] = EMOTION_LABELS) -> Dict[str, float]:
    arr = np.array(vec, dtype=float)
    if arr.ndim != 1 or len(arr) != len(labels):
        raise ValueError(f"Emotion vector must have length {len(labels)}.")
    if np.any(arr < 0):
        raise ValueError("Emotion probabilities must be non-negative.")
    s = float(arr.sum())
    if s <= 0:
        raise ValueError("Emotion vector sum must be > 0.")
    arr = arr / s
    return {labels[i]: float(arr[i]) for i in range(len(labels))}

def emotion_rollup(emotion_probs: Dict[str, float]) -> Dict[str, float]:
    pos = sum(emotion_probs[k] for k in ["amusement", "inspiration", "joy", "tenderness"])
    neg = sum(emotion_probs[k] for k in ["anger", "fear", "disgust", "sadness"])
    neu = emotion_probs["neutral"]
    return {"positive": round(pos, 4), "negative": round(neg, 4), "neutral": round(neu, 4)}

def top_emotions(emotion_probs: Dict[str, float], k: int = 4) -> List[Tuple[str, float]]:
    return sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True)[:k]

def trim_to_limit(s: str, limit: int) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) <= limit:
        return s
    return s[:limit].rstrip(" ,;:-")

def sanitize_suno_params(obj: Dict[str, Any]) -> Dict[str, str]:
    required = ["topic", "tags", "negative_tags", "prompt"]
    out: Dict[str, str] = {}
    for k in required:
        v = obj.get(k, "")
        if isinstance(v, list):
            v = ", ".join(str(x).strip() for x in v if str(x).strip())
        elif v is None:
            v = ""
        else:
            v = str(v)
        out[k] = v.strip()

    out["topic"] = trim_to_limit(out["topic"], 500)
    out["tags"] = trim_to_limit(out["tags"], 100)
    out["negative_tags"] = trim_to_limit(out["negative_tags"], 100)
    out["prompt"] = out["prompt"].strip()
    return out

def parse_json_loose(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty model output.")
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            raise ValueError("No JSON object found in model output.")
        return json.loads(m.group(0))

def make_synthetic_reference_audio(out_path: str, sr: int = 22050, duration_sec: int = 24, seed: int = 42) -> str:
    rng = np.random.default_rng(seed)
    n = int(sr * duration_sec)
    t = np.arange(n, dtype=np.float32) / sr

    chord_roots = [220.0, 196.0, 246.94, 174.61]
    segment = n // len(chord_roots)
    y = np.zeros_like(t)

    for i, f0 in enumerate(chord_roots):
        a = i * segment
        b = n if i == len(chord_roots) - 1 else (i + 1) * segment
        tt = t[a:b]
        chord = (
            0.45 * np.sin(2 * np.pi * f0 * tt) +
            0.25 * np.sin(2 * np.pi * (f0 * 1.5) * tt) +
            0.20 * np.sin(2 * np.pi * (f0 * 2.0) * tt)
        )
        am = 0.85 + 0.15 * np.sin(2 * np.pi * 0.6 * (tt - tt[0]))
        y[a:b] += chord * am

    bpm = 120.0
    beat_interval = int(sr * 60.0 / bpm)
    click_len = int(0.012 * sr)
    click = np.hanning(click_len) * np.sin(2 * np.pi * 1800 * np.arange(click_len) / sr)
    for start in range(0, n - click_len, beat_interval):
        y[start:start + click_len] += 0.25 * click

    y += 0.01 * rng.standard_normal(n).astype(np.float32)

    fade = int(0.03 * n)
    if fade > 1:
        env = np.ones(n, dtype=np.float32)
        env[:fade] = np.linspace(0, 1, fade, dtype=np.float32)
        env[-fade:] = np.linspace(1, 0, fade, dtype=np.float32)
        y *= env

    peak = float(np.max(np.abs(y))) if y.size else 1.0
    if peak > 0:
        y = 0.9 * (y / peak)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() != ".wav":
        out = out.with_suffix(".wav")
    sf.write(str(out), y, sr, subtype="PCM_16")
    return str(out)

def ensure_audio_file(path: str) -> Tuple[str, bool]:
    p = Path(path)
    if p.exists() and p.is_file():
        return str(p), False
    synth_target = str(p.with_name("reference_synthetic.wav"))
    synth_path = make_synthetic_reference_audio(
        out_path=synth_target,
        sr=SYNTHETIC_SR,
        duration_sec=SYNTHETIC_DURATION_SECONDS,
        seed=SYNTHETIC_SEED,
    )
    print(f"Input audio not found at '{path}'. Generated synthetic reference audio: {synth_path}")
    return synth_path, True

def get_audio_context(path: str, preview_seconds: int = 30) -> Dict[str, Any]:
    info = sf.info(path)
    filesize_mb = round(os.path.getsize(path) / (1024 ** 2), 3)

    with sf.SoundFile(path) as f:
        frames_to_read = min(f.frames, int(preview_seconds * f.samplerate))
        x = f.read(frames_to_read, dtype="float32", always_2d=False)

    if isinstance(x, np.ndarray) and x.ndim > 1:
        x_mono = np.mean(x, axis=1)
    else:
        x_mono = x if isinstance(x, np.ndarray) else np.array([], dtype=np.float32)

    preview_rms = float(np.sqrt(np.mean(np.square(x_mono, dtype=np.float64)))) if x_mono.size else None

    return {
        "filename": Path(path).name,
        "format": info.format,
        "subtype": info.subtype,
        "samplerate": int(info.samplerate),
        "channels": int(info.channels),
        "duration_seconds": round(float(info.duration), 3),
        "filesize_mb": filesize_mb,
        "preview_rms": round(preview_rms, 6) if preview_rms is not None else None,
    }

def transcribe_audio(path: str, client: OpenAI, model: str = TRANSCRIBE_MODEL) -> str:
    size_mb = os.path.getsize(path) / (1024 ** 2)
    if size_mb > 25:
        print(f"Transcription skipped: {size_mb:.2f} MB > 25 MB limit.")
        return ""
    try:
        with open(path, "rb") as f:
            tx = client.audio.transcriptions.create(
                model=model,
                file=f,
                response_format="text",
            )
        if isinstance(tx, str):
            return tx.strip()
        txt = getattr(tx, "text", "")
        return (txt or "").strip()
    except Exception as e:
        print(f"Transcription failed/skipped: {e}")
        return ""

def generate_suno_params_with_gpt(
    client: OpenAI,
    emotion_probs: Dict[str, float],
    audio_context: Dict[str, Any],
    transcript: str,
    purpose: str = "cover",
    model: str = OPENAI_MODEL,
) -> Tuple[Dict[str, str], str]:
    summary = {
        "purpose": purpose,
        "emotion_distribution": emotion_probs,
        "emotion_rollup": emotion_rollup(emotion_probs),
        "top_emotions": top_emotions(emotion_probs, k=4),
        "audio_context": audio_context,
        "transcript_excerpt": transcript[:1500],
    }

    schema = {
        "type": "object",
        "properties": {
            "topic": {"type": "string", "maxLength": 500},
            "tags": {"type": "string", "maxLength": 100},
            "negative_tags": {"type": "string", "maxLength": 100},
            "prompt": {"type": "string"},
        },
        "required": ["topic", "tags", "negative_tags", "prompt"],
        "additionalProperties": False,
    }

    system_msg = (
        "You are a music prompt engineer for the Suno hackathon API. "
        "Return ONLY valid JSON with EXACT keys: topic, tags, negative_tags, prompt. "
        "Use emotion vector + transcript + audio_context. "
        "If purpose='seed_from_audio', create a general, musically coherent base track suitable as a future cover source. "
        "If purpose='cover_from_seed', create a richer target cover direction. "
        "No artist names. No copyrighted lyric reuse."
    )
    user_msg = "Create Suno params from this input:\n" + json.dumps(summary, ensure_ascii=False, indent=2)

    raw = ""
    try:
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "suno_cover_params",
                    "schema": schema,
                    "strict": True,
                }
            },
        )
        raw = (getattr(resp, "output_text", "") or "").strip()
        parsed = parse_json_loose(raw)
    except Exception:
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg + "\n\nReturn JSON only."},
            ],
        )
        raw = (getattr(resp, "output_text", "") or "").strip()
        parsed = parse_json_loose(raw)

    return sanitize_suno_params(parsed), raw

def extract_cover_clip_id(value: str) -> str:
    value = (value or "").strip()
    m = UUID_RE.search(value)
    if not m:
        raise ValueError("cover_clip_id must contain a valid UUID.")
    return m.group(0)

def suno_request(
    method: str,
    path: str,
    token: str,
    params: Optional[Dict[str, str]] = None,
    json_payload: Optional[Dict[str, Any]] = None,
    timeout: int = 90,
) -> Any:
    url = f"{SUNO_BASE_URL}{path}"
    headers = {"Authorization": f"Bearer {token}"}
    if json_payload is not None:
        headers["Content-Type"] = "application/json"

    r = requests.request(
        method=method.upper(),
        url=url,
        headers=headers,
        params=params,
        json=json_payload,
        timeout=timeout,
    )
    if not r.ok:
        try:
            detail = r.json()
        except Exception:
            detail = r.text
        raise RuntimeError(f"Suno API error {r.status_code} on {method.upper()} {path}: {detail}")

    if not r.text.strip():
        return {}
    try:
        return r.json()
    except Exception:
        return {"raw_text": r.text}

def suno_generate_clip(token: str, suno_params: Dict[str, str], cover_clip_id: Optional[str] = None) -> Dict[str, Any]:
    payload = {
        "topic": suno_params.get("topic", ""),
        "tags": suno_params.get("tags", ""),
        "negative_tags": suno_params.get("negative_tags", ""),
        "prompt": suno_params.get("prompt", ""),
    }
    if cover_clip_id:
        payload["cover_clip_id"] = extract_cover_clip_id(cover_clip_id)

    payload = {k: v for k, v in payload.items() if (k == "cover_clip_id" or (isinstance(v, str) and v.strip()))}
    return suno_request("POST", "/generate", token=token, json_payload=payload)

def suno_get_clip(token: str, clip_id: str) -> Dict[str, Any]:
    data = suno_request("GET", "/clips", token=token, params={"ids": clip_id})
    if isinstance(data, dict) and isinstance(data.get("clips"), list):
        clips = data["clips"]
    elif isinstance(data, list):
        clips = data
    else:
        raise RuntimeError(f"Unexpected /clips response shape: {data}")

    if not clips:
        raise RuntimeError(f"No clip data returned for id={clip_id}")

    for c in clips:
        if isinstance(c, dict) and c.get("id") == clip_id:
            return c
    return clips[0]

def poll_clip(
    token: str,
    clip_id: str,
    poll_interval: int = POLL_INTERVAL_SECONDS,
    max_seconds: int = MAX_POLL_SECONDS,
    wait_for_complete: bool = True,
) -> Dict[str, Any]:
    start = time.time()
    first_stream_url_printed = False

    while True:
        elapsed = int(time.time() - start)
        clip = suno_get_clip(token, clip_id)
        status = str(clip.get("status", "unknown")).lower()
        audio_url = clip.get("audio_url")

        print(f"[{elapsed:>3}s] status={status:<10} audio_url={'yes' if bool(audio_url) else 'no'}")

        if status in {"streaming", "complete"} and audio_url and not first_stream_url_printed:
            print("Streaming/final audio URL:", audio_url)
            first_stream_url_printed = True

        if status == "complete":
            return clip

        if (not wait_for_complete) and status in {"streaming", "complete"} and audio_url:
            return clip

        if status in {"error", "failed"}:
            md = clip.get("metadata") or {}
            raise RuntimeError(
                f"Generation failed: error_type={md.get('error_type')} "
                f"error_message={md.get('error_message')}"
            )

        if elapsed >= max_seconds:
            raise TimeoutError(f"Polling timed out after {max_seconds}s. Last status={status}")

        time.sleep(poll_interval)

def needs_auto_cover_id(value: str) -> bool:
    s = (value or "").strip()
    if not s:
        return True
    if s.upper() in {"AUTO_FROM_AUDIO", "AUTO", "NONE"}:
        return True
    return ("PUT_EXISTING_SUNO_CLIP_UUID_HERE" in s)

print("Config loaded")
print("Using AUDIO_FILE_PATH:", AUDIO_FILE_PATH)

audio_path, used_synthetic_audio = ensure_audio_file(AUDIO_FILE_PATH)
emotion_probs = normalize_emotion_vector(EMOTION_VECTOR)
audio_context = get_audio_context(audio_path)

transcript = ""
if RUN_TRANSCRIPTION and not used_synthetic_audio:
    transcript = transcribe_audio(audio_path, client=client, model=TRANSCRIBE_MODEL)
elif RUN_TRANSCRIPTION and used_synthetic_audio:
    print("Synthetic reference audio detected; skipping transcription.")

print("\nEmotion probs (normalized):")
print(json.dumps(emotion_probs, indent=2))

print("\nAudio path used:", audio_path)
print("Synthetic fallback used:", used_synthetic_audio)

print("\nAudio context:")
print(json.dumps(audio_context, indent=2))

print("\nTranscript preview:")
print((transcript[:500] + "...") if len(transcript) > 500 else transcript)

if needs_auto_cover_id(COVER_CLIP_ID):
    print("\nNo valid COVER_CLIP_ID provided -> creating seed clip ID from input audio context...")
    seed_params, seed_raw = generate_suno_params_with_gpt(
        client=client,
        emotion_probs=emotion_probs,
        audio_context=audio_context,
        transcript=transcript,
        purpose="seed_from_audio",
        model=OPENAI_MODEL,
    )
    print("\nSeed params:")
    print(json.dumps(seed_params, indent=2))

    seed_obj = suno_generate_clip(
        token=SUNO_API_TOKEN,
        suno_params=seed_params,
        cover_clip_id=None,
    )
    print("\nSeed generate response:")
    print(json.dumps(seed_obj, indent=2))

    seed_clip_id = seed_obj.get("id")
    if not seed_clip_id:
        raise RuntimeError("Seed generation did not return clip id.")
    print("\nSeed CLIP_ID:", seed_clip_id)

    _seed_clip = poll_clip(
        token=SUNO_API_TOKEN,
        clip_id=seed_clip_id,
        poll_interval=POLL_INTERVAL_SECONDS,
        max_seconds=MAX_POLL_SECONDS,
        wait_for_complete=WAIT_FOR_COMPLETE_SEED,
    )
    resolved_cover_clip_id = seed_clip_id
else:
    resolved_cover_clip_id = extract_cover_clip_id(COVER_CLIP_ID)

print("\nResolved COVER_CLIP_ID:", resolved_cover_clip_id)

final_params, final_raw = generate_suno_params_with_gpt(
    client=client,
    emotion_probs=emotion_probs,
    audio_context=audio_context,
    transcript=transcript,
    purpose="cover_from_seed",
    model=OPENAI_MODEL,
)

print("\nFinal (cover) params:")
print(json.dumps(final_params, indent=2))

final_gen_obj = suno_generate_clip(
    token=SUNO_API_TOKEN,
    suno_params=final_params,
    cover_clip_id=resolved_cover_clip_id,
)

print("\nFinal generate response:")
print(json.dumps(final_gen_obj, indent=2))

final_clip_id = final_gen_obj.get("id")
if not final_clip_id:
    raise RuntimeError("Final generation did not return clip id.")
print("\nFinal CLIP_ID:", final_clip_id)

final_clip = poll_clip(
    token=SUNO_API_TOKEN,
    clip_id=final_clip_id,
    poll_interval=POLL_INTERVAL_SECONDS,
    max_seconds=MAX_POLL_SECONDS,
    wait_for_complete=WAIT_FOR_COMPLETE_FINAL,
)

print("\nFinal clip object:")
print(json.dumps(final_clip, indent=2))

audio_url = final_clip.get("audio_url")
if audio_url:
    print("\nAudio URL:", audio_url)
    try:
        from IPython.display import Audio, display
        display(Audio(audio_url))
    except Exception as e:
        print(f"(Notebook playback unavailable: {e})")
else:
    print("No audio_url yet; run the poll section again if needed.")
