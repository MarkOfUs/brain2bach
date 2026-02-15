import os
import re
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import requests

from openai import OpenAI


SUNO_BASE_URL = "https://studio-api.prod.suno.com/api/v2/external/hackathons"

EMOTION_LABELS = [
    "amusement", "inspiration", "joy", "tenderness",
    "anger", "fear", "disgust", "sadness", "neutral"
]

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")
POLL_INTERVAL_SECONDS = int(os.getenv("SUNO_POLL_INTERVAL_SECONDS", "6"))
MAX_POLL_SECONDS = int(os.getenv("SUNO_MAX_POLL_SECONDS", "240"))

UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)


def _require_env(name: str) -> str:
    v = os.getenv(name, "").strip()
    if not v:
        raise ValueError(f"{name} is missing.")
    return v


def _trim_to_limit(s: str, limit: int) -> str:
    s = re.sub(r"\s+", " ", (s or "")).strip()
    if len(s) <= limit:
        return s
    return s[:limit].rstrip(" ,;:")


def _parse_json_loose(text: str) -> Dict[str, Any]:
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


def _sanitize_suno_params(obj: Dict[str, Any]) -> Dict[str, str]:
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

    out["topic"] = _trim_to_limit(out["topic"], 500)
    out["tags"] = _trim_to_limit(out["tags"], 100)
    out["negative_tags"] = _trim_to_limit(out["negative_tags"], 100)
    out["prompt"] = (out["prompt"] or "").strip()
    return out


def normalize_emotion_vector(vec: List[float], labels: List[str] = EMOTION_LABELS) -> Dict[str, float]:
    arr = np.array(vec, dtype=float)
    if arr.ndim != 1 or len(arr) != len(labels):
        raise ValueError(f"Emotion vector must have length {len(labels)}.")
    if np.any(arr < 0):
        raise ValueError("Emotion probabilities must be non negative.")
    s = float(arr.sum())
    if s <= 0:
        raise ValueError("Emotion vector sum must be greater than zero.")
    arr = arr / s
    return {labels[i]: float(arr[i]) for i in range(len(labels))}


def _emotion_rollup(emotion_probs: Dict[str, float]) -> Dict[str, float]:
    pos = sum(emotion_probs[k] for k in ["amusement", "inspiration", "joy", "tenderness"])
    neg = sum(emotion_probs[k] for k in ["anger", "fear", "disgust", "sadness"])
    neu = emotion_probs["neutral"]
    return {"positive": round(pos, 4), "negative": round(neg, 4), "neutral": round(neu, 4)}


def _top_emotions(emotion_probs: Dict[str, float], k: int = 4) -> List[Tuple[str, float]]:
    return sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True)[:k]


def generate_suno_params_with_gpt(
    client: OpenAI,
    emotion_probs: Dict[str, float],
    audio_context: Dict[str, Any],
    transcript: str,
    purpose: str,
    model: str = OPENAI_MODEL
) -> Tuple[Dict[str, str], str]:
    """
    This matches your notebook approach: GPT creates JSON with topic/tags/negative_tags/prompt.
    """

    summary = {
        "purpose": purpose,
        "emotion_distribution": emotion_probs,
        "emotion_rollup": _emotion_rollup(emotion_probs),
        "top_emotions": _top_emotions(emotion_probs, k=4),
        "audio_context": audio_context,
        "transcript_excerpt": (transcript or "")[:1500],
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
        "Use emotion vector plus transcript plus audio_context. "
        "If purpose is seed_from_audio, create a general musically coherent base track. "
        "If purpose is cover_from_seed, create a richer target cover direction. "
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
        parsed = _parse_json_loose(raw)
    except Exception:
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg + "\n\nReturn JSON only."},
            ],
        )
        raw = (getattr(resp, "output_text", "") or "").strip()
        parsed = _parse_json_loose(raw)

    return _sanitize_suno_params(parsed), raw


def _extract_cover_clip_id(value: str) -> str:
    value = (value or "").strip()
    m = UUID_RE.search(value)
    if not m:
        raise ValueError("cover_clip_id must contain a valid UUID.")
    return m.group(0)


def _needs_auto_cover_id(value: str) -> bool:
    s = (value or "").strip()
    if not s:
        return True
    if s.upper() in {"AUTO_FROM_AUDIO", "AUTO", "NONE"}:
        return True
    return "PUT_EXISTING_SUNO_CLIP_UUID_HERE" in s


def suno_request(
    method: str,
    path: str,
    token: str,
    params: Optional[Dict[str, str]] = None,
    json_payload: Optional[Dict[str, Any]] = None,
    timeout: int = 90
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
    payload: Dict[str, Any] = {
        "topic": suno_params.get("topic", ""),
        "tags": suno_params.get("tags", ""),
        "negative_tags": suno_params.get("negative_tags", ""),
        "prompt": suno_params.get("prompt", ""),
    }
    if cover_clip_id:
        payload["cover_clip_id"] = _extract_cover_clip_id(cover_clip_id)

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
    wait_for_complete: bool = True
) -> Dict[str, Any]:
    start = time.time()
    first_stream_url_printed = False

    while True:
        elapsed = int(time.time() - start)
        clip = suno_get_clip(token, clip_id)
        status = str(clip.get("status", "unknown")).lower()
        audio_url = clip.get("audio_url")

        print(f"[SUNO {elapsed:>3}s] status={status} audio_url={'yes' if bool(audio_url) else 'no'}")

        if status in {"streaming", "complete"} and audio_url and not first_stream_url_printed:
            print("Suno audio URL:", audio_url)
            first_stream_url_printed = True

        if status == "complete":
            return clip

        if (not wait_for_complete) and status in {"streaming", "complete"} and audio_url:
            return clip

        if status in {"error", "failed"}:
            md = clip.get("metadata") or {}
            raise RuntimeError(
                f"Generation failed. error_type={md.get('error_type')} error_message={md.get('error_message')}"
            )

        if elapsed >= max_seconds:
            raise TimeoutError(f"Polling timed out after {max_seconds}s. Last status={status}")

        time.sleep(poll_interval)


def build_audio_context_stub(audio_file_path: Optional[str]) -> Dict[str, Any]:
    """
    Your Colab used a real audio context. In a live EEG app you often do not have one.
    This stub keeps the GPT prompt generator stable.
    """
    if audio_file_path:
        p = Path(audio_file_path)
        return {
            "filename": p.name,
            "format": p.suffix.lower().lstrip("."),
            "subtype": "unknown",
            "samplerate": None,
            "channels": None,
            "duration_seconds": None,
            "filesize_mb": round(p.stat().st_size / (1024 ** 2), 3) if p.exists() else None,
            "preview_rms": None,
        }
    return {
        "filename": None,
        "format": None,
        "subtype": None,
        "samplerate": None,
        "channels": None,
        "duration_seconds": None,
        "filesize_mb": None,
        "preview_rms": None,
    }


def download_suno_audio(audio_url: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(audio_url, timeout=120)
    r.raise_for_status()
    out_path.write_bytes(r.content)
    return out_path


def run_suno_from_emotion_probs(
    emotion_probs: Dict[str, float],
    out_dir: Path,
    reference_audio_path: Optional[str] = None,
    cover_clip_id: str = "AUTO_FROM_AUDIO",
    run_transcription: bool = False,
    wait_for_complete_seed: bool = False,
    wait_for_complete_final: bool = False
) -> Dict[str, Any]:
    """
    Main entry point for Flask.

    emotion_probs must contain the same keys as EMOTION_LABELS.
    out_dir is where generated WAV is saved.

    Returns dict with:
      {
        "seed_clip_id": "...",
        "final_clip_id": "...",
        "audio_url": "...",
        "wav_path": "...",
        "seed_params": {...},
        "final_params": {...}
      }
    """

    openai_key = _require_env("OPENAI_API_KEY")
    suno_token = _require_env("SUNO_API_TOKEN")

    client = OpenAI(api_key=openai_key)

    audio_context = build_audio_context_stub(reference_audio_path)
    transcript = ""

    if run_transcription:
        raise NotImplementedError(
            "Transcription is not implemented in this backend file. "
            "If you want it, you must provide a real audio file and enable a transcription path."
        )

    seed_params, _seed_raw = generate_suno_params_with_gpt(
        client=client,
        emotion_probs=emotion_probs,
        audio_context=audio_context,
        transcript=transcript,
        purpose="seed_from_audio",
        model=OPENAI_MODEL
    )

    seed_obj = suno_generate_clip(token=suno_token, suno_params=seed_params, cover_clip_id=None)
    seed_clip_id = seed_obj.get("id")
    if not seed_clip_id:
        raise RuntimeError(f"Suno seed generation did not return id. Response: {seed_obj}")

    _seed_clip = poll_clip(
        token=suno_token,
        clip_id=seed_clip_id,
        poll_interval=POLL_INTERVAL_SECONDS,
        max_seconds=MAX_POLL_SECONDS,
        wait_for_complete=wait_for_complete_seed
    )

    if _needs_auto_cover_id(cover_clip_id):
        resolved_cover_clip_id = seed_clip_id
    else:
        resolved_cover_clip_id = _extract_cover_clip_id(cover_clip_id)

    final_params, _final_raw = generate_suno_params_with_gpt(
        client=client,
        emotion_probs=emotion_probs,
        audio_context=audio_context,
        transcript=transcript,
        purpose="cover_from_seed",
        model=OPENAI_MODEL
    )

    final_obj = suno_generate_clip(token=suno_token, suno_params=final_params, cover_clip_id=resolved_cover_clip_id)
    final_clip_id = final_obj.get("id")
    if not final_clip_id:
        raise RuntimeError(f"Suno final generation did not return id. Response: {final_obj}")

    final_clip = poll_clip(
        token=suno_token,
        clip_id=final_clip_id,
        poll_interval=POLL_INTERVAL_SECONDS,
        max_seconds=MAX_POLL_SECONDS,
        wait_for_complete=wait_for_complete_final
    )

    audio_url = final_clip.get("audio_url")
    if not audio_url:
        raise RuntimeError(f"No audio_url returned in final clip. Clip: {final_clip}")

    wav_path = out_dir / "suno_generated.wav"
    download_suno_audio(audio_url, wav_path)

    return {
        "seed_clip_id": seed_clip_id,
        "final_clip_id": final_clip_id,
        "audio_url": audio_url,
        "wav_path": str(wav_path),
        "seed_params": seed_params,
        "final_params": final_params,
        "emotion_rollup": _emotion_rollup(emotion_probs),
        "top_emotions": _top_emotions(emotion_probs, k=4),
    }


if __name__ == "__main__":
    # CLI test harness (optional)
    # This expects you to set OPENAI_API_KEY and SUNO_API_TOKEN in your environment.

    test_probs = normalize_emotion_vector(
        [0.18, 0.14, 0.19, 0.10, 0.08, 0.07, 0.04, 0.06, 0.14]
    )

    out = run_suno_from_emotion_probs(
        emotion_probs=test_probs,
        out_dir=Path("../data/recordings"),
        reference_audio_path=None,
        cover_clip_id="AUTO_FROM_AUDIO",
        run_transcription=False,
        wait_for_complete_seed=False,
        wait_for_complete_final=True
    )

    print(json.dumps(out, indent=2))
