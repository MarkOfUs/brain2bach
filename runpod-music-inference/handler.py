import base64
import runpod

from music_inference import load_models, mat_bytes_to_wav_bytes

_MODELS = None

def _get_models():
    global _MODELS
    if _MODELS is None:
        _MODELS = load_models()
    return _MODELS

def handler(job):
    try:
        inp = job.get("input", {}) or {}

        # Required input: capture.mat bytes encoded as base64
        mat_b64 = inp.get("mat_b64")
        if not mat_b64:
            return {"error": "Missing required field: input.mat_b64 (base64-encoded .mat bytes)"}

        # Optional knobs (defaults match your training/inference script)
        target_t = int(inp.get("target_t", 2101))
        griffin_iters = int(inp.get("griffin_iters", 32))
        use_refiner = bool(inp.get("use_refiner", True))
        include_generated = bool(inp.get("include_generated", False))

        # Optional: mat keys and channel columns for different capture formats
        mat_key_data = inp.get("mat_key_data", "data")
        mat_key_time = inp.get("mat_key_time", "t_sec")
        channel_cols = inp.get("channel_cols", [1, 2, 3])  # 0-based column indices in `data`

        mat_bytes = base64.b64decode(mat_b64)

        wav_bytes, meta, wav_gen_bytes = mat_bytes_to_wav_bytes(
            _get_models(),
            mat_bytes,
            target_t=target_t,
            griffin_iters=griffin_iters,
            use_refiner=use_refiner,
            mat_key_data=mat_key_data,
            mat_key_time=mat_key_time,
            channel_cols=channel_cols,
        )

        out = {
            "wav_b64": base64.b64encode(wav_bytes).decode("utf-8"),
            "meta": meta,
        }

        if include_generated and wav_gen_bytes is not None:
            out["wav_gen_b64"] = base64.b64encode(wav_gen_bytes).decode("utf-8")

        return out

    except Exception as e:
        # Return a clean error for runsync
        return {"error": f"{type(e).__name__}: {e}"}

runpod.serverless.start({"handler": handler})
