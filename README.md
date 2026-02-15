# Brain2Bach – EEG to Music

Generate music from brain signals using two RunPod Serverless pods plus Suno. The web app records EEG (or uses mock data), sends `.mat` snapshots to RunPod, and produces music either via emotion inference → Suno or via direct EEG → Music AI.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set environment variables
export OPENAI_API_KEY="your-openai-key"
export SUNO_API_TOKEN="your-suno-token"
export RUNPOD_API_KEY="your-runpod-key"

# 3. (Optional) Use mock EEG if no serial device
export MOCK_EEG=1

# 4. Run the Flask app
python app.py
```

Open http://127.0.0.1:5000 in your browser. Record EEG (or use mock mode), then click **Create song (Emotion → Suno)** or **Create song (Direct)**.

---

# RunPod Serverless Pods (Base64 `.mat` Flow)

This project uses **two RunPod Serverless pods** (Docker images). Both endpoints are **JSON in / JSON out**.  
To send a `.mat` file, we base64-encode the raw bytes and include it in the request payload.

---

## Endpoints (RunPod IDs)

- **Music pod endpoint ID:** `4c1o9eu4bhzzz2`
- **Emotion pod endpoint ID:** `kw804mmqrwyhzz`

### `runsync` URLs (use in cURL)

```bash
# Music
https://api.runpod.ai/v2/4c1o9eu4bhzzz2/runsync

# Emotion
https://api.runpod.ai/v2/kw804mmqrwyhzz/runsync
```

---

## Pod 1: Emotion Inference (`markofus/eeg-mat-infer:latest`)

### Input payload (JSON)

```json
{
  "input": {
    "mat_b64": "BASE64_BYTES_OF_MAT",
    "topk": 3,
    "window_start": 0,
    "window_end": 250,
    "mat_channel_names": ["FP1", "FZ", "FP2"]
  }
}
```

### Input fields

- `mat_b64` *(required)*: base64-encoded `.mat` file bytes
- `topk` *(optional)*: number of top predictions to return (default: `3`)
- `window_start`, `window_end` *(optional)*: slice a time window (in samples) before inference
- `mat_channel_names` *(optional)*: channel names for selection/mapping (only used if the worker supports it)

### Output payload (JSON)

Exact fields may vary by model/version, but the response will contain a prediction summary. Typical shape:

```json
{
  "topk": [
    {"label": "neutral", "prob": 0.41, "rank": 1},
    {"label": "joy", "prob": 0.22, "rank": 2},
    {"label": "sadness", "prob": 0.11, "rank": 3}
  ],
  "classes": ["neutral", "joy", "sadness", "..."],
  "probs": [0.41, 0.22, 0.11, "..."],
  "meta": {
    "window_start": 0,
    "window_end": 250
  }
}
```

---

## Pod 2: Music Inference (`markofus/eeg-music-infer:latest`)

This pod converts EEG `.mat` → **generated WAV audio** and returns the WAV bytes as base64.

### Input payload (JSON)

```json
{
  "input": {
    "mat_b64": "BASE64_BYTES_OF_CAPTURE_MAT",
    "target_t": 2101,
    "griffin_iters": 32,
    "use_refiner": false,
    "include_generated": false,
    "mat_key_data": "data",
    "mat_key_time": "t_sec",
    "channel_cols": [1, 2, 3]
  }
}
```

### Input fields

- `mat_b64` *(required)*: base64-encoded `.mat` file bytes
- `target_t` *(optional)*: target length after resampling to 100 Hz (default: `2101`)
- `griffin_iters` *(optional)*: Griffin–Lim iterations (default: `32`; higher = slower but cleaner audio)
- `use_refiner` *(optional)*: apply refiner model if available (default: `false`)
- `include_generated` *(optional)*: if `true`, also return the **pre-refiner** WAV as `wav_gen_b64` (default: `false`)
  - `wav_b64` is always the **main output**.
  - `wav_gen_b64` is useful for A/B comparison if `use_refiner=true`.
- `.mat parsing knobs` *(optional)*:
  - `mat_key_data` (default `"data"`)
  - `mat_key_time` (default `"t_sec"`)
  - `channel_cols` (default `[1,2,3]`)

### Output payload (JSON)

```json
{
  "wav_b64": "BASE64_WAV_BYTES",
  "meta": {
    "sr": 22050,
    "duration_s": 12.3,
    "griffin_iters": 32,
    "used_refiner": false,
    "wav_bytes": 123456
  }
}
```

If `include_generated=true`, the output may also include:

```json
{
  "wav_gen_b64": "BASE64_WAV_BYTES_PRE_REFINER"
}
```

---

## Build a base64 payload from a local `.mat`

This snippet reads `/content/capture.mat`, base64-encodes it, and writes a request JSON payload.

```python
import base64, json

MAT_PATH = "/content/capture.mat"

with open(MAT_PATH, "rb") as f:
    mat_b64 = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "input": {
        "mat_b64": mat_b64
    }
}

with open("/content/payload.json", "w") as f:
    json.dump(payload, f)

print("Base64 length:", len(mat_b64))
print("Wrote /content/payload.json")
print(mat_b64[:200] + "...")
```

---

## RunPod: `runsync` Example (cURL)

Replace `$RUNPOD_API_KEY` with your API key, and use one of the endpoint IDs above.

### Music pod

```bash
curl -X POST "https://api.runpod.ai/v2/4c1o9eu4bhzzz2/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d @payload.json
```

### Emotion pod

```bash
curl -X POST "https://api.runpod.ai/v2/kw804mmqrwyhzz/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d @payload.json
```

---

## Decoding the music output WAV

After you call `runsync` and get a JSON response, decode `wav_b64` to a playable WAV:

```python
import base64
from IPython.display import Audio

# RunPod often nests the result under resp["output"].
out = resp.get("output", resp)

wav_bytes = base64.b64decode(out["wav_b64"])
with open("/content/out.wav", "wb") as f:
    f.write(wav_bytes)

Audio("/content/out.wav")
```
