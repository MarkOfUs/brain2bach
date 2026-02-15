# RunPod Serverless Pods (Base64 `.mat` Flow)

This project uses **two RunPod Serverless pods** (Docker images). Both endpoints are **JSON in / JSON out**.  
To send a `.mat` file, we base64-encode the raw bytes and include it in the request payload.

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
- `include_generated` *(optional)*: if `true`, also returns unrefined audio as `wav_gen_b64` (default: `false`)
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

## Colab: Build a base64 payload from a local `.mat`

This snippet reads `/content/capture.mat`, base64-encodes it, and writes a request JSON payload.

### Generic payload (edit per pod)

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

### Emotion pod payload example

```python
payload = {
    "input": {
        "mat_b64": mat_b64,
        "topk": 3,
        "window_start": 0,
        "window_end": 250,
        "mat_channel_names": ["FP1", "FZ", "FP2"]
    }
}
```

### Music pod payload example (refiner off)

```python
payload = {
    "input": {
        "mat_b64": mat_b64,
        "target_t": 2101,
        "griffin_iters": 32,
        "use_refiner": False,
        "include_generated": False,
        "mat_key_data": "data",
        "mat_key_time": "t_sec",
        "channel_cols": [1, 2, 3]
    }
}
```

---

## RunPod: `runsync` Example (cURL)

Replace:
- `<ENDPOINT_ID>` with your RunPod Serverless endpoint ID
- `$RUNPOD_API_KEY` with your API key

```bash
curl -X POST "https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d @payload.json
```

---

## Decoding the music output WAV (Colab)

After you call `runsync` and get a JSON response, decode `wav_b64` to a playable WAV:

```python
import base64

# If your response JSON is in `resp`, adjust as needed:
# RunPod often nests result under resp["output"].
out = resp.get("output", resp)

wav_bytes = base64.b64decode(out["wav_b64"])
with open("/content/out.wav", "wb") as f:
    f.write(wav_bytes)

from IPython.display import Audio
Audio("/content/out.wav")
```
