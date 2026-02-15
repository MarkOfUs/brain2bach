# RunPod Serverless — EEG (.mat) -> Music (WAV)

This is a RunPod Serverless worker that takes a **base64-encoded** `capture.mat` and returns a **WAV** as base64.

It wraps the same pipeline from your `three_channel_to_song.py`:
- pull 3 channels from `data[:, 1:4]`
- resample to 100 Hz, pad/crop to `target_t` (default 2101)
- Hilbert envelope + normalize
- BiLSTM -> envelope prediction
- Env2Mel -> log-mel prediction
- Griffin-Lim -> audio
- optional AudioRefiner (if `models/refiner_final.pt` exists)

## 1) Put models in `models/`

Drop your checkpoints into `models/` (filenames are defaults, but env vars can override):

- `models/sub-17_run5_bilstm.pt`
- `models/sub-17_run5_env2mel.pt`
- `models/refiner_final.pt` (optional)

## 2) Build + push

From this folder:

```bash
docker build --platform linux/amd64 -t <DOCKERHUB_USER>/eeg-music-infer:latest .
docker login
docker push <DOCKERHUB_USER>/eeg-music-infer:latest
```

Then create a new RunPod **Serverless Endpoint** using that image.

## 3) Request format

JSON (base64 `.mat`):

```json
{
  "input": {
    "mat_b64": "BASE64_BYTES_OF_CAPTURE_MAT",
    "target_t": 2101,
    "griffin_iters": 32,
    "use_refiner": true,
    "include_generated": false,
    "mat_key_data": "data",
    "mat_key_time": "t_sec",
    "channel_cols": [1,2,3]
  }
}
```

### Response format

```json
{
  "wav_b64": "...",
  "meta": { "sr": 22050, "duration_s": 20.9, ... },
  "wav_gen_b64": "..."  // only if include_generated=true
}
```

## 4) Quick client

```bash
python client_call_runsync.py \
  --mat /path/to/capture.mat \
  --endpoint <ENDPOINT_ID> \
  --api-key $RUNPOD_API_KEY \
  --out out.wav
```

## Notes

- RunPod is **JSON in / JSON out** — returning WAV as base64 keeps it self-contained.
- If you later generate much longer audio and hit payload limits, switch to uploading the WAV to object storage and returning a URL instead.
