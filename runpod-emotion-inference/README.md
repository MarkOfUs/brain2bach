# RunPod Serverless: .mat -> emotion predictions

This worker accepts a **MATLAB `.mat` file** (base64 or URL) and returns model predictions as JSON.

The handler style matches RunPod's Serverless docs/blog:
- `handler(job)` reads `job["input"]` and returns a JSON-serializable result. citeturn2view0turn2view1
- Docker worker runs `python -u handler.py` inside the container. citeturn1view0

## 1) Add your model checkpoint

Place your torch checkpoint at:

```
models/cnn_faced_best.pt
```

(or change `CKPT_PATH` env var / Dockerfile copy path).

## 2) Build

RunPod recommends building for `linux/amd64` (important on Apple Silicon). citeturn1view0

```bash
docker build --platform linux/amd64 -t YOUR_DOCKERHUB_USERNAME/eeg-mat-infer:latest .
```

## 3) Push (Step 5)

```bash
docker login
docker push YOUR_DOCKERHUB_USERNAME/eeg-mat-infer:latest
```
citeturn1view0

## 4) Request format

Send either:
- `mat_b64`: base64 of the `.mat` bytes
- OR `mat_url`: a public URL to the `.mat`

Example JSON body:

```json
{
  "input": {
    "mat_b64": "BASE64_HERE",
    "topk": 3,
    "window_start": 0,
    "window_end": 250,
    "mat_channel_names": ["FP1", "FZ", "FP2"]
  }
}
```

RunPod endpoints accept requests on `/run` (async) and `/runsync` (sync). citeturn4search0turn4search10

## 5) Call your endpoint (runsync)

```bash
curl --request POST   --url https://api.runpod.ai/v2/$ENDPOINT_ID/runsync   -H "accept: application/json"   -H "authorization: $RUNPOD_API_KEY"   -H "content-type: application/json"   -d @payload.json
```
citeturn4search0
