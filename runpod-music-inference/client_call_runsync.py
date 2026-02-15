import argparse
import base64
import json
import os
import sys
from pathlib import Path

import requests


def b64_file(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", required=True, help="Path to capture.mat")
    ap.add_argument("--endpoint", required=True, help="RunPod endpoint id (the '.../v2/<id>/runsync' id)")
    ap.add_argument("--api-key", default=os.getenv("RUNPOD_API_KEY"), help="RunPod API key (or env RUNPOD_API_KEY)")
    ap.add_argument("--out", default="out.wav", help="Output wav filename")
    ap.add_argument("--griffin-iters", type=int, default=32)
    ap.add_argument("--target-t", type=int, default=2101)
    ap.add_argument("--use-refiner", action="store_true")
    ap.add_argument("--no-refiner", dest="use_refiner", action="store_false")
    ap.set_defaults(use_refiner=True)
    ap.add_argument("--include-generated", action="store_true", help="Also return wav_gen_b64 (unrefined)")
    args = ap.parse_args()

    if not args.api_key:
        print("Missing --api-key (or set RUNPOD_API_KEY).", file=sys.stderr)
        sys.exit(2)

    payload = {
        "input": {
            "mat_b64": b64_file(args.mat),
            "griffin_iters": args.griffin_iters,
            "target_t": args.target_t,
            "use_refiner": bool(args.use_refiner),
            "include_generated": bool(args.include_generated),
        }
    }

    url = f"https://api.runpod.ai/v2/{args.endpoint}/runsync"
    r = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {args.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        data=json.dumps(payload),
        timeout=600,
    )
    r.raise_for_status()
    resp = r.json()

    out = resp.get("output", resp)
    if "error" in out:
        raise RuntimeError(out["error"])

    wav_b64 = out["wav_b64"]
    wav_bytes = base64.b64decode(wav_b64)

    Path(args.out).write_bytes(wav_bytes)
    print(f"Wrote {args.out} ({len(wav_bytes)} bytes)")
    print("meta:", json.dumps(out.get("meta", {}), indent=2))


if __name__ == "__main__":
    main()
