\
import argparse, base64, json, os, requests

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", required=True, help="Path to .mat file")
    ap.add_argument("--endpoint", default=os.getenv("ENDPOINT_ID"), help="RunPod endpoint id")
    ap.add_argument("--api-key", default=os.getenv("RUNPOD_API_KEY"), help="RunPod API key")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--window-start", type=int, default=0)
    ap.add_argument("--window-end", type=int, default=250)
    args = ap.parse_args()

    if not args.endpoint or not args.api_key:
        raise SystemExit("Provide --endpoint/--api-key or set ENDPOINT_ID/RUNPOD_API_KEY env vars.")

    mat_b64 = base64.b64encode(open(args.mat, "rb").read()).decode("utf-8")

    payload = {
        "input": {
            "mat_b64": mat_b64,
            "topk": args.topk,
            "window_start": args.window_start,
            "window_end": args.window_end,
        }
    }

    url = f"https://api.runpod.ai/v2/{args.endpoint}/runsync"
    r = requests.post(
        url,
        headers={"authorization": args.api_key, "content-type": "application/json", "accept": "application/json"},
        data=json.dumps(payload),
        timeout=120,
    )
    print("HTTP", r.status_code)
    print(r.text)

if __name__ == "__main__":
    main()
