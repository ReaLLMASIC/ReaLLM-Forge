#!/usr/bin/env python3
"""The stream-lane-lab API (ReaLLM-Forge PR #924) through the Heart API: GET /api/models and POST /api/forecast,
end to end (api.py + forecast_service.py on spare ports, a scratch data folder). Needs PyTorch, transformers
and the model weights (run it with the forecast Python); skips otherwise. CPU is fine (slow)."""
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "forecast", "models")
try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
except ImportError:
    print("skipped: PyTorch / transformers aren't installed for this Python")
    sys.exit(0)
if not any(os.path.exists(os.path.join(MODELS, "ecg", w)) for w in ("model.safetensors", "pytorch_model.bin")):
    print("skipped: no weights in forecast/models/ecg")
    sys.exit(0)

APP, API = 5091, 5092
tmp = tempfile.mkdtemp()
env = dict(os.environ, BIODASH_DATA_DIR=os.path.join(tmp, "data"), BIODASH_FORECAST_API_PORT=str(API),
           BIODASH_FORECAST_LOG="0", BIODASH_FORECAST_DEVICE="cpu")
app = subprocess.Popen([sys.executable, "-c", f"import uvicorn, api; uvicorn.run(api.app, host='127.0.0.1', port={APP}, log_level='warning')"],
                       cwd=HERE, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
svc = subprocess.Popen([sys.executable, "forecast_service.py", "--url", f"http://127.0.0.1:{APP}", "--device", "cpu"],
                       cwd=HERE, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
ok = 0


def check(name, cond, detail=""):
    global ok
    print(("  ok  " if cond else "  FAIL ") + name + ("" if cond else f"  {detail}"))
    if not cond:
        raise SystemExit(1)
    ok += 1


def call(path, body=None):
    req = urllib.request.Request(f"http://127.0.0.1:{APP}{path}", data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"null")


code, models = 0, None
try:
    for _ in range(120):                                    # the service imports torch: give it time
        try:
            code, models = call("/api/models")
            if code == 200:
                break
        except (urllib.error.URLError, OSError, ValueError):
            pass
        time.sleep(1)
    check("GET /api/models answers", code == 200, models)
    ids = {m["id"]: m for m in models["models"]}
    check("the ECG model is listed with its column", "ecg" in ids and ids["ecg"]["columns"][0]["name"] == "ecg_mV"
          and ids["ecg"]["columns"][0]["minimum"] == -2.5 and ids["ecg"]["interval_ms"] == 8.0, ids.get("ecg"))
    import math
    hist = [[round(0.9 * math.exp(-((t % 100 - 50) / 2.0) ** 2) - 0.05, 3)] for t in range(200)]   # a spike every 0.8 s
    code, out = call("/api/forecast", {"model_id": "ecg", "history": hist, "horizon": 16, "mode": "mean"})
    check("POST /api/forecast (mean)", code == 200, out)
    check("predictions: horizon rows x 1 column, in mV", len(out["predictions"]) == 16 and len(out["predictions"][0]) == 1
          and all(-2.5 <= r[0] <= 2.5 for r in out["predictions"]), out["predictions"][:3])
    check("PR #924's fields", all(k in out for k in ("latency_ms", "model_id", "history_frames_used", "horizon", "mode",
                                                     "model_loaded", "out_of_range_counts", "note")), sorted(out))
    check("history trimmed to fit the window", out["history_frames_used"] == min(200, 256 - 16), out["history_frames_used"])
    c1, a1 = call("/api/forecast", {"model_id": "ecg", "history": hist, "horizon": 8, "mode": "sample", "seed": 7})
    c2, a2 = call("/api/forecast", {"model_id": "ecg", "history": hist, "horizon": 8, "mode": "sample", "seed": 7})
    check("sample: the same seed gives the same forecast", c1 == c2 == 200 and a1["predictions"] == a2["predictions"])
    code, out = call("/api/forecast", {"model_id": "ecg", "history": hist, "horizon": 4, "mode": "greedy"})
    check("greedy", code == 200 and len(out["predictions"]) == 4, out)
    code, out = call("/api/forecast", {"model_id": "ecg", "history": [[9.0]] * 20, "horizon": 2})
    check("out-of-range values are clipped and counted", code == 200 and out["out_of_range_counts"] == [20], out)
    check("unknown model -> 404", call("/api/forecast", {"model_id": "nope", "history": hist})[0] == 404)
    check("wrong column count -> 422", call("/api/forecast", {"model_id": "ecg", "history": [[0.1, 0.2]]})[0] == 422)
    check("horizon over 128 -> 422", call("/api/forecast", {"model_id": "ecg", "history": hist, "horizon": 500})[0] == 422)
    check("non-finite history -> 422", call("/api/forecast", {"model_id": "ecg", "history": [["nan"]]})[0] == 422)
    if "rr" in ids:                                         # V6.3: the ECG + R-R model
        check("the R-R model is listed with ECG (mV) and R-R (ms) columns",
              [c["name"] for c in ids["rr"]["columns"]] == ["ecg_mV", "rr_ms"]
              and ids["rr"]["columns"][1]["maximum"] == 2500, ids["rr"])
        code, out = call("/api/forecast", {"model_id": "rr", "history": [[h[0], 800.0] for h in hist], "horizon": 8,
                                           "mode": "mean"})
        check("POST /api/forecast with the R-R model: rows of [mV, ms]", code == 200 and len(out["predictions"]) == 8
              and all(len(r) == 2 and 0 <= r[1] <= 2500 for r in out["predictions"]), out)
finally:
    for p in (svc, app):
        p.terminate()
    for p in (svc, app):
        try:
            p.wait(10)
        except subprocess.TimeoutExpired:
            p.kill()
print(f"{ok} checks passed")
