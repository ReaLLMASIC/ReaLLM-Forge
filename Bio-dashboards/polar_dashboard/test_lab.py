#!/usr/bin/env python3
"""The /lab viewer (ReaLLM-Forge PR #924's Stream Lane Lab, with the live strap as a source): the forecast
service's GET /rows and GET /signal, then both routes through api.py (the Lab
page itself is served by server.py: test_server.py). Needs no PyTorch or model weights (the forecast itself is covered by test_forecast_api.py)."""
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import forecast_service as fs  # noqa: E402

ok = 0


def check(name, cond, detail=""):
    global ok
    print(("  ok  " if cond else "  FAIL ") + name + ("" if cond else f"  {detail}"))
    if not cond:
        raise SystemExit(1)
    ok += 1


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# ---------------- a service with one ECG overlay and no model loaded
svc = fs.Service.__new__(fs.Service)
svc.torch, svc.device_name, svc.state, svc.detail = None, "cpu", "forecasting", ""
ov = fs.Overlay("ecg", os.path.join(tempfile.mkdtemp(), "no-model"))
ov.cfg_error = None                       # config defaults: one ECG lane, µV integers, 8 ms rows
ov.lanes[0]["int_max"] = 2500             # as in the real models' config.json
svc.overlays = {"ecg": ov}
for i in range(300):                      # more than the 256-row ring: the oldest 44 fall out
    with ov.lock:
        ov.ring.append([i - 100])
        ov.times.append(1000.0 + 8.0 * i)

code, r = svc.api_rows({"model_id": "ecg"})
check("rows: first call returns the whole ring, as a restart", code == 200 and r["restart"] and len(r["rows"]) == 256)
check("rows: real units (µV integers -> mV) and strap times",
      r["rows"][0] == [round((44 - 100) / 1000, 4)] and r["times"][0] == 1000.0 + 8 * 44 and r["interval_ms"] == 8.0)
check("rows: column named with its unit", r["columns"] == ["ecg_mV"])
last, piece = r["times"][-1], r["piece"]
code, r = svc.api_rows({"model_id": "ecg", "after": str(last), "piece": str(piece)})
check("rows: nothing new after the newest row", code == 200 and not r["restart"] and r["rows"] == [])
for i in range(300, 305):
    with ov.lock:
        ov.ring.append([i - 100])
        ov.times.append(1000.0 + 8.0 * i)
code, r = svc.api_rows({"model_id": "ecg", "after": str(last), "piece": str(piece)})
check("rows: only the 5 rows after `after`", not r["restart"] and len(r["rows"]) == 5 and r["times"][0] == last + 8)
code, r = svc.api_rows({"model_id": "ecg", "after": "500", "piece": str(piece)})
check("rows: `after` older than the ring -> restart with everything", r["restart"] and len(r["rows"]) == 256)
ov.reset()
code, r = svc.api_rows({"model_id": "ecg", "after": str(last), "piece": str(piece)})
check("rows: a gap / reset bumps the piece -> restart", r["restart"] and r["piece"] == piece + 1 and r["rows"] == [])
check("rows: unknown model -> 404", svc.api_rows({"model_id": "nope"})[0] == 404)
check("rows: a bad `after` -> 422", svc.api_rows({"model_id": "ecg", "after": "soon"})[0] == 422)

# ---------------- the sine source: identical to PR #924's lane_lab/data.py sine_series
code, s = svc.api_signal({"seed": "2026", "frames": "5", "channels": "2"})
ref = [[-0.3733466565608978, 0.6280932426452637], [-0.3513078987598419, 0.6573283076286316],
       [-0.3362863063812256, 0.6813640594482422], [-0.3266359269618988, 0.6964678168296814],
       [-0.3200916647911072, 0.6993130445480347]]
check("signal: same values as PR #924's sine_series (seed 2026, 2 channels)",
      code == 200 and all(abs(a - b) < 1e-6 for ra, rb in zip(s["values"], ref) for a, b in zip(ra, rb)))
code, s = svc.api_signal({"seed": "7", "frames": "3", "start": "1000"})
check("signal: `start` offsets the frames (seed 7 from frame 1000)",
      all(abs(a[0] - b) < 1e-6 for a, b in zip(s["values"], [-0.11872991919517517, -0.07611604034900665, -0.02816060185432434])))
check("signal: limits -> 422", svc.api_signal({"frames": "0"})[0] == 422 and svc.api_signal({"channels": "9"})[0] == 422)

# ---------------- end to end: the service's HTTP API + api.py
API, APP = free_port(), free_port()
server = fs.start_api(svc, API)
check("service API listening", server is not None)
for i in range(10):
    with ov.lock:
        ov.ring.append([i * 10])
        ov.times.append(50000.0 + 8.0 * i)
tmp = tempfile.mkdtemp()
env = dict(os.environ, BIODASH_DATA_DIR=os.path.join(tmp, "data"), BIODASH_FORECAST_API_PORT=str(API))
app = subprocess.Popen([sys.executable, "-c", f"import uvicorn, api; uvicorn.run(api.app, host='127.0.0.1', port={APP}, log_level='warning')"],
                       cwd=HERE, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def get(path):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{APP}{path}", timeout=10) as r:
            return r.status, r.read(), r.headers
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers


try:
    for _ in range(100):
        try:
            code, _, _ = get("/api/status")
            break
        except OSError:
            time.sleep(0.2)
    code, body, _ = get("/api/lab/rows?model_id=ecg")
    r = json.loads(body)
    check("GET /api/lab/rows passes the live rows through", code == 200 and r["rows"] == [[0.0], [0.01], [0.02], [0.03], [0.04],
                                                                                     [0.05], [0.06], [0.07], [0.08], [0.09]])
    code, body, _ = get("/api/signal?seed=2026&frames=5&channels=2")
    check("GET /api/signal passes the sine source through", code == 200 and len(json.loads(body)["values"]) == 5)
    code, body, _ = get("/api/models")
    m = json.loads(body)
    check("GET /api/models lists the ECG model for the viewer", code == 200 and [x["id"] for x in m["models"]] == ["ecg"])
    server.shutdown()
    code, body, _ = get("/api/lab/rows?model_id=ecg")
    check("service down -> 503 with a reason", code == 503 and "forecast service" in json.loads(body)["detail"])
finally:
    app.terminate()
    app.wait(timeout=10)

print(f"{ok} checks passed")
