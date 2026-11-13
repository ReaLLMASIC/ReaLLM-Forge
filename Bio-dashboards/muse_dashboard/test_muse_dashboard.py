"""The headband end to end with the stand-in (fake_muse.py): worker -> files -> API.
Run:  python3 muse_dashboard/test_muse_dashboard.py      (about a minute; the Brain page is checked by ../test_web.py)

It runs in a scratch copy of this folder with its own HOME and data folder, so nothing here or in
~/Documents is touched."""
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ok = 0


def check(name, cond, detail=""):
    global ok
    print(("  ok  " if cond else "  FAIL ") + name + ("" if cond else f"  {detail}"))
    if not cond:
        raise SystemExit(1)
    ok += 1


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


tmp = tempfile.mkdtemp(prefix="muse-test-")
app_dir = os.path.join(tmp, "muse_dashboard")
shutil.copytree(HERE, app_dir, ignore=shutil.ignore_patterns("__pycache__", "logs_muse", "*.json", "test_data"))
data = os.path.join(tmp, "data")
port = free_port()
env = {**os.environ, "HOME": tmp, "XDG_CONFIG_HOME": os.path.join(tmp, ".config"), "BIODASH_DATA_DIR": data,
       "PYTHONUNBUFFERED": "1", "PYTHONPATH": os.path.dirname(HERE)}          # the shared bt_debug.py
log = open(os.path.join(tmp, "log.txt"), "w")
procs = [subprocess.Popen([sys.executable, "fake_muse.py"], cwd=app_dir, env=env, stdout=log, stderr=subprocess.STDOUT),
         subprocess.Popen([sys.executable, "-c", f"import uvicorn, api; uvicorn.run(api.app, host='127.0.0.1', port={port}, log_level='warning')"], cwd=app_dir, env=env, stdout=log, stderr=subprocess.STDOUT)]
base = f"http://127.0.0.1:{port}"


def get(path):
    with urllib.request.urlopen(base + path, timeout=5) as r:
        return json.loads(r.read())


def post(path, body=None):
    req = urllib.request.Request(base + path, data=json.dumps(body or {}).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def wait(cond, timeout, step=0.5):
    end = time.time() + timeout
    while time.time() < end:
        try:
            v = cond()
            if v:
                return v
        except Exception:
            pass
        time.sleep(step)
    return None


def session_files():
    root = os.path.join(data, "Muse")
    out = {}
    for dp, _, files in os.walk(root):
        for f in files:
            out.setdefault(f.split("_", 1)[0], {})[f.split("_", 1)[1][:-4]] = os.path.join(dp, f)
    return dict(sorted(out.items()))


def header(path):
    with open(path) as f:
        return f.readline().strip().split(",")


try:
    devices = wait(lambda: get("/api/scan-results"), 20)
    check(f"the scanner lists the headband: {devices and devices[0]['name']}", devices and devices[0]["name"] == "MuseS-Demo-0000")
    post("/api/connect", {"address": devices[0]["address"]})
    st = wait(lambda: (lambda s: s if s["state"] == "streaming" and s["debug"]["rates"].get("eeg_hz", 0) > 200 and s["debug"]["signals"].get("ppg", {}).get("hr") else None)(get("/api/status")), 40)
    check("connects and streams", bool(st), get("/api/status").get("state"))
    d = st["debug"]
    r = d["rates"]
    check(f"rates: EEG {r['eeg_hz']} / 256, optics {r['optics_hz']} / 64, motion {r['imu_hz']} / 52 Hz",
          abs(r["eeg_hz"] - 256) < 8 and abs(r["optics_hz"] - 64) < 3 and abs(r["imu_hz"] - 52) < 3)
    check("nothing dropped, no clock restarts", not any(r["dropped"].values()) and r["clock_reanchors"] == 0, r)
    check(f"the headband's answers: fw {d['info'].get('fw')}, battery {d['info'].get('bp')} %", d["info"].get("fw") == "3.2.1-demo" and d["info"].get("bp") == 87.5)
    check("all four EEG sensors in contact", set(d["signals"]["contact"].values()) == {"good"}, d["signals"]["contact"])
    ppg = d["signals"]["ppg"]
    check(f"pulse from {ppg['channel']}: {ppg['hr']} bpm (the stand-in beats at 66)", abs(ppg["hr"] - 66) <= 2 and ppg["channel"].endswith("IR"))

    files = session_files()
    first = list(files.values())[0]
    check(f"the session's files: {', '.join(sorted(first))}", set(first) == {"eeg", "optics", "imu", "bands", "heart", "fnirs", "battery"})
    check("EEG: 8 channels in uV", header(first["eeg"]) == ["Timestamp_Epoch_ms", "TP9_uV", "AF7_uV", "AF8_uV", "TP10_uV", "AUX1_uV", "AUX2_uV", "AUX3_uV", "AUX4_uV"])
    check("optics: 16 channels (p1041)", len(header(first["optics"])) == 17 and header(first["optics"])[1] == "LO_NIR")
    with open(first["eeg"]) as f:
        rows = [l.split(",") for l in f.read().strip().split("\n")[1:]]
    ts = [int(x[0]) for x in rows]
    steps = sorted(b - a for a, b in zip(ts, ts[1:]))
    check(f"EEG timestamps: 256 Hz, never backwards ({len(ts)} rows)", steps[0] >= 0 and 3 <= steps[len(steps) // 2] <= 4)
    check("EEG near the computer's clock", abs(time.time() * 1000 - ts[-1]) < 2000)
    with open(first["bands"]) as f:
        b = f.read().strip().split("\n")
    hb = b[0].split(",")
    last = dict(zip(hb, b[-1].split(",")))
    check("bands: TP9's strongest is alpha (the stand-in's 10 Hz)", max(("delta", "theta", "alpha", "beta", "gamma"), key=lambda k: float(last[f"TP9_{k}"])) == "alpha")
    check("bands: AF7's beta beats its alpha (18 Hz)", float(last["AF7_beta"]) > float(last["AF7_alpha"]))
    h = wait(lambda: get("/api/history?metric=hr&minutes=5")["hr"], 10)
    check(f"history for the heart-rate tile: {len(h or [])} points", h and abs(h[-1]["y"] - 66) <= 2)
    a = wait(lambda: get("/api/history?metric=alpha&minutes=5")["alpha"], 10)
    check(f"history for the alpha tile: {a and a[-1]['y']} %", a and 10 < a[-1]["y"] < 90)

    # a new preset: the session restarts with its own files (EEG 4 channels, no optics)
    post("/api/preset", {"preset": "p21"})
    new = wait(lambda: (lambda f: f if len(f) > 1 else None)(session_files()), 30)
    check("preset p21: a new session", bool(new))
    second = list(new.values())[-1]
    check(f"preset p21: its files: {', '.join(sorted(second))}", "optics" not in second and "fnirs" not in second)
    check("preset p21: EEG with 4 channels", wait(lambda: header(second["eeg"]) == ["Timestamp_Epoch_ms", "TP9_uV", "AF7_uV", "AF8_uV", "TP10_uV"], 5))
    st = wait(lambda: (lambda s: s if s["debug"]["rates"].get("eeg_hz", 0) > 200 and s["debug"].get("preset") == "p21" else None)(get("/api/status")), 20)
    check("preset p21: streaming EEG again", bool(st))

    post("/api/disconnect")
    st = wait(lambda: (lambda s: s if s["state"] == "scanning" and s["auto_held"] else None)(get("/api/status")), 20)
    check("Disconnect: back to the scanner, no reconnecting by itself", bool(st))
    print(f"{ok} checks passed")
finally:
    for p in procs:
        p.terminate()
    for p in procs:
        try:
            p.wait(5)
        except subprocess.TimeoutExpired:
            p.kill()
    log.close()
    if os.environ.get("KEEP"):
        print("kept", tmp)
    else:
        shutil.rmtree(tmp, ignore_errors=True)
