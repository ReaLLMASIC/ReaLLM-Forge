"""Viatom / Checkme O2 ring: the web API (FastAPI), mounted by ../server.py at /api/d/oxygen.

ble_worker.py does the Bluetooth and writes small JSON files next to it; this reads them:
  data.json     latest reading {spo2, hr, battery, status}   (the server streams it to the page)
  debug.json    radio / link / rate telemetry, once a second
  devices.json  rings seen by the last scan
Recordings: <Documents>/Bio-dash/Viatom O2/<device>/<date>/<time>_vitals.csv (unchanged).
"""
import json
import os
import time

from fastapi import FastAPI
from pydantic import BaseModel

import os as _os, sys as _sys
_sys.path.insert(1, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # shared bt_debug.py, in the release folder
import bt_debug

BASE_DIR = bt_debug.state_dir(os.path.dirname(os.path.abspath(__file__)))   # one more per extra person: see bt_debug.state_dir
DATA_FILE = os.path.join(BASE_DIR, "data.json")
DEVICES_FILE = os.path.join(BASE_DIR, "devices.json")
COMMAND_FILE = os.path.join(BASE_DIR, "command.json")
DEBUG_FILE = os.path.join(BASE_DIR, "debug.json")
HISTORY_MAX_POINTS = 1500

# ble_worker.py's words -> the state every device page uses
STATES = {"Connected": "streaming", "Calibrating": "streaming", "Connecting": "connecting", "Scanning": "scanning"}

app = FastAPI()


def _read_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def _command(cmd):
    tmp = COMMAND_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cmd, f)
    os.replace(tmp, COMMAND_FILE)          # atomic: the worker never reads half a command


@app.get("/api/status")
def status():
    """Same shape as the other devices: {state, address, debug, live}."""
    dbg = _read_json(DEBUG_FILE, {})
    live = _read_json(DATA_FILE, {})
    word = live.get("status") or dbg.get("state") or "Scanning"
    dev = dbg.get("device") or {}
    return {"state": STATES.get(word, "scanning"), "detail": word, "address": dev.get("address"),
            "live": live, "debug": dbg, "last_device": LAST_DEVICES["device"].load()}


@app.get("/api/live")
def live():
    return _read_json(DATA_FILE, {})


@app.get("/api/debug")
def debug_info():
    return _read_json(DEBUG_FILE, {})


@app.get("/api/scan-results")
def scan_results():
    return _read_json(DEVICES_FILE, [])


def _own_recordings(limit=12):
    """This ring's newest recordings, oldest first (never another person's ring)."""
    now = bt_debug.current_session_file(status(), "Viatom O2", "vitals", LAST_DEVICES["device"].load())
    if not now:
        return []
    day_dir = os.path.dirname(now)
    unit = os.path.dirname(day_dir)
    out = []
    try:
        for day in sorted(os.listdir(unit), reverse=True)[:3]:
            folder = os.path.join(unit, day)
            out += [os.path.join(folder, n) for n in sorted(os.listdir(folder), reverse=True) if n.endswith("_vitals.csv")]
    except OSError:
        pass
    return list(reversed(out[:limit]))


@app.get("/api/history")
def history(minutes: int = 15):
    """Trend history from the recordings (one row per ~5 s while connected), this ring only."""
    minutes = max(1, min(minutes, 24 * 60))
    cutoff_ms = (time.time() - minutes * 60) * 1000
    out = {"spo2": [], "hr": [], "battery": []}
    for path in _own_recordings():
        try:
            with open(path) as f:
                next(f, None)                                  # header
                for line in f:
                    p = line.strip().split(",")
                    try:
                        x, spo2, hr, batt = int(p[0]), int(float(p[1])), int(float(p[2])), int(float(p[3]))
                    except (ValueError, IndexError):
                        continue
                    if x < cutoff_ms:
                        continue
                    if 40 <= spo2 <= 100: out["spo2"].append({"x": x, "y": spo2})
                    if hr > 0:            out["hr"].append({"x": x, "y": hr})
                    if batt > 0:          out["battery"].append({"x": x, "y": batt})
        except OSError:
            continue
    for k, pts in out.items():
        pts.sort(key=lambda p: p["x"])
        out[k] = pts[:: max(1, len(pts) // HISTORY_MAX_POINTS)]
    return out


class ConnectRequest(BaseModel):
    address: str


@app.post("/api/connect")
def connect(req: ConnectRequest):
    _command({"action": "connect", "address": req.address})
    return {"status": "command_sent"}


# ---------- auto-reconnect to the last device + radio pin (same as every device) ----------
LAST_DEVICES = {"device": bt_debug.LastDevice(os.path.join(BASE_DIR, "last_device.json"))}
radio_pin = bt_debug.RadioPin(os.path.join(BASE_DIR, "radio_config.json"))


class LastDeviceRequest(BaseModel):
    role: str = "device"
    auto: bool | None = None
    forget: bool = False


@app.get("/api/last-device")
def get_last_device():
    return {"devices": {role: ld.load() for role, ld in LAST_DEVICES.items()}}


@app.post("/api/last-device")
def set_last_device(req: LastDeviceRequest):
    ld = LAST_DEVICES.get(req.role)
    if ld is None:
        return {"error": f"unknown role {req.role}"}
    if req.forget:
        ld.forget()
    elif req.auto is not None:
        ld.set_auto(req.auto)
    return get_last_device()


class RadioRequest(BaseModel):
    address: str = "auto"


@app.get("/api/radio")
def get_radio():
    return {"radio": radio_pin.config()}


@app.post("/api/radio")
def set_radio(req: RadioRequest):
    """Pin the ring to a radio by adapter address ("auto" to unpin); the worker re-resolves within a second."""
    radio_pin.save(req.address)
    return {"radio": radio_pin.config()}
