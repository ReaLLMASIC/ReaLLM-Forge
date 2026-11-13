"""Muse S Athena headband: the web API (FastAPI), mounted by ../server.py at /api/d/brain.

muse_worker.py records <Documents>/Bio-dash/Muse/<headband>/<date>/<time>_<stream>.csv; this follows the
newest of each stream and hands what arrived since the last look to the server as one message
{"type": stream, "rows": [...]} (eeg, optics, imu, bands, heart, fnirs, battery), which the server passes to
the pages showing Brain. Commands go to the worker through logs_muse/command.json; its state comes back
through status.json.
"""
import asyncio
import json
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

import os as _os, sys as _sys
_sys.path.insert(1, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # shared bt_debug.py, in the release folder
import bt_debug
import muse_protocol as mp

BASE_DIR = bt_debug.state_dir(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(BASE_DIR, "logs_muse")
DEVICES_FILE = os.path.join(LOGS_DIR, "devices.json")
COMMAND_FILE = os.path.join(LOGS_DIR, "command.json")
STATUS_FILE = os.path.join(LOGS_DIR, "status.json")
SETTINGS_FILE = os.path.join(BASE_DIR, "muse_settings.json")
STREAMS = ("eeg", "optics", "imu", "bands", "heart", "fnirs", "battery")
POLL_S, ROTATE_CHECK_S = 0.02, 2.0


class ConnectionManager:
    """Who gets the live rows: the server's tap, while a page is open (see server.py Hub)."""

    def __init__(self):
        self.active_connections = set()

    async def broadcast(self, text):
        conns = list(self.active_connections)
        results = await asyncio.gather(*(c.send_text(text) for c in conns), return_exceptions=True)
        for c, r in zip(conns, results):
            if isinstance(r, Exception):
                self.active_connections.discard(c)


manager = ConnectionManager()


def latest(stream):
    files = bt_debug.latest_session_files("Muse", stream, limit=1)
    return files[0] if files else None


def parse_rows(lines):
    """CSV lines -> [[ts_ms, v, ...]]: the header and broken lines are skipped; empty cells -> null."""
    rows = []
    for line in lines:
        parts = line.split(",")
        try:
            row = [int(parts[0])]
            row += [float(p) if p != "" else None for p in parts[1:]]
        except ValueError:
            continue
        rows.append(row)
    return rows


async def follow(stream):
    """The newest <time>_<stream>.csv, as it grows (a new session's file is picked up within 2 s)."""
    loop = asyncio.get_running_loop()
    current, f, partial, next_check = None, None, "", 0.0
    try:
        while True:
            if loop.time() >= next_check:
                next_check = loop.time() + ROTATE_CHECK_S
                path = await run_in_threadpool(latest, stream)
                if path and path != current:
                    first = current is None
                    if f:
                        f.close()
                    f = open(path, "r")
                    if first and os.path.getsize(path) > 64 * 1024:
                        f.seek(0, os.SEEK_END)            # an old session at startup: don't replay it
                    current, partial = path, ""
            if f is None:
                await asyncio.sleep(1)
                continue
            if not manager.active_connections:
                f.seek(0, os.SEEK_END)                    # nobody is watching: skip ahead instead of reading 50x a second
                partial = ""
                await asyncio.sleep(0.5)
                continue
            chunk = f.read()
            if not chunk:
                await asyncio.sleep(POLL_S)
                continue
            lines = (partial + chunk).split("\n")
            partial = lines.pop()
            rows = parse_rows(l.strip() for l in lines if l.strip())
            if rows:
                    await manager.broadcast(json.dumps({"type": stream, "rows": rows}, separators=(",", ":")))
    finally:
        if f:
            f.close()


@asynccontextmanager
async def lifespan(_app):
    if os.path.exists(COMMAND_FILE):
        try:
            os.remove(COMMAND_FILE)
        except OSError:
            pass
    tasks = [asyncio.create_task(follow(s)) for s in STREAMS]
    yield
    for t in tasks:
        t.cancel()


app = FastAPI(lifespan=lifespan)
def _command(obj):
    os.makedirs(LOGS_DIR, exist_ok=True)
    with open(COMMAND_FILE + ".tmp", "w") as f:
        json.dump(obj, f)
    os.replace(COMMAND_FILE + ".tmp", COMMAND_FILE)       # atomic: the worker never reads half a command


class ConnectRequest(BaseModel):
    address: str


@app.get("/api/scan-results")
def scan_results():
    try:
        with open(DEVICES_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return []


@app.get("/api/status")
def status():
    try:
        with open(STATUS_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"state": "scanning"}


@app.post("/api/connect")
def connect(req: ConnectRequest):
    _command({"action": "connect", "address": req.address})
    return {"status": "command_sent"}


@app.post("/api/disconnect")
def disconnect():
    _command({"action": "disconnect"})
    return {"status": "command_sent"}


@app.post("/api/fnirs/zero")
def fnirs_zero():
    """A new fNIRS baseline: the next 10 s become "zero" (e.g. after settling in, before a task)."""
    _command({"action": "zero"})
    return {"status": "command_sent"}


@app.get("/api/preset")
def get_preset():
    try:
        with open(SETTINGS_FILE) as f:
            preset = json.load(f).get("preset") or mp.DEFAULT_PRESET
    except (OSError, ValueError, AttributeError):
        preset = mp.DEFAULT_PRESET
    return {"preset": preset, "presets": mp.PRESETS, "default": mp.DEFAULT_PRESET}


class PresetRequest(BaseModel):
    preset: str


@app.post("/api/preset")
def set_preset(req: PresetRequest):
    """Which streams the headband sends. A live session restarts with it (new files: the columns change)."""
    if req.preset not in mp.PRESETS:
        return JSONResponse({"error": f"unknown preset {req.preset}"}, status_code=400)
    with open(SETTINGS_FILE + ".tmp", "w") as f:
        json.dump({"preset": req.preset}, f)
    os.replace(SETTINGS_FILE + ".tmp", SETTINGS_FILE)
    _command({"action": "preset"})
    return get_preset()


# ---------- tile trends, pre-filled from this session's files ----------
HISTORY_MAX_POINTS = 1500


def _tail_lines(path, max_bytes):
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - max_bytes))
        lines = f.read().decode("utf-8", errors="replace").splitlines()
    return lines[1:] if size > max_bytes else lines


def _thin(points):
    return points[::max(1, len(points) // HISTORY_MAX_POINTS)]


@app.get("/api/history")
def history(metric: str, minutes: int = 15):
    """hr (from the beats), alpha (alpha's share of the power, mean of the channels in contact) or battery."""
    minutes = max(1, min(minutes, 60))
    cutoff = time.time() * 1000 - minutes * 60_000
    stream = {"hr": "heart", "alpha": "bands", "battery": "battery"}.get(metric)
    path = latest(stream) if stream else None
    if not path:
        return {metric: []}
    rows = [r for r in parse_rows(_tail_lines(path, minutes * 60 * (400 if stream == "bands" else 40) + 8192)) if r[0] >= cutoff]
    if metric == "hr":
        return {"hr": _thin([{"x": r[0], "y": r[2]} for r in rows if r[2]])}
    if metric == "battery":
        return {"battery": _thin([{"x": r[0], "y": r[1]} for r in rows])}
    out = []
    for r in rows:                      # bands: 4 channels x 5 bands (log10 uV^2), then 4 contact codes
        shares = []
        for c in range(4):
            if r[21 + c] in (1, 2):
                p = [10 ** v for v in r[1 + 5 * c:6 + 5 * c]]
                shares.append(p[2] / sum(p))
        if shares:
            out.append({"x": r[0], "y": round(100 * sum(shares) / len(shares), 1)})
    return {"alpha": _thin(out)}


# ---------- auto-reconnect and radio pin (same as every device) ----------
LAST_DEVICES = {"device": bt_debug.LastDevice(os.path.join(BASE_DIR, "last_device.json"))}


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


radio_pin = bt_debug.RadioPin(os.path.join(BASE_DIR, "radio_config.json"))


@app.get("/api/radio")
def get_radio():
    return {"radio": radio_pin.config()}


@app.post("/api/radio")
def set_radio(req: RadioRequest):
    radio_pin.save(req.address)
    return {"radio": radio_pin.config()}
