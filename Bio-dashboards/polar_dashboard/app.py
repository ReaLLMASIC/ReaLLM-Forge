from starlette.concurrency import run_in_threadpool
import asyncio
import os
import glob
import json
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import bt_debug

WEB_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs_advanced")
DEVICES_FILE = os.path.join(LOGS_DIR, "devices.json")
COMMAND_FILE = os.path.join(LOGS_DIR, "command.json")
STATUS_FILE = os.path.join(LOGS_DIR, "status.json")

class ConnectRequest(BaseModel):
    address: str


class ConnectionManager:
    def __init__(self):
        self.active_connections: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)

    async def broadcast(self, message: str):
        conns = list(self.active_connections)
        results = await asyncio.gather(*(c.send_text(message) for c in conns), return_exceptions=True)
        for c, r in zip(conns, results):
            if isinstance(r, Exception):
                self.disconnect(c)

manager = ConnectionManager()

# CSV row layouts written by advanced_worker.py -> (column count, per-column converters)
ROW_SCHEMAS = {
    "ecg": (3, (int, float, int)),   # ts_ms, mV, hr
    "acc": (4, (int, float, float, float)),  # ts_ms, x, y, z (mG)
    "ppi": (3, (int, int, int)),     # ts_ms, ppi_ms, hr
}

POLL_S = 0.02          # how often to check the file for new bytes
ROTATE_CHECK_S = 2.0   # how often to look for a newer log file (worker restart)

# Recordings live in <Documents>/Bio-dash/Polar H10/<device>/<date>/<time>_<stream>.csv
# (bt_debug.SessionFiles); the newest session is the one being recorded now.
_STREAM_OF = {"polar_ecg": "ecg", "polar_acc": "acc", "polar_ppi": "rr"}

def get_latest_file(prefix):
    files = bt_debug.latest_session_files("Polar H10", _STREAM_OF[prefix], limit=1)
    return files[0] if files else None

def parse_rows(lines, stream_type):
    ncols, conv = ROW_SCHEMAS[stream_type]
    rows = []
    for line in lines:
        parts = line.split(",")
        if len(parts) != ncols:
            continue
        try:
            rows.append([c(p) for c, p in zip(conv, parts)])
        except ValueError:
            continue  # header row or garbage
    return rows

async def tail_file_and_broadcast(prefix, stream_type):
    """Follows the newest `prefix_*.csv`, sending everything that arrived since the
    last poll as ONE websocket message instead of one message per sample."""
    loop = asyncio.get_running_loop()
    current, f, partial = None, None, ""
    next_rotate_check = 0.0
    try:
        while True:
            now = loop.time()
            if now >= next_rotate_check:
                next_rotate_check = now + ROTATE_CHECK_S
                latest = get_latest_file(prefix)
                if latest and latest != current:
                    first_lock = current is None
                    if f:
                        f.close()
                    f = open(latest, "r")
                    # On startup, skip past a file that already holds a lot (don't replay an old
                    # session), but read a fresh one from the top so nothing is dropped.
                    if first_lock and os.path.getsize(latest) > 64 * 1024:
                        f.seek(0, os.SEEK_END)
                    current, partial = latest, ""
                    print(f"📡 Router locked onto {stream_type.upper()}: {latest}")

            if f is None:
                await asyncio.sleep(1)
                continue

            chunk = f.read()
            if not chunk:
                await asyncio.sleep(POLL_S)
                continue

            # Keep any half-written trailing line for the next read
            data = partial + chunk
            lines = data.split("\n")
            partial = lines.pop()

            if manager.active_connections:
                rows = parse_rows((l.strip() for l in lines if l.strip()), stream_type)
                if rows:
                    await manager.broadcast(json.dumps({"type": stream_type, "rows": rows}, separators=(",", ":")))
    finally:
        if f:
            f.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.path.exists(COMMAND_FILE):
        try: os.remove(COMMAND_FILE)
        except Exception: pass

    tasks = [
        asyncio.create_task(tail_file_and_broadcast("polar_ecg", "ecg")),
        asyncio.create_task(tail_file_and_broadcast("polar_acc", "acc")),
        asyncio.create_task(tail_file_and_broadcast("polar_ppi", "ppi")),
    ]
    yield
    for t in tasks:
        t.cancel()

# --- APP INITIALIZATION ---
app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=os.path.join(WEB_DIR, "static")), name="static")

# --- ENDPOINTS ---
@app.get("/api/scan-results")
def get_scan_results():
    if os.path.exists(DEVICES_FILE):
        try:
            with open(DEVICES_FILE, "r") as f:
                return json.load(f)
        except Exception: pass
    return []

HISTORY_MAX_POINTS = 1500
ECG_PP_BUCKET_MS = 1000

def _tail_lines(path, max_bytes):
    """Last lines of a CSV without reading the whole file."""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - max_bytes))
        lines = f.read().decode("utf-8", errors="replace").splitlines()
    return lines[1:] if size > max_bytes else lines      # first line may be cut

def _thin(points):
    step = max(1, len(points) // HISTORY_MAX_POINTS)
    return points[::step]

@app.get("/api/history")
def get_history(metric: str, minutes: int = 15):
    """Trend history for a tile, from this session's CSV logs.
    metric: hr | rmssd (from the R-R log) or ecg_pp (ECG peak-to-peak per second)."""
    minutes = max(1, min(minutes, 60))
    cutoff = time.time() * 1000 - minutes * 60_000

    if metric in ("hr", "rmssd"):
        path = get_latest_file("polar_ppi")
        if not path:
            return {metric: []}
        beats = []
        for line in _tail_lines(path, minutes * 60 * 4 * 32 + 8192):   # <= ~4 beats/s, ~30 B/row
            p = line.split(",")
            try:
                beats.append((int(p[0]), int(p[1]), int(p[2])))
            except (ValueError, IndexError):
                continue
        if metric == "hr":
            return {"hr": _thin([{"x": ts, "y": hr} for ts, _, hr in beats if hr > 0 and ts >= cutoff])}
        # same RMSSD as the page: last 20 R-R intervals
        out, window = [], []
        for ts, ppi, _ in beats:
            window.append(ppi)
            window = window[-20:]
            if len(window) > 2 and ts >= cutoff:
                sq = sum((window[i] - window[i - 1]) ** 2 for i in range(1, len(window)))
                out.append({"x": ts, "y": round((sq / (len(window) - 1)) ** 0.5, 1)})
        return {"rmssd": _thin(out)}

    if metric == "ecg_pp":
        path = get_latest_file("polar_ecg")
        if not path:
            return {"ecg_pp": []}
        buckets = {}
        for line in _tail_lines(path, minutes * 60 * 130 * 28 + 8192):    # 130 Hz, ~25 B/row
            p = line.split(",")
            try:
                ts, mv = int(p[0]), float(p[1])
            except (ValueError, IndexError):
                continue
            if ts < cutoff:
                continue
            b = buckets.setdefault(ts // ECG_PP_BUCKET_MS, [mv, mv])
            if mv < b[0]: b[0] = mv
            if mv > b[1]: b[1] = mv
        pts = [{"x": k * ECG_PP_BUCKET_MS + ECG_PP_BUCKET_MS // 2, "y": round(hi - lo, 3)}
               for k, (lo, hi) in sorted(buckets.items())]
        return {"ecg_pp": _thin(pts)}

    return {}

# ---------- auto-reconnect to the last device (see bt_debug.LastDevice) ----------
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
    """Toggle auto-reconnect ({"auto": true|false}) or forget ({"forget": true}).
    The worker reads the file on its next scan, so this applies within a few seconds."""
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

radio_pin = bt_debug.RadioPin(os.path.join(WEB_DIR, "radio_config.json"))

@app.get("/api/radio")
def get_radio():
    return {"radio": radio_pin.config()}

@app.post("/api/radio")
def set_radio(req: RadioRequest):
    """Pin the Polar to a radio by adapter address ("auto" to unpin). The worker
    applies it on its next scan sweep (a live connection isn't dropped)."""
    radio_pin.save(req.address)
    return {"radio": radio_pin.config()}

@app.get("/api/status")
def get_status():
    try:
        with open(STATUS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {"state": "scanning"}

@app.post("/api/connect")
def connect_device(req: ConnectRequest):
    os.makedirs(LOGS_DIR, exist_ok=True)
    tmp = COMMAND_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"action": "connect", "address": req.address}, f)
    os.replace(tmp, COMMAND_FILE)  # atomic, so the worker never reads a half-written command
    return {"status": "command_sent"}

@app.get("/")
async def index(): return FileResponse(os.path.join(WEB_DIR, "templates", "index.html"))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True: await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket)

# ---------- shutdown / reboot (debug panel "System" card) ----------
from fastapi import Request
from fastapi.responses import JSONResponse


@app.get("/api/system/power")
def power_info(request: Request):
    return bt_debug.power_info(request.client.host if request.client else "")


@app.post("/api/system/power")
async def power(request: Request):
    try:
        action = (await request.json()).get("action")
    except Exception:
        action = None
    code, out = bt_debug.system_power(action, request.client.host if request.client else "",
                                      request.headers.get(bt_debug.POWER_HEADER))
    return JSONResponse(out, status_code=code)

@app.get("/api/system/update")
def update_info(request: Request, check: int = 0):
    return bt_debug.update_info(request.client.host if request.client else "", fetch=bool(check))


@app.post("/api/system/update")
async def update(request: Request):
    try:
        action = (await request.json()).get("action")
    except Exception:
        action = None
    code, out = await run_in_threadpool(bt_debug.system_update, action, request.client.host if request.client else "",
                                        request.headers.get(bt_debug.POWER_HEADER))
    return JSONResponse(out, status_code=code)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5001, reload=False)
