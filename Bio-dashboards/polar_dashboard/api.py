import asyncio
import os
import json
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel

import os as _os, sys as _sys
_sys.path.insert(1, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # shared bt_debug.py, in the release folder
import bt_debug

BASE_DIR = bt_debug.state_dir(os.path.dirname(os.path.abspath(__file__)))   # one more per extra person: see bt_debug.state_dir
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
# (bt_debug.SessionFiles). This API follows the session its own worker is recording (the worker
# lists the files in its status), else the last one of the strap it streamed from.
_STREAM_OF = {"polar_ecg": "ecg", "polar_acc": "acc", "polar_ppi": "rr"}

def get_latest_file(prefix):
    return bt_debug.current_session_file(get_status(), "Polar H10", _STREAM_OF[prefix], LAST_DEVICES["device"].load())

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
            if not manager.active_connections:
                # nobody is watching: skip to the end now and then instead of reading 50x a second
                f.seek(0, os.SEEK_END)
                partial = ""
                await asyncio.sleep(0.5)
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

radio_pin = bt_debug.RadioPin(os.path.join(BASE_DIR, "radio_config.json"))

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

def _command(cmd):
    os.makedirs(LOGS_DIR, exist_ok=True)
    tmp = COMMAND_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cmd, f)
    os.replace(tmp, COMMAND_FILE)  # atomic, so the worker never reads a half-written command


@app.post("/api/connect")
def connect_device(req: ConnectRequest):
    _command({"action": "connect", "address": req.address})
    return {"status": "command_sent"}


@app.post("/api/disconnect")
def disconnect_device():
    """Ends the session with the strap; auto-reconnect waits until a device is picked (the finder lists them)."""
    _command({"action": "disconnect"})
    return {"status": "command_sent"}


@app.get("/api/session")
def session_files():
    """The recording being written now, per stream (forecast_service.py follows these)."""
    return {s: get_latest_file(p) for p, s in _STREAM_OF.items()}


# ---------- stream-lane-lab's API (ReaLLM-Forge PR #924) for the H10 models ----------
# GET /api/models and POST /api/forecast (history in real units on the 8 ms grid -> predictions), answered by
# forecast_service.py (it holds the models) on 127.0.0.1:BIODASH_FORECAST_API_PORT and passed on from here.
# The Lab page (web/lab.html) is PR #924's viewer, with the live strap as a source.
FORECAST_API = f"http://127.0.0.1:{os.environ.get('BIODASH_FORECAST_API_PORT', '5011')}"


def _service_call(path, body=None, timeout=60):
    import urllib.error
    import urllib.request
    req = urllib.request.Request(FORECAST_API + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="GET" if body is None else "POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"null")
        except ValueError:
            return e.code, {"detail": str(e)}
    except (urllib.error.URLError, OSError, ValueError) as e:
        return 503, {"detail": f"The forecast service isn't answering ({e}). It starts with Bio-dash "
                               "when PyTorch and the models are installed (see forecast/README.md)."}


@app.get("/api/models")
async def forecast_models():
    code, body = await run_in_threadpool(_service_call, "/models", None, 10)
    return JSONResponse(body, status_code=code)


@app.post("/api/forecast")
async def forecast_api(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"detail": "bad JSON"}, status_code=422)
    code, out = await run_in_threadpool(_service_call, "/forecast", body)
    return JSONResponse(out, status_code=code)


@app.get("/api/signal")
async def forecast_signal(request: Request):
    """PR #924's demo source (summed sine waves), for the Lab viewer."""
    code, body = await run_in_threadpool(_service_call, "/signal?" + request.url.query, None, 10)
    return JSONResponse(body, status_code=code)


@app.get("/api/lab/rows")
async def lab_rows(request: Request):
    """The strap's newest rows on the model's 8 ms grid (mV / mg), for the Lab viewer's live source."""
    code, body = await run_in_threadpool(_service_call, "/rows?" + request.url.query, None, 5)
    return JSONResponse(body, status_code=code)


# ---------- live forecasts (forecast_service.py) ----------
# server.py starts forecast_service.py when the models are there. It tails the same session files and posts
# its forecasts here; they reach the Heart page as ecg_forecast / acc_forecast / rr_forecast messages on the
# heart stream (overlays: they don't count as live data). The page's Forecast button switches them (saved in
# forecast_settings.json); the service reads the switches with its heartbeat and only computes what is on.
FORECAST_SETTINGS_FILE = os.path.join(BASE_DIR, "forecast_settings.json")
FORECAST_DEFAULTS = {"ecg": False, "acc": False, "rr": False, "horizon_s": 1.0, "refresh_s": 0.5, "samples": 8}
FORECAST_KINDS = ("ecg", "acc", "rr")
FORECAST_LIMITS = {"horizon_s": (0.25, 1.0), "refresh_s": (0.25, 5.0), "samples": (1, 32)}


def _load_forecast_settings():
    s = dict(FORECAST_DEFAULTS)
    try:
        with open(FORECAST_SETTINGS_FILE) as f:
            s.update({k: v for k, v in json.load(f).items() if k in FORECAST_DEFAULTS})
    except (OSError, ValueError, AttributeError):
        pass
    return s


forecast_settings = _load_forecast_settings()
forecast_service = {"last_seen": 0.0, "status": {}}


def _forecast_local(request):
    """The service runs on this computer: forecasts / heartbeats from elsewhere are refused
    (BIODASH_FORECAST_REMOTE=1 allows them, e.g. a service on another machine)."""
    host = request.client.host if request.client else ""
    return host in ("127.0.0.1", "::1", "localhost") or os.environ.get("BIODASH_FORECAST_REMOTE") == "1"


def forecast_view():
    seen = forecast_service["last_seen"]
    age = time.time() - seen
    return {"settings": forecast_settings,
            "service": {**forecast_service["status"], "online": bool(seen) and age < 4.0,
                        "age_s": round(age, 1) if seen else None}}


class ForecastSettingsRequest(BaseModel):
    ecg: bool | None = None
    acc: bool | None = None
    rr: bool | None = None
    horizon_s: float | None = None
    refresh_s: float | None = None
    samples: int | None = None


@app.get("/api/forecast/settings")
def get_forecast_settings():
    return forecast_view()


@app.post("/api/forecast/settings")
async def set_forecast_settings(req: ForecastSettingsRequest):
    dump = getattr(req, "model_dump", None) or req.dict          # pydantic 2 / 1
    for k, v in dump(exclude_none=True).items():
        if k in FORECAST_LIMITS:
            lo, hi = FORECAST_LIMITS[k]
            v = type(FORECAST_DEFAULTS[k])(min(hi, max(lo, v)))
        forecast_settings[k] = v
    try:
        tmp = FORECAST_SETTINGS_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(forecast_settings, f)
        os.replace(tmp, FORECAST_SETTINGS_FILE)
    except OSError:
        pass
    view = forecast_view()
    await manager.broadcast(json.dumps({"type": "forecast_settings", **view}))
    return view


@app.post("/api/forecast/heartbeat")
async def forecast_heartbeat(request: Request):
    """From forecast_service.py once a second: its state; the answer is the overlay switches."""
    if not _forecast_local(request):
        return JSONResponse({"error": "the forecast service must run on this computer"}, status_code=403)
    try:
        status = await request.json()
    except Exception:
        status = {}
    forecast_service["last_seen"] = time.time()
    forecast_service["status"] = status if isinstance(status, dict) else {}
    return forecast_settings


@app.post("/api/forecast/live")
async def post_forecast(request: Request):
    """From forecast_service.py: one forecast (band, median, representative sample) -> the pages."""
    if not _forecast_local(request):
        return JSONResponse({"error": "the forecast service must run on this computer"}, status_code=403)
    try:
        fc = await request.json()
    except Exception:
        return JSONResponse({"error": "bad JSON"}, status_code=400)
    kind = fc.get("kind") if isinstance(fc, dict) else None
    if kind not in FORECAST_KINDS:
        return JSONResponse({"error": "kind must be ecg, acc or rr"}, status_code=400)
    if not forecast_settings.get(kind):
        return {"ok": False, "reason": f"the {kind} overlay is off"}
    fc["type"] = f"{kind}_forecast"
    if manager.active_connections:
        await manager.broadcast(json.dumps(fc, separators=(",", ":")))
    return {"ok": True}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True: await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket)
