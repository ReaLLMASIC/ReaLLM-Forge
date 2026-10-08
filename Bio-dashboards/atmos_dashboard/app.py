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

from atmos_parser import normalize_name, PROFILES
import bt_debug

WEB_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs_atmos")
DEVICES_FILE = os.path.join(LOGS_DIR, "devices.json")
COMMAND_FILE = os.path.join(LOGS_DIR, "command.json")
STATUS_FILE = os.path.join(LOGS_DIR, "status.json")
PROFILE_FILE = os.path.join(BASE_DIR, "atmos_profile.json")

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

POLL_S = 0.1
ROTATE_CHECK_S = 2.0

def log_files():
    """This device's recordings, oldest first: <Documents>/Bio-dash/Atmos/<device>/<date>/
    <time>[_partN]_readings.csv. Only the newest file's device, so switching units doesn't
    mix their histories."""
    return list(reversed(bt_debug.latest_session_files("Atmos", "readings", limit=12, same_device=True)))

def get_latest_file():
    files = log_files()
    return files[-1] if files else None

def _columns(header_line):
    """CSV header -> series ids. Column 0 is the timestamp."""
    names = [h.strip() for h in header_line.strip().split(",")]
    return [normalize_name(n) or n for n in names[1:]]

def _row(columns, line):
    p = line.strip().split(",")
    if len(p) < 2:
        return None
    try:
        ts = int(float(p[0]))
    except ValueError:
        return None
    vals = {}
    for c, v in zip(columns, p[1:]):
        if v == "":
            continue
        try:
            x = float(v)
        except ValueError:
            continue
        if x == x:                     # skip NaN
            vals[c] = x
    return {"ts": ts, "v": vals} if vals else None

async def tail_file_and_broadcast():
    """Follows the newest log; each file's header names its columns. Sends
    {"rows": [{"ts": ms, "v": {series_id: value}}]} to the browser."""
    loop = asyncio.get_running_loop()
    current_file, f, partial, columns = None, None, "", None
    next_rotate_check = 0.0
    try:
        while True:
            now = loop.time()
            if now >= next_rotate_check:
                next_rotate_check = now + ROTATE_CHECK_S
                latest_file = get_latest_file()
                if latest_file and latest_file != current_file:
                    if f: f.close()
                    current_file, partial, columns = latest_file, "", None
                    print(f"📡 Router locked onto: {current_file}")
                    f = open(current_file, 'r')
                    columns = _columns(f.readline() or "ts")
                    if os.path.getsize(current_file) > 64 * 1024:
                        f.seek(0, os.SEEK_END)     # don't replay a long file; /api/history covers it

            if f is None:
                await asyncio.sleep(1)
                continue

            chunk = f.read()
            if not chunk:
                await asyncio.sleep(POLL_S)
                continue

            lines = (partial + chunk).split("\n")
            partial = lines.pop()
            rows = [r for r in (_row(columns, l) for l in lines if l.strip()) if r]
            if rows and manager.active_connections:
                await manager.broadcast(json.dumps({"rows": rows}, separators=(",", ":")))
    finally:
        if f: f.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.path.exists(COMMAND_FILE):
        try: os.remove(COMMAND_FILE)
        except Exception: pass
        
    tail_task = asyncio.create_task(tail_file_and_broadcast())
    yield
    tail_task.cancel()

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=os.path.join(WEB_DIR, "static")), name="static")

# --- ENDPOINTS ---
@app.get("/api/scan-results")
def get_scan_results():
    if os.path.exists(DEVICES_FILE):
        try:
            with open(DEVICES_FILE, 'r') as f: 
                return json.load(f)
        except Exception: pass
    return []

@app.get("/api/status")
def get_status():
    """Worker state (scanning / connecting / streaming) plus debug telemetry."""
    try:
        with open(STATUS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {"state": "scanning"}

HISTORY_MAX_POINTS = 1500

@app.get("/api/history")
def get_history(minutes: int = 15):
    """Recent rows from the newest log file(s), whatever their columns:
    {"rows": [{"ts": ms, "v": {series_id: value}}, ...]}."""
    minutes = max(1, min(minutes, 24 * 60))
    cutoff = time.time() * 1000 - minutes * 60_000
    rows = []
    for path in reversed(log_files()[-6:]):            # newest first; stop once the window is covered
        try:
            with open(path, "rb") as fh:
                header = fh.readline().decode("utf-8", errors="replace")
                columns = _columns(header)
                start = fh.tell()
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                want = minutes * 60 * 400 + 4096            # rows are ~60-300 bytes at ~1 Hz
                fh.seek(max(start, size - want))
                chunk = fh.read().decode("utf-8", errors="replace")
        except OSError:
            continue
        lines = chunk.splitlines()
        if size - want > start:
            lines = lines[1:]                               # first line probably cut
        file_rows = [r for r in (_row(columns, l) for l in lines) if r and r["ts"] >= cutoff]
        rows = file_rows + rows
        if file_rows and file_rows[0]["ts"] <= cutoff + 60_000:
            break
    step = max(1, len(rows) // HISTORY_MAX_POINTS)
    return {"rows": rows[::step]}

class ProfileRequest(BaseModel):
    profile: str = "auto"
    columns: list[str] | None = None

@app.get("/api/profile")
def get_profile():
    try:
        with open(PROFILE_FILE) as fh:
            cfg = json.load(fh)
    except Exception:
        cfg = {"profile": "auto", "columns": None}
    return {"config": cfg, "profiles": {k: {"label": v["label"], "sensors": v["sensors"], "columns": v["columns"],
                                            "order_confirmed": v["order_confirmed"]} for k, v in PROFILES.items()}}

@app.post("/api/profile")
def set_profile(req: ProfileRequest):
    """Saved next to the dashboard; the worker applies it within a second (no reconnect)."""
    if req.profile != "auto" and req.profile not in PROFILES:
        return {"error": f"unknown profile {req.profile}"}
    cols = [c.strip() for c in (req.columns or []) if c.strip()] or None
    tmp = PROFILE_FILE + ".tmp"
    with open(tmp, "w") as fh:
        json.dump({"profile": req.profile, "columns": cols}, fh, indent=2)
    os.replace(tmp, PROFILE_FILE)
    return get_profile()

@app.post("/api/connect")
def connect_device(req: ConnectRequest):
    os.makedirs(LOGS_DIR, exist_ok=True)
    tmp = COMMAND_FILE + ".tmp"
    with open(tmp, 'w') as f:
        json.dump({"action": "connect", "address": req.address}, f)
    os.replace(tmp, COMMAND_FILE)
    return {"status": "command_sent"}

TCP_TARGET_FILE = os.path.join(BASE_DIR, "atmos_tcp.json")   # last Wi-Fi target, to pre-fill the form

class TcpRequest(BaseModel):
    host: str
    port: int = 8080

@app.get("/api/tcp-target")
def get_tcp_target():
    try:
        with open(TCP_TARGET_FILE) as f:
            return json.load(f)
    except Exception:
        return {"host": "", "port": 8080}

@app.post("/api/connect-tcp")
def connect_tcp(req: TcpRequest):
    """Connect over Wi-Fi to a device streaming CSV on a TCP port (the S4 uses 8080)."""
    host = req.host.strip()
    if not host or not (1 <= req.port <= 65535):
        return {"error": "need a host and a port between 1 and 65535"}
    os.makedirs(LOGS_DIR, exist_ok=True)
    with open(TCP_TARGET_FILE, "w") as f:
        json.dump({"host": host, "port": req.port}, f)
    tmp = COMMAND_FILE + ".tmp"
    with open(tmp, 'w') as f:
        json.dump({"action": "connect_tcp", "host": host, "port": req.port}, f)
    os.replace(tmp, COMMAND_FILE)
    return {"status": "command_sent"}

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
    """Pin this dashboard to a radio by adapter address ("auto" to unpin); applies on the next scan."""
    radio_pin.save(req.address)
    return {"radio": radio_pin.config()}

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
    uvicorn.run("app:app", host="0.0.0.0", port=5002, reload=False)
