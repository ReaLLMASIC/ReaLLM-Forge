"""Hydro-dash web app (port 5004). Totals, charts and history are computed from the
recorded sip log (the single source of truth), so they survive restarts and include
sips the bottle replayed after being out of range."""
from starlette.concurrency import run_in_threadpool
import datetime
import json
import os
import re
import time

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import bt_debug

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs_hydro")
os.makedirs(LOGS_DIR, exist_ok=True)
STATUS_FILE = os.path.join(LOGS_DIR, "status.json")
DEVICES_FILE = os.path.join(LOGS_DIR, "devices.json")
COMMAND_FILE = os.path.join(LOGS_DIR, "command.json")
def _settings_path():
    """Settings live outside the dashboard folder, so replacing the folder with a new release
    keeps the goal, reminders and the bottle's calibration. An old in-folder file is adopted once."""
    d = os.path.join(os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config"), "bio-dash")
    new, old = os.path.join(d, "hydro_settings.json"), os.path.join(BASE_DIR, "hydro_settings.json")
    try:
        os.makedirs(d, exist_ok=True)
        if not os.path.exists(new) and os.path.exists(old):
            with open(old, "rb") as src, open(new, "wb") as dst:
                dst.write(src.read())
    except OSError:
        return old
    return new


SETTINGS_FILE = _settings_path()
DASH = "HidrateSpark"
DEFAULT_SETTINGS = {"capacity_ml": 621, "capacity_auto": True, "goal_ml": 2500, "units": "ml",
                    "cal_min": None, "cal_max": None, "cal_at": None,
                    "reminders": {"enabled": True, "from": "08:00", "to": "22:00", "every_min": 60,
                                  "always": False, "sound": False},
                    "goal_glow": True, "sip_glow": None,
                    "glow": {"style": "pulse", "color1": "#2bff00", "color2": "#1499ff"}}
GLOW_STYLES = ("pulse", "spin", "flash", "solid", "rainbow")

app = FastAPI()
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


def _read_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def _write_json(path, obj):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(path + ".tmp", path)


def settings():
    s = dict(DEFAULT_SETTINGS)
    s.update(_read_json(SETTINGS_FILE, {}))
    return s


def _command(cmd):
    _write_json(COMMAND_FILE, cmd)


# ---------------------------------------------------------------------------
# Reading the recordings
# ---------------------------------------------------------------------------
def _day_start(ts=None):
    d = datetime.datetime.fromtimestamp(ts if ts is not None else time.time())
    return datetime.datetime(d.year, d.month, d.day).timestamp()


def load_sips(since_ts):
    """All recorded sips since `since_ts` for the current bottle, de-duplicated across
    sessions (same volume within 2 s), oldest first: [(ts, ml, source)]."""
    files = bt_debug.latest_session_files(DASH, "sips", same_device=True)
    sips = []
    for path in files:
        # session folders are dated by when the session started; replayed sips can be older,
        # so read every session that could contain sips in range (plus a day of margin)
        folder_day = os.path.basename(os.path.dirname(path))
        try:
            if datetime.datetime.strptime(folder_day, "%Y-%m-%d").timestamp() < since_ts - 86400 * 2:
                continue
        except ValueError:
            pass
        try:
            with open(path) as f:
                next(f, None)
                for line in f:
                    p = line.rstrip("\n").split(",")
                    try:
                        ts, ml = int(p[0]) / 1000, int(float(p[1]))
                    except (ValueError, IndexError):
                        continue
                    if ts >= since_ts:
                        sips.append((ts, ml, p[4] if len(p) > 4 else ""))
        except OSError:
            continue
    sips.sort()
    out = []
    for ts, ml, src in sips:
        if out and any(abs(ts - t) <= 2 and ml == m for t, m, _ in out[-5:]):
            continue
        out.append((ts, ml, src))
    return out


def load_events(since_ts, names):
    out = []
    for path in bt_debug.latest_session_files(DASH, "events", limit=40, same_device=True):
        try:
            with open(path) as f:
                next(f, None)
                for line in f:
                    p = line.rstrip("\n").split(",", 2)
                    if len(p) >= 2 and p[1] in names and int(p[0]) / 1000 >= since_ts:
                        out.append((int(p[0]) / 1000, p[1], p[2] if len(p) > 2 else ""))
        except (OSError, ValueError):
            continue
    return sorted(out)


def load_status_rows(since_ts):
    rows = []
    for path in bt_debug.latest_session_files(DASH, "status", limit=12, same_device=True):
        try:
            with open(path) as f:
                next(f, None)
                for line in f:
                    p = line.rstrip("\n").split(",")
                    try:
                        ts = int(p[0]) / 1000
                    except (ValueError, IndexError):
                        continue
                    if ts >= since_ts:
                        rows.append((ts, p))
        except OSError:
            continue
    return sorted(rows)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(os.path.join(BASE_DIR, "templates", "index.html"))


@app.get("/api/status")
def get_status():
    st = _read_json(STATUS_FILE, {"state": "scanning"})
    st["settings"] = settings()
    return st


@app.get("/api/scan-results")
def scan_results():
    return _read_json(DEVICES_FILE, [])


class ConnectRequest(BaseModel):
    address: str


@app.post("/api/connect")
def connect(req: ConnectRequest):
    _command({"action": "connect", "address": req.address})
    return {"ok": True}


@app.get("/api/summary")
def summary(days: int = 7):
    """Today's intake (total, per hour, cumulative), recent sips, refills, and daily totals."""
    days = max(1, min(days, 90))
    s = settings()
    today = _day_start()
    sips = load_sips(today - (days - 1) * 86400)
    today_sips = [x for x in sips if x[0] >= today]
    hourly = [0] * 24
    cumulative, run = [], 0
    for ts, ml, _ in today_sips:
        hourly[datetime.datetime.fromtimestamp(ts).hour] += ml
        run += ml
        cumulative.append({"x": int(ts * 1000), "y": run})
    daily = []
    for i in range(days - 1, -1, -1):
        start = today - i * 86400
        total = sum(ml for ts, ml, _ in sips if start <= ts < start + 86400)
        daily.append({"date": datetime.date.fromtimestamp(start + 3600).isoformat(), "ml": total})
    refills = load_events(today, {"refill"})
    last = sips[-1] if sips else None
    return {
        "today_ml": run, "goal_ml": s["goal_ml"], "sips_today": len(today_sips),
        "refills_today": len(refills), "last_sip": {"ts": last[0], "ml": last[1]} if last else None,
        "hourly": hourly, "cumulative": cumulative, "daily": daily,
        "recent": [{"ts": ts, "ml": ml, "source": src} for ts, ml, src in reversed(today_sips[-12:])],
    }


@app.get("/api/history")
def history(metric: str, minutes: int = 60):
    """Tile-trend history: fill (mL), battery (%), or intake (cumulative mL today)."""
    minutes = max(1, min(minutes, 60 * 24 * 31))
    since = time.time() - minutes * 60
    if metric in ("fill", "battery"):
        col = 1 if metric == "fill" else 5
        pts = []
        for ts, p in load_status_rows(since):
            try:
                pts.append({"x": int(ts * 1000), "y": float(p[col])})
            except (ValueError, IndexError):
                continue
        return {metric: pts[:: max(1, len(pts) // 1500)]}
    if metric == "intake":
        run, pts = 0, []
        day = None
        for ts, ml, _ in load_sips(since):
            d = _day_start(ts)
            if d != day:
                day, run = d, 0          # restart at each midnight
            run += ml
            pts.append({"x": int(ts * 1000), "y": run})
        return {"intake": pts}
    return {}


class SettingsRequest(BaseModel):
    capacity_ml: int | None = None
    goal_ml: int | None = None
    units: str | None = None
    capacity_auto: bool | None = None
    reminders: dict | None = None
    goal_glow: bool | None = None
    sip_glow: str | None = None            # "on" / "off" / "leave"


@app.get("/api/settings")
def get_settings():
    return settings()


@app.post("/api/settings")
def set_settings(req: SettingsRequest):
    s = settings()
    if req.capacity_auto:                       # back to whatever the bottle reports on its next connect
        s["capacity_auto"] = True
    elif req.capacity_ml and 100 <= req.capacity_ml <= 3000:
        s["capacity_ml"] = req.capacity_ml
        s["capacity_auto"] = False              # you picked a size: stop following the bottle
    if req.goal_ml and 250 <= req.goal_ml <= 8000:
        s["goal_ml"] = req.goal_ml
    if req.units in ("ml", "oz"):
        s["units"] = req.units
    resync = bool(req.goal_ml or req.capacity_ml or req.capacity_auto)
    if req.reminders is not None:
        r = dict(DEFAULT_SETTINGS["reminders"], **(s.get("reminders") or {}))
        for key in ("enabled", "always", "sound"):
            if key in req.reminders:
                r[key] = bool(req.reminders[key])
        for key in ("from", "to"):
            v = str(req.reminders.get(key, ""))
            if re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", v):
                r[key] = v
        try:
            r["every_min"] = max(15, min(240, int(req.reminders.get("every_min", r["every_min"]))))
        except (TypeError, ValueError):
            pass
        s["reminders"], resync = r, True
    if req.goal_glow is not None:
        s["goal_glow"], resync = req.goal_glow, True
    if req.sip_glow in ("on", "off", "leave"):
        s["sip_glow"], resync = {"on": True, "off": False, "leave": None}[req.sip_glow], True
    _write_json(SETTINGS_FILE, s)
    if resync:
        _command({"action": "sync"})       # picked up if the bottle is connected; otherwise sent at the next connect
    return s


class GlowPatternRequest(BaseModel):
    style: str
    color1: str = "#2bff00"
    color2: str = "#1499ff"


@app.post("/api/glow/pattern")
def glow_pattern(req: GlowPatternRequest):
    """Save a glow style + colours and upload it to the bottle (needs the bottle connected)."""
    ok_colour = lambda c: bool(re.fullmatch(r"#[0-9a-fA-F]{6}", c or ""))
    if req.style not in GLOW_STYLES or not ok_colour(req.color1) or not ok_colour(req.color2):
        return JSONResponse({"error": "style or colour not valid"}, status_code=400)
    s = settings()
    s["glow"] = {"style": req.style, "color1": req.color1.lower(), "color2": req.color2.lower()}
    _write_json(SETTINGS_FILE, s)
    _command({"action": "glow_pattern"})
    return {"ok": True, "glow": s["glow"]}


@app.post("/api/glow")
def glow():
    """Play the bottle's glow now (needs the bottle connected)."""
    _command({"action": "glow"})
    return {"ok": True}


class CalibrateRequest(BaseModel):
    which: str


@app.post("/api/calibrate")
def calibrate(req: CalibrateRequest):
    if req.which not in ("full", "empty"):
        return {"error": "which must be full or empty"}
    _command({"action": "calibrate", "which": req.which})
    return {"ok": True}


# ---------- radio pinning + auto-reconnect (same as the other dashboards) ----------
radio_pin = bt_debug.RadioPin(os.path.join(BASE_DIR, "radio_config.json"), prefer="internal")
LAST_DEVICES = {"device": bt_debug.LastDevice(os.path.join(BASE_DIR, "last_device.json"))}


class RadioRequest(BaseModel):
    address: str = "auto"


@app.get("/api/radio")
def get_radio():
    return {"radio": radio_pin.config()}


@app.post("/api/radio")
def set_radio(req: RadioRequest):
    radio_pin.save(req.address)
    return {"radio": radio_pin.config()}


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
    uvicorn.run(app, host="0.0.0.0", port=5004, log_level="warning")
