"""Hydro-dash web API (the app's Water page; mounted by ../server.py at /api/d/water). Totals, charts and history are computed from the
recorded sip log (the single source of truth), so they survive restarts and include
sips the bottle replayed after being out of range."""
import datetime
from contextlib import asynccontextmanager
import json
import os
import re
import threading
import time

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import os as _os, sys as _sys
_sys.path.insert(1, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # shared bt_debug.py, in the release folder
import bt_debug
import water

BASE_DIR = bt_debug.state_dir(os.path.dirname(os.path.abspath(__file__)))
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
                    "cal_min": None, "cal_max": None, "cal_at": None, "cal_pending": None,
                    "reminders": {"enabled": True, "from": "08:00", "to": "22:00", "every_min": 60,
                                  "always": False, "sound": False},
                    "goal_glow": True, "sip_glow": None,
                    # inputs to the goal calculator, kept so it opens as you left it
                    "goal_calc": {"weight": None, "weight_unit": "lb", "reference": "men",
                                  "exercise_min": 0,
                                  "weather": "off",        # "auto" (local forecast), "hot" (by hand), "off"
                                  "place": None,           # {"name", "label", "lat", "lon"} for the forecast
                                  "follow": False,         # set the goal from the suggestion each morning
                                  "followed_on": None},
                    "glow": {"style": "pulse", "color1": "#2bff00", "color2": "#1499ff"}}
GLOW_STYLES = ("pulse", "spin", "flash", "solid", "rainbow")

@asynccontextmanager
async def lifespan(app):
    start_daily_goal()
    yield


app = FastAPI(lifespan=lifespan)


@app.exception_handler(Exception)
async def explain_errors(request, exc):
    """Anything unexpected: say what it was (the page shows it) and log the full traceback."""
    import traceback
    traceback.print_exc()
    tb = traceback.extract_tb(exc.__traceback__)
    where = f" ({os.path.basename(tb[-1].filename)} line {tb[-1].lineno})" if tb else ""
    from fastapi.responses import JSONResponse as _J
    return _J({"detail": f"{type(exc).__name__}: {exc}{where}"}, status_code=500)



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
    goal_calc: dict | None = None          # the goal calculator's inputs (nothing is sent to the bottle)


def clean_goal_calc(d, old):
    c = dict(DEFAULT_SETTINGS["goal_calc"], **(old or {}))
    if "hot" in c:                                  # before the weather choice it was a checkbox
        if c.pop("hot") and c.get("weather") == "off":
            c["weather"] = "hot"
    if "weight" in d:                               # only when sent: a partial update keeps it
        try:
            w = float(d["weight"]) if d["weight"] not in (None, "") else None
            c["weight"] = w if w is None or 20 <= w <= 700 else c["weight"]
        except (TypeError, ValueError):
            pass
    if d.get("weight_unit") in ("lb", "kg"):
        c["weight_unit"] = d["weight_unit"]
    if d.get("reference") in ("men", "women"):
        c["reference"] = d["reference"]
    try:
        c["exercise_min"] = max(0, min(600, int(d.get("exercise_min", c["exercise_min"]) or 0)))
    except (TypeError, ValueError):
        pass
    if d.get("weather") in ("auto", "hot", "off"):
        c["weather"] = d["weather"]
    if "place" in d:
        p = d["place"]
        try:
            lat, lon = float(p["lat"]), float(p["lon"])
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                c["place"] = {"name": str(p.get("name") or "")[:80], "label": str(p.get("label") or p.get("name") or "")[:160],
                              "lat": round(lat, 4), "lon": round(lon, 4)}
        except (TypeError, KeyError, ValueError):
            if p is None:
                c["place"] = None
    if "follow" in d:
        c["follow"] = bool(d["follow"])
        if c["follow"]:
            c["followed_on"] = None                 # turning it on applies today's suggestion
    return c


def goal_calc():
    return clean_goal_calc({}, settings().get("goal_calc"))


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
    if req.goal_calc is not None:
        s["goal_calc"] = clean_goal_calc(req.goal_calc, s.get("goal_calc"))
    _write_json(SETTINGS_FILE, s)
    if resync:
        _command({"action": "sync"})       # picked up if the bottle is connected; otherwise sent at the next connect
    return s


# ---------- goal calculator: suggestion, local weather, the morning update ----------
class SuggestRequest(BaseModel):
    goal_calc: dict | None = None          # inputs being edited (not saved); omitted -> the saved ones


@app.post("/api/goal/suggest")
def goal_suggest(req: SuggestRequest):
    calc = clean_goal_calc(req.goal_calc or {}, settings().get("goal_calc"))
    weather, note = water.weather_for(calc)
    out = water.suggest_goal(calc, weather)
    out["weather_note"] = note
    out["place"] = calc.get("place")
    out["follow"] = calc.get("follow")
    return out


@app.get("/api/goal/places")
def goal_places(q: str = ""):
    try:
        return {"places": water.find_places(q)}
    except Exception as e:
        return {"places": [], "error": f"town search unavailable ({e.__class__.__name__})"}


def apply_daily_goal(now=None):
    """Once each morning, if "update my goal every morning" is on: today's suggestion becomes the
    goal (and goes to the bottle). Returns the new goal, or None."""
    now = now or datetime.datetime.now()
    s = settings()
    calc = clean_goal_calc({}, s.get("goal_calc"))
    today = now.date().isoformat()
    if not water.due_for_daily_update(calc, today, now.hour):
        return None
    weather, note = water.weather_for(calc)
    if note and calc.get("weather") == "auto" and calc.get("place") and now.hour < 12:
        return None                          # forecast not reachable yet: try again later this morning
    goal = water.suggest_goal(calc, weather)["goal"]
    calc["followed_on"] = today
    s["goal_calc"], s["goal_ml"] = calc, goal
    _write_json(SETTINGS_FILE, s)
    _command({"action": "sync"})
    print(f"💧 Goal for {today}: {goal} mL" + (f" ({weather['feels_max_c']:.0f} °C feels-like high)" if weather else ""))
    return goal


def _daily_goal_loop():
    while True:
        try:
            apply_daily_goal()
        except Exception as e:
            print(f"daily goal: {e!r}")
        time.sleep(600)


_daily_goal_started = False


def start_daily_goal():
    """The morning goal update runs in a background thread (started once, by the server's startup)."""
    global _daily_goal_started
    if not _daily_goal_started:
        _daily_goal_started = True
        threading.Thread(target=_daily_goal_loop, name="daily-goal", daemon=True).start()


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
radio_pin = bt_debug.RadioPin(os.path.join(BASE_DIR, "radio_choice.json"), prefer="internal", default="any")   # not pinned unless you pick one
LAST_DEVICES = {"device": bt_debug.LastDevice(os.path.join(BASE_DIR, "last_device.json"))}


class RadioRequest(BaseModel):
    address: str = "any"


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
