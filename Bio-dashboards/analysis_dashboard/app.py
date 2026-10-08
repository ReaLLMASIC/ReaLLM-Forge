"""Bio-dash Analysis (port 5005): summaries of what every device recorded, every signal on one
clock, CSV export and the overnight reports. Reads the recordings folder only; it talks to no
devices, so it can run anywhere the recordings are (copied with rsync, say)."""
from starlette.concurrency import run_in_threadpool
import datetime as dt
import io
import os
import subprocess
import sys
import threading
import time

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

import biodata as b
import bt_debug

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(BASE_DIR)
NIGHT_REPORT = os.path.join(REPO, "tools", "night_report.py")
MAX_DAYS = 92
MAX_CHART_POINTS = 20_000
MAX_EXPORT_CELLS = 40_000_000

app = FastAPI()


@app.exception_handler(Exception)
async def explain_errors(request: Request, exc: Exception):
    """Anything unexpected: say what it was (the page shows it) and log the full traceback."""
    import traceback
    traceback.print_exc()
    where = ""
    tb = traceback.extract_tb(exc.__traceback__)
    if tb:
        where = f" ({os.path.basename(tb[-1].filename)} line {tb[-1].lineno})"
    return JSONResponse({"detail": f"{type(exc).__name__}: {exc}{where}"}, status_code=500)


app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


def root():
    return b.data_root()


def day_range(start, end):
    """Validated (start_ms, end_ms, start_day, end_day) from 'YYYY-MM-DD' strings."""
    try:
        s = dt.date.fromisoformat(start)
        e = dt.date.fromisoformat(end or start)
    except (TypeError, ValueError):
        raise HTTPException(400, "Dates must look like 2026-10-07.")
    if e < s:
        s, e = e, s
    if (e - s).days + 1 > MAX_DAYS:
        raise HTTPException(400, f"Pick at most {MAX_DAYS} days.")
    return b.local_ms(s.isoformat()), b.local_ms(e.isoformat(), end=True), s.isoformat(), e.isoformat()


# signals for a range, kept briefly so the page's several requests read the files once
_sig_cache, _sig_lock = {}, threading.Lock()


def signals_for(start_ms, end_ms):
    key = (root(), start_ms, end_ms)
    with _sig_lock:
        hit = _sig_cache.get(key)
        if hit and time.time() - hit[0] < 30:
            return hit[1]
    sigs = b.load_signals(key[0], start_ms, end_ms)
    with _sig_lock:
        _sig_cache.clear() if len(_sig_cache) > 6 else None
        _sig_cache[key] = (time.time(), sigs)
    return sigs


@app.get("/")
def index():
    return FileResponse(os.path.join(BASE_DIR, "templates", "index.html"))


@app.get("/api/days")
def days():
    r = root()
    return {"root": r, "exists": os.path.isdir(r), "days": b.days_with_data(r),
            "families": {f: {"label": b.FAMILIES[f][0], "color": b.FAMILIES[f][1]} for f in b.FAMILY_ORDER},
            "today": dt.date.today().isoformat()}


def nights_in(summary, start_day, end_day):
    """Evening dates (18:00 -> 12:00 next day) in or touching the range that have SpO2 readings."""
    segs = summary["timeline"].get("oxygen", [])
    out = []
    first = dt.date.fromisoformat(start_day) - dt.timedelta(days=1)
    last = dt.date.fromisoformat(end_day)
    d = first
    while d <= last:
        a = int(dt.datetime.combine(d, dt.time(18)).timestamp() * 1000)
        z = a + 18 * 3_600_000
        mins = sum(max(0, min(y, z) - max(x, a)) for x, y in segs) / 60000
        if mins >= 30:
            out.append({"night": d.isoformat(), "minutes": round(mins)})
        d += dt.timedelta(days=1)
    return out


@app.get("/api/summary")
def summary(start: str, end: str = ""):
    s, e, sd, ed = day_range(start, end)
    sigs = signals_for(s, e)
    out = b.summarise(root(), s, e, sigs)
    out["nights"] = nights_in(out, sd, ed)
    out["days"] = b.days_between(sd, ed)
    return out


@app.get("/api/signals")
def signal_list(start: str, end: str = ""):
    s, e, _, _ = day_range(start, end)
    sigs = signals_for(s, e)
    order = {f: i for i, f in enumerate(b.FAMILY_ORDER)}
    metas = sorted((x.meta() for x in sigs.values()),
                   key=lambda m: (m["id"].startswith("raw."), order.get(m["family"], 9), m["id"]))
    return {"signals": metas, "steps": b.STEPS_MS}


def _ids_and_step(sigs, ids, step, span_ms, max_points):
    wanted = [i for i in (ids or "").split(",") if i]
    missing = [i for i in wanted if i not in sigs]
    chosen = {i: sigs[i] for i in wanted if i in sigs}
    try:
        st = b.parse_step(step)
    except ValueError as ex:
        raise HTTPException(400, str(ex))
    st = st or b.auto_step(span_ms)
    if span_ms // st > max_points:
        raise HTTPException(400, f"That step gives {span_ms // st:,} points per signal over this range; "
                                 f"pick a larger step or fewer days.")
    return chosen, missing, st


def _r(v, nd=2):
    return None if v is None else round(v, nd)


@app.get("/api/series")
def series(start: str, end: str = "", ids: str = "", step: str = "auto", max_gap: int = 60):
    s, e, _, _ = day_range(start, end)
    sigs = signals_for(s, e)
    chosen, missing, st = _ids_and_step(sigs, ids, step, e - s + 1, MAX_CHART_POINTS)
    tab = b.table(chosen, s, e, step=st, max_gap=max_gap * 1000)
    return {"step": tab["step"], "first": tab["first"], "n": tab["n"], "missing": missing,
            "meta": tab["meta"],
            "series": {sid: {"mean": [_r(v) for v in r["mean"]],
                             "min": [_r(v) for v in r["min"]] if tab["meta"][sid]["kind"] == "readings" else None,
                             "max": [_r(v) for v in r["max"]] if tab["meta"][sid]["kind"] == "readings" else None,
                             "n": r["n"]}
                       for sid, r in tab["series"].items()}}


@app.get("/api/export.csv")
def export(start: str, end: str = "", ids: str = "", step: str = "1min", minmax: int = 0, counts: int = 1,
           max_gap: int = 60):
    s, e, sd, ed = day_range(start, end)
    sigs = signals_for(s, e)
    chosen, _, st = _ids_and_step(sigs, ids, step, e - s + 1, 10_000_000)
    if not chosen:
        raise HTTPException(400, "Pick at least one signal.")
    if (e - s) // st * len(chosen) > MAX_EXPORT_CELLS:
        raise HTTPException(400, "That export would be too large; pick a larger step, fewer days or fewer signals.")
    tab = b.table(chosen, s, e, step=st, max_gap=max_gap * 1000)
    buf = io.StringIO()
    b.write_csv(buf, tab, minmax=bool(minmax), counts=bool(counts))
    step_txt = f"{st // 60000}min" if st % 60000 == 0 else f"{st / 1000:g}s"
    name = f"bio-dash_{sd}" + (f"_to_{ed}" if ed != sd else "") + f"_{step_txt}.csv"
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.get("/night/{night}")
def night(night: str):
    try:
        d = dt.date.fromisoformat(night)
    except ValueError:
        raise HTTPException(400, "Nights are named by their evening, e.g. 2026-10-06.")
    out = os.path.join(root(), "Reports", f"night_{d.isoformat()}.html")
    p = subprocess.run([sys.executable, NIGHT_REPORT, d.isoformat(), "--data-dir", root(), "--out", out],
                       capture_output=True, text=True, timeout=120)
    if p.returncode != 0 or not os.path.isfile(out):
        msg = (p.stderr or p.stdout or "the report couldn't be made").strip().splitlines()[-1]
        return Response(f"<!doctype html><meta charset=utf-8><body style='font:16px system-ui;background:#0f172a;"
                        f"color:#e2e8f0;padding:2rem'><p>No report for the night of {d}: {msg}</p>",
                        media_type="text/html", status_code=404)
    return FileResponse(out, media_type="text/html")


# ---------- sidebar: updates, shutdown / reboot (same as every dashboard) ----------
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
    print(f"📂 Reading recordings from {root()}")
    uvicorn.run(app, host="0.0.0.0", port=5005, log_level="warning")
