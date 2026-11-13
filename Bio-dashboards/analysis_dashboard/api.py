"""Bio-dash Analysis (the app's Analysis page, mounted at /api/d/analysis): summaries of what every device recorded, every signal on one
clock, CSV export and the overnight reports. Reads the recordings folder only; it talks to no
devices, so it can run anywhere the recordings are (copied with rsync, say)."""
import datetime as dt
import os
import subprocess
import sys
import multiprocessing
import threading
from concurrent.futures import ProcessPoolExecutor

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response

import analysis_jobs as jobs
import biodata as b

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(BASE_DIR)
NIGHT_REPORT = os.path.join(REPO, "tools", "night_report.py")

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




def root():
    return b.data_root()


# ---------- the heavy work runs in its own process, which exits when Analysis isn't in use ----------
IDLE_S = 60
_pool, _pool_lock, _idle_timer = None, threading.Lock(), None


def _stop_pool():
    global _pool
    with _pool_lock:
        pool, _pool = _pool, None
    if pool:
        pool.shutdown(wait=False, cancel_futures=True)


def run_job(name, *args):
    """jobs.<name>(*args) in the analysis process; JobError becomes the page's error message."""
    global _pool, _idle_timer
    with _pool_lock:
        if _pool is None:
            _pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
        pool = _pool
        if _idle_timer:
            _idle_timer.cancel()
    try:
        return pool.submit(getattr(jobs, name), *args).result()
    except jobs.JobError as e:
        raise HTTPException(e.status, e.message)
    finally:
        with _pool_lock:
            _idle_timer = threading.Timer(IDLE_S, _stop_pool)
            _idle_timer.daemon = True
            _idle_timer.start()


@app.get("/api/days")
def days():
    r = root()
    return {"root": r, "exists": os.path.isdir(r), "days": b.days_with_data(r),
            "families": {f: {"label": b.FAMILIES[f][0], "color": b.FAMILIES[f][1]} for f in b.FAMILY_ORDER},
            "today": dt.date.today().isoformat()}


@app.get("/api/units")
def units(start: str, end: str = ""):
    """The straps and rings recorded in the range (with several people, Analysis shows one of each)."""
    return run_job("units", start, end)


@app.get("/api/summary")
def summary(start: str, end: str = "", units: str = ""):
    return run_job("summary", start, end, units)


@app.get("/api/signals")
def signal_list(start: str, end: str = "", units: str = ""):
    return run_job("signal_list", start, end, units)


@app.get("/api/series")
def series(start: str, end: str = "", ids: str = "", step: str = "auto", max_gap: int = 60, units: str = ""):
    return run_job("series", start, end, ids, step, max_gap, units)


@app.get("/api/export.csv")
def export(start: str, end: str = "", ids: str = "", step: str = "1min", minmax: int = 0, counts: int = 1,
           max_gap: int = 60, units: str = ""):
    text, name = run_job("export", start, end, ids, step, minmax, counts, max_gap, units)
    return Response(text, media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.get("/api/export_mcint.zip")
def export_mcint(start: str, end: str = "", ids: str = "", step: str = "10s", max_gap: int = 60, hold: int = 120,
                 missing: str = "mark", flags: int = 0, units: str = ""):
    data, name = run_job("export_mcint", start, end, ids, step, max_gap, hold, missing, flags, units)
    return Response(data, media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="{name}"'})


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
