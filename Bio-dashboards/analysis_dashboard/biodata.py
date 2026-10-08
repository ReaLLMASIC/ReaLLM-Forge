"""Bio-dash recordings, read back for analysis (standard library only).

Every dashboard records CSV files into <Documents>/Bio-dash/<dashboard>/<device>/<YYYY-MM-DD>/
<HH-MM-SS>_<stream>.csv. This module finds them, turns them into named signals, puts any
set of signals on one clock, and summarises a day or a range of days.

Putting signals on one clock (resample) treats each signal according to what it is:

    readings   heart rate, SpO2, CO2 ...  Each step holds the mean / min / max of the real
                                          readings inside it. A step with no reading is filled
                                          by a straight line between the readings either side,
                                          unless they are more than `max_gap` apart.
    state      bottle fill level, battery Each step holds the last known value, carried
                                          forward (a straight line would invent a slow change
                                          that never happened), for up to `hold_max`.
    events     sips                       Each step holds the sum of the events inside it
                                          (mL drunk in that step); empty steps are 0 while
                                          the device was recording.

Every step also carries `n`, the number of real readings in it: 0 means the value is an
estimate (interpolated or carried forward).
"""
import bisect
import csv
import datetime as dt
import glob
import json
import math
import os
import re
import statistics
import threading

FILE_RE = re.compile(r"^(\d\d)-(\d\d)-(\d\d)(?:_part\d+)?_(.+)\.csv$")
DAY_RE = re.compile(r"^\d{4}-\d\d-\d\d$")
BIG_STREAMS = {"ecg", "acc"}            # waveforms: tens of samples a second, read only on request
SKIP_STREAMS = {"raw", "events"}        # one row per event / packet, not a signal (events: refills are read separately)
COVERAGE_GAP_MS = 120_000               # readings further apart than this = the device wasn't recording
DAY_MS = 86_400_000

# ---------------------------------------------------------------------------------- devices
# family: (label, colour). Colours: a validated categorical set for the dark surface.
FAMILIES = {
    "heart":  ("Heart (Polar H10)", "#3987e5"),
    "oxygen": ("Oxygen (O2 ring)", "#d95926"),
    "air":    ("Air (Atmos)", "#199e70"),
    "water":  ("Water (HidrateSpark)", "#c98500"),
    "other":  ("Other", "#9085e9"),
}
FAMILY_ORDER = ["heart", "oxygen", "air", "water", "other"]


def family_of(dashboard, stream):
    d = (dashboard or "").lower()
    if d.startswith("polar"):
        return "heart"
    if d.startswith("viatom"):
        return "oxygen"
    if d.startswith("omni"):
        return "oxygen" if stream == "vitals" else "heart"
    if d.startswith("atmos"):
        return "air"
    if d.startswith("hidrate"):
        return "water"
    return "other"


AIR_META = {   # metric: (label, unit)
    "co2": ("CO₂", "ppm"), "pm1": ("PM1", "µg/m³"), "pm25": ("PM2.5", "µg/m³"), "pm4": ("PM4", "µg/m³"),
    "pm10": ("PM10", "µg/m³"), "temp": ("Temperature", "°C"), "rh": ("Humidity", "%"),
    "voc": ("VOC index", ""), "nox": ("NOx index", ""), "hcho": ("Formaldehyde", "ppb"),
}
# Curated signals: the ones most people want, merged across dashboards (the same strap recorded
# by Polar-dash in the day and Omni at night is one heart rate).
CURATED = {
    "heart.hr":     ("heart", "Heart rate", "bpm", "readings", [("rr", "HR_BPM"), ("hrv", "HR_BPM")], (25, 230)),
    "heart.rr":     ("heart", "R-R interval", "ms", "readings", [("rr", "RR_ms")], (300, 2000)),
    "oxygen.spo2":  ("oxygen", "SpO₂", "%", "readings", [("vitals", "SpO2_pct")], (50, 100)),
    "oxygen.pulse": ("oxygen", "Pulse (O2 ring)", "bpm", "readings", [("vitals", "HR_BPM")], (25, 250)),
    "water.fill":   ("water", "In the bottle", "mL", "state", [("status", "Fill_ml")], (0, 5000)),
    "water.sips":   ("water", "Drunk", "mL", "events", [("sips", "Volume_ml")], (1, 2000)),
}
STATE_COLUMNS = {"Fill_ml", "Fill_pct", "Battery_pct", "Weight_stable_raw"}
UNITS = {"bpm": "bpm", "ms": "ms", "pct": "%", "ml": "mL", "mv": "mV", "mg": "mg"}


# ---------------------------------------------------------------------------------- folders
def documents_dir():
    try:
        with open(os.path.join(os.path.expanduser("~"), ".config", "user-dirs.dirs")) as f:
            for line in f:
                if line.startswith("XDG_DOCUMENTS_DIR="):
                    path = line.split("=", 1)[1].strip().strip('"').replace("$HOME", os.path.expanduser("~"))
                    if path:
                        return path
    except OSError:
        pass
    return os.path.join(os.path.expanduser("~"), "Documents")


def data_root(arg=None):
    return arg or os.environ.get("BIODASH_DATA_DIR") or os.path.join(documents_dir(), "Bio-dash")


def local_ms(day, end=False):
    """Local midnight of a 'YYYY-MM-DD' day in epoch ms (end=True: the last ms of that day)."""
    d0 = dt.datetime.strptime(day, "%Y-%m-%d")
    t = int(d0.timestamp() * 1000)
    if end:
        t = int((d0 + dt.timedelta(days=1)).timestamp() * 1000) - 1
    return t


def day_of(ms):
    return dt.datetime.fromtimestamp(ms / 1000).strftime("%Y-%m-%d")


def days_between(start_day, end_day):
    d, out = dt.date.fromisoformat(start_day), []
    last = dt.date.fromisoformat(end_day)
    while d <= last and len(out) < 400:
        out.append(d.isoformat())
        d += dt.timedelta(days=1)
    return out


class Recording:
    __slots__ = ("path", "dashboard", "device", "day", "stream", "family")

    def __init__(self, path, dashboard, device, day, stream):
        self.path, self.dashboard, self.device, self.day, self.stream = path, dashboard, device, day, stream
        self.family = family_of(dashboard, stream)


def recordings(root, start_ms=None, end_ms=None, include=()):
    """Recording files whose day folder can overlap [start_ms, end_ms] (a session that started
    the day before may run past midnight)."""
    out = []
    for path in glob.glob(os.path.join(root, "*", "*", "*", "*.csv")):
        parts = path.split(os.sep)
        dashboard, device, day, name = parts[-4], parts[-3], parts[-2], parts[-1]
        if dashboard in ("Exports", "Reports") or not DAY_RE.match(day):
            continue
        m = FILE_RE.match(name)
        if not m:
            continue
        stream = m.group(4)
        if stream in SKIP_STREAMS or (stream in BIG_STREAMS and stream not in include):
            continue
        if start_ms is not None:
            try:
                lo = local_ms(day)
            except ValueError:                   # a folder named like a date that isn't one
                continue
            if lo > end_ms or lo + 2 * DAY_MS < start_ms:
                continue
        out.append(Recording(path, dashboard, device, day, stream))
    out.sort(key=lambda r: r.path)
    return out


def days_with_data(root):
    """{day: [families]} for every day folder that holds recordings."""
    days = {}
    for r in recordings(root):
        days.setdefault(r.day, set()).add(r.family)
    return {d: [f for f in FAMILY_ORDER if f in fams] for d, fams in sorted(days.items())}


# ---------------------------------------------------------------------------------- reading
_cache = {}
_cache_lock = threading.Lock()


def to_ms(raw):
    t = float(raw)
    return int(round(t if t > 1e11 else t * 1000))


def clean_lines(f):
    """Lines of a recording with NUL bytes removed. A power cut or crash while a CSV is being
    written can leave a run of zero bytes in it; the readings either side are still good."""
    for line in f:
        yield line.replace("\0", "") if "\0" in line else line


def read_columns(path):
    """{column: [(t_ms, value)]} for every numeric column of one recording (cached by mtime)."""
    try:
        st = os.stat(path)
    except OSError:
        return {}
    key = (path, st.st_mtime_ns, st.st_size)
    with _cache_lock:
        hit = _cache.get(path)
        if hit and hit[0] == key:
            return hit[1]
    cols = {}
    try:
        with open(path, newline="", errors="replace") as f:
            rd = csv.reader(clean_lines(f))
            head = next(rd, None)
            if head and len(head) >= 2:
                names = [h.strip() for h in head[1:]]
                lists = [[] for _ in names]
                for row in rd:
                    if len(row) < 2:
                        continue
                    try:
                        t = to_ms(row[0])
                    except ValueError:
                        continue
                    for i, cell in enumerate(row[1:len(names) + 1]):
                        if not cell:
                            continue
                        try:
                            v = float(cell)
                        except ValueError:
                            continue
                        if math.isfinite(v):
                            lists[i].append((t, v))
                cols = {n: l for n, l in zip(names, lists) if l}
    except (OSError, csv.Error):
        cols = {}
    with _cache_lock:
        _cache[path] = (key, cols)
    return cols


def clip(points, start_ms, end_ms):
    if not points:
        return points
    ts = [p[0] for p in points]
    return points[bisect.bisect_left(ts, start_ms):bisect.bisect_right(ts, end_ms)]


def merge(lists):
    """Sorted union of several [(t, v)] lists; one value per timestamp (the first seen)."""
    seen, out = set(), []
    for t, v in sorted((p for l in lists for p in l), key=lambda p: p[0]):
        if t not in seen:
            seen.add(t)
            out.append((t, v))
    return out


def dedupe_sips(points, window_ms=2000, tol_ml=10):
    """Drop a sip already recorded: a replayed record lands within 2 s with about the same volume."""
    out = []
    for t, v in points:
        if any(abs(t - t2) <= window_ms and abs(v - v2) <= tol_ml for t2, v2 in out[-6:]):
            continue
        out.append((t, v))
    return out


def clean_name(s):
    return re.sub(r"[^\w\-@.]+", "_", str(s)).strip("_")


# ---------------------------------------------------------------------------------- signals
class Signal:
    __slots__ = ("id", "family", "label", "unit", "kind", "points", "device")

    def __init__(self, id, family, label, unit, kind, points, device=None):
        self.id, self.family, self.label, self.unit, self.kind = id, family, label, unit, kind
        self.points, self.device = points, device

    def meta(self):
        return {"id": self.id, "family": self.family, "label": self.label, "unit": self.unit,
                "kind": self.kind, "readings": len(self.points), "device": self.device,
                "color": FAMILIES[self.family][1]}


def _unit_of(column):
    tail = column.rsplit("_", 1)[-1].lower() if "_" in column else ""
    return UNITS.get(tail, "")


def load_signals(root, start_ms, end_ms, include=(), extras=True):
    """{id: Signal} in the range: the curated ones, Atmos metrics, and (extras) every other
    numeric column as 'raw.<dashboard>.<stream>.<column>'."""
    recs = recordings(root, start_ms, end_ms, include)
    by_key = {}                    # (family, stream, column) -> [(device, points)]
    raw_cols = {}                  # (dashboard, stream, column) -> [(device, points)]
    hr_from_rr = []                # heart rate for recordings that only have R-R intervals (Omni)
    for r in recs:
        cols = read_columns(r.path)
        if r.family == "heart" and r.stream == "rr" and "HR_BPM" not in cols and "RR_ms" in cols:
            hr_from_rr.append([(t, round(60000 / v, 1)) for t, v in clip(cols["RR_ms"], start_ms, end_ms)
                               if 300 <= v <= 2000])
        for col, pts in cols.items():
            pts = clip(pts, start_ms, end_ms)
            if not pts:
                continue
            by_key.setdefault((r.family, r.stream, col), []).append((r.device, pts))
            raw_cols.setdefault((r.dashboard, r.stream, col), []).append((r.device, pts))
    out, used = {}, set()
    for sid, (fam, label, unit, kind, sources, (lo, hi)) in CURATED.items():
        lists = []
        for stream, col in sources:
            for dev, pts in by_key.get((fam, stream, col), []):
                lists.append([(t, v) for t, v in pts if lo <= v <= hi])
            used.add((fam, stream, col))
        if sid == "heart.hr":
            lists += hr_from_rr
        pts = merge(lists)
        if sid == "water.sips":
            pts = dedupe_sips(pts)
        if pts:
            out[sid] = Signal(sid, fam, label, unit, kind, pts)
    # heart-rate variability, from the R-R intervals: RMSSD of each minute
    if "heart.rr" in out:
        rm = rmssd_series(out["heart.rr"].points)
        if rm:
            out["heart.rmssd"] = Signal("heart.rmssd", "heart", "HRV (RMSSD, per minute)", "ms", "readings", rm)
    # running total drunk today
    if "water.sips" in out:
        out["water.total"] = Signal("water.total", "water", "Drunk so far today", "mL", "daily_total",
                                    running_daily_total(out["water.sips"].points))
    # Atmos: one signal per metric and sensor (and per device when there are several)
    air = {}
    for (fam, stream, col), lst in by_key.items():
        if fam == "air":
            for dev, pts in lst:
                air.setdefault(col, {}).setdefault(dev, []).append(pts)
            used.add((fam, stream, col))
    for col, devs in air.items():
        metric, _, sensor = col.partition("@")
        label, unit = AIR_META.get(metric, (metric, ""))
        if sensor:
            label += f" ({sensor.upper()})"
        for dev, lists in devs.items():
            sid = f"air.{clean_name(col)}" + (f".{clean_name(dev)}" if len(devs) > 1 else "")
            out[sid] = Signal(sid, "air", label + (f" · {dev}" if len(devs) > 1 else ""), unit, "readings",
                              merge(lists), dev if len(devs) > 1 else None)
    if extras:
        for (dash, stream, col), lst in raw_cols.items():
            fam = family_of(dash, stream)
            if fam == "air" or (fam, stream, col) in used or stream == "sips":
                continue                                  # already a named signal above
            sid = f"raw.{clean_name(dash)}.{stream}.{clean_name(col)}"
            kind = "state" if col in STATE_COLUMNS else "readings"
            out[sid] = Signal(sid, fam, f"{col} ({dash} · {stream})", _unit_of(col), kind,
                              merge([p for _, p in lst]))
    return out


def rmssd_series(rr_points, window_ms=60_000):
    """RMSSD of the R-R intervals in each minute, skipping ectopic / artefact jumps (>20 %)."""
    out, bucket, start = [], [], None
    def flush():
        if len(bucket) >= 10:
            diffs = [b - a for a, b in zip(bucket, bucket[1:]) if abs(b - a) <= 0.2 * a]
            if len(diffs) >= 8:
                out.append((start + window_ms // 2, round(math.sqrt(sum(d * d for d in diffs) / len(diffs)), 1)))
    for t, v in rr_points:
        if start is None or t >= start + window_ms:
            flush()
            bucket, start = [], t - t % window_ms
        bucket.append(v)
    flush()
    return out


def running_daily_total(sips):
    out, day, total = [], None, 0.0
    for t, v in sips:
        d = day_of(t)
        if d != day:
            day, total = d, 0.0
        total += v
        out.append((t, total))
    return out


# ---------------------------------------------------------------------------------- one clock
STEPS_MS = [1000, 5000, 10_000, 30_000, 60_000, 300_000, 900_000, 3_600_000, 10_800_000]


def auto_step(span_ms, max_points=1500):
    for s in STEPS_MS:
        if span_ms / s <= max_points:
            return s
    return STEPS_MS[-1]


def parse_step(text):
    """'auto', '250ms', '1s', '5s', '1min', '1h' or a number of seconds -> ms (None for auto)."""
    if text in (None, "", "auto"):
        return None
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(ms|s|sec|m|min|h)?\s*", str(text).lower())
    if not m:
        raise ValueError(f"can't read '{text}' as a step (try 1s, 30s, 1min, 5min, 1h)")
    n, unit = float(m.group(1)), m.group(2) or "s"
    ms = int(round(n * {"ms": 1, "s": 1000, "sec": 1000, "m": 60000, "min": 60000, "h": 3600000}[unit]))
    if ms <= 0:
        raise ValueError("the step must be more than 0")
    return ms


def grid_for(start_ms, end_ms, step):
    """Step start times: aligned to whole steps of local time, covering [start_ms, end_ms]."""
    origin = local_ms(day_of(start_ms))
    first = origin + ((start_ms - origin) // step) * step
    n = (end_ms - first) // step + 1
    return first, max(0, int(n))


def resample(signal, first, n, step, max_gap=60_000, hold_max=1_800_000, span=None):
    """Put one signal on the grid. Returns dict of lists: mean, min, max, n (real readings per step).
    `span` = (lo, hi): where the device was recording; event steps outside it are None, not 0."""
    pts = signal.points
    mean, lo, hi, cnt = [None] * n, [None] * n, [None] * n, [0] * n
    if not pts or n <= 0:
        return {"mean": mean, "min": lo, "max": hi, "n": cnt}
    ts = [p[0] for p in pts]
    # 1) real readings inside each step
    i = bisect.bisect_left(ts, first)
    end = first + n * step
    acc = {}
    while i < len(pts) and pts[i][0] < end:
        k = (pts[i][0] - first) // step
        acc.setdefault(k, []).append(pts[i][1])
        i += 1
    for k, vals in acc.items():
        cnt[k] = len(vals)
        lo[k], hi[k] = min(vals), max(vals)
        if signal.kind == "events":
            mean[k] = sum(vals)
        elif signal.kind in ("state", "daily_total"):
            mean[k] = vals[-1]
        else:
            mean[k] = sum(vals) / len(vals)
    # 2) steps with no reading
    for k in range(n):
        if cnt[k]:
            continue
        t = first + k * step + step // 2          # the middle of the step
        j = bisect.bisect_left(ts, t)
        if signal.kind == "events":
            if span and span[0] <= t <= span[1]:
                mean[k] = lo[k] = hi[k] = 0.0
            continue
        if signal.kind == "state":
            if j > 0 and t - ts[j - 1] <= hold_max:
                mean[k] = lo[k] = hi[k] = pts[j - 1][1]
            continue
        if signal.kind == "daily_total":          # holds until midnight; 0 before the day's first event
            if j > 0 and day_of(ts[j - 1]) == day_of(t):
                mean[k] = lo[k] = hi[k] = pts[j - 1][1]
            elif span and span[0] <= t <= span[1]:
                mean[k] = lo[k] = hi[k] = 0.0
            continue
        if 0 < j < len(ts):
            (t0, v0), (t1, v1) = pts[j - 1], pts[j]
            if not max_gap or t1 - t0 <= max_gap:
                v = v0 + (v1 - v0) * (t - t0) / (t1 - t0)
                mean[k] = lo[k] = hi[k] = v
    return {"mean": mean, "min": lo, "max": hi, "n": cnt}


def table(signals, start_ms, end_ms, step=None, max_gap=60_000, max_points=1500):
    """All signals on one grid: {"step", "t": [...], "series": {id: resample(...)}, "meta": {...}}."""
    step = step or auto_step(end_ms - start_ms + 1, max_points)
    first, n = grid_for(start_ms, end_ms, step)
    series, meta = {}, {}
    for sid, sig in signals.items():
        span = None
        if sig.kind in ("events", "daily_total"):
            span = recording_span(sig.family, signals)
        series[sid] = resample(sig, first, n, step, max_gap=max_gap, span=span)
        meta[sid] = sig.meta()
    return {"step": step, "first": first, "n": n, "t": [first + k * step for k in range(n)],
            "series": series, "meta": meta}


def recording_span(family, signals):
    """(first, last) reading time of any non-event signal of the family (when it was recording)."""
    lo = hi = None
    for s in signals.values():
        if s.family == family and s.kind not in ("events", "daily_total") and s.points:
            lo = s.points[0][0] if lo is None else min(lo, s.points[0][0])
            hi = s.points[-1][0] if hi is None else max(hi, s.points[-1][0])
    if lo is None:
        return None
    return (lo, hi)


# ---------------------------------------------------------------------------------- summaries
def coverage(points, gap_ms=COVERAGE_GAP_MS):
    """[(start, end)] stretches with readings no further apart than gap_ms."""
    segs = []
    for t, _ in points:
        if segs and t - segs[-1][1] <= gap_ms:
            segs[-1][1] = t
        else:
            segs.append([t, t])
    return [(a, b) for a, b in segs]


def union(segment_lists):
    segs = sorted(s for l in segment_lists for s in l)
    out = []
    for a, b in segs:
        if out and a <= out[-1][1] + COVERAGE_GAP_MS:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def hours(segs):
    return round(sum(b - a for a, b in segs) / 3_600_000, 2)


def time_below(points, limit, max_step_ms=30_000):
    """Minutes spent below `limit`, counting each reading until the next (at most 30 s)."""
    total = 0
    for (t, v), (t2, _) in zip(points, points[1:]):
        if v < limit:
            total += min(t2 - t, max_step_ms)
    return round(total / 60000, 1)


def lowest_rolling_mean(points, window_ms=300_000):
    """Lowest average over any 5-minute window (a resting-rate estimate)."""
    best, j, s = None, 0, 0.0
    for i, (t, v) in enumerate(points):
        s += v
        while points[j][0] < t - window_ms:
            s -= points[j][1]
            j += 1
        cnt = i - j + 1
        if t - points[j][0] >= window_ms * 0.8 and cnt >= 30:
            m = s / cnt
            best = m if best is None else min(best, m)
    return round(best, 1) if best is not None else None


def _stats(points):
    vals = [v for _, v in points]
    if not vals:
        return None
    return {"min": round(min(vals), 1), "avg": round(statistics.fmean(vals), 1), "max": round(max(vals), 1),
            "readings": len(vals)}


def hydro_goal_ml():
    path = os.path.join(os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config"),
                        "bio-dash", "hydro_settings.json")
    try:
        with open(path) as f:
            return int(json.load(f).get("goal_ml") or 2500)
    except Exception:
        return 2500


def refills(root, start_ms, end_ms):
    n = 0
    for path in glob.glob(os.path.join(root, "HidrateSpark", "*", "*", "*_events.csv")):
        try:
            with open(path, newline="", errors="replace") as f:
                rd = csv.reader(clean_lines(f))
                next(rd, None)
                for row in rd:
                    if len(row) > 1 and row[1] == "refill":
                        try:
                            t = to_ms(row[0])
                        except ValueError:
                            continue
                        if start_ms <= t <= end_ms:
                            n += 1
        except (OSError, csv.Error):
            continue
    return n


def summarise(root, start_ms, end_ms, signals=None):
    """Per-family numbers for a range, the recording timeline, and (for several days) a row per day."""
    signals = signals if signals is not None else load_signals(root, start_ms, end_ms, extras=True)
    fam_segments = {}
    for s in signals.values():
        if s.kind not in ("events", "daily_total"):
            fam_segments.setdefault(s.family, []).append(coverage(s.points))
    timeline = {f: union(l) for f, l in fam_segments.items()}
    cards = {}
    g = lambda sid: signals[sid].points if sid in signals else []

    hr, rr, rm = g("heart.hr"), g("heart.rr"), g("heart.rmssd")
    if "heart" in timeline:
        cards["heart"] = {"hours": hours(timeline["heart"]), "hr": _stats(hr),
                          "resting_hr": lowest_rolling_mean(hr),
                          "rmssd_median": round(statistics.median([v for _, v in rm]), 1) if rm else None,
                          "beats": len(rr)}
    spo2, pulse = g("oxygen.spo2"), g("oxygen.pulse")
    if "oxygen" in timeline:
        cards["oxygen"] = {"hours": hours(timeline["oxygen"]), "spo2": _stats(spo2), "pulse": _stats(pulse),
                           "below_95_min": time_below(spo2, 95), "below_90_min": time_below(spo2, 90),
                           "below_88_min": time_below(spo2, 88)}
    if "air" in timeline:
        metrics = []
        for s in signals.values():
            if s.family == "air" and s.points:
                st = _stats(s.points)
                row = {"id": s.id, "label": s.label, "unit": s.unit, **st}
                if s.id.startswith("air.co2"):
                    row["over_1000_min"] = round(sum(min(t2 - t, 30_000) for (t, v), (t2, _) in
                                                     zip(s.points, s.points[1:]) if v > 1000) / 60000, 1)
                metrics.append(row)
        order = {m: i for i, m in enumerate(AIR_META)}
        metrics.sort(key=lambda r: (order.get(r["id"].split(".")[1].split("@")[0], 99), r["id"]))
        cards["air"] = {"hours": hours(timeline["air"]), "metrics": metrics}
    sips = g("water.sips")
    if "water" in timeline or sips:
        days = max(1, round((end_ms - start_ms + 1) / DAY_MS))
        total = round(sum(v for _, v in sips))
        goal = hydro_goal_ml()
        fill = g("water.fill")
        cards["water"] = {"hours": hours(timeline.get("water", [])), "intake_ml": total, "sips": len(sips),
                          "per_day_ml": round(total / days), "goal_ml": goal,
                          "goal_pct": round(100 * total / (goal * days)) if goal else None,
                          "refills": refills(root, start_ms, end_ms),
                          "largest_sip_ml": round(max((v for _, v in sips), default=0)) or None,
                          "fill_last_ml": round(fill[-1][1]) if fill else None}
    other = sorted({s.family for s in signals.values()} - set(cards) - {"other"})
    per_day = []
    days = days_between(day_of(start_ms), day_of(end_ms))
    if len(days) > 1:
        for d in days:
            a, b = local_ms(d), local_ms(d, end=True)
            sub = lambda sid: clip(g(sid), a, b)
            hrd, spd, sipd = sub("heart.hr"), sub("oxygen.spo2"), sub("water.sips")
            co2 = next((clip(s.points, a, b) for s in signals.values() if s.id.startswith("air.co2")), [])
            per_day.append({
                "day": d,
                "hours": {f: hours([(max(x, a), min(y, b)) for x, y in segs if y >= a and x <= b])
                          for f, segs in timeline.items()},
                "hr_avg": round(statistics.fmean(v for _, v in hrd), 1) if hrd else None,
                "spo2_avg": round(statistics.fmean(v for _, v in spd), 1) if spd else None,
                "spo2_min": round(min(v for _, v in spd), 1) if spd else None,
                "co2_avg": round(statistics.fmean(v for _, v in co2)) if co2 else None,
                "intake_ml": round(sum(v for _, v in sipd)) if sipd else
                             (0 if any(y >= a and x <= b for x, y in timeline.get("water", [])) else None),
            })
    return {"start": start_ms, "end": end_ms,
            "timeline": {f: [[a, b] for a, b in segs] for f, segs in timeline.items()},
            "families": {f: {"label": FAMILIES[f][0], "color": FAMILIES[f][1]} for f in FAMILY_ORDER},
            "cards": cards, "per_day": per_day, "unrecognised": other}


# ---------------------------------------------------------------------------------- CSV
def fmt(v):
    if v is None:
        return ""
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.3f}".rstrip("0").rstrip(".")


def write_csv(f, tab, minmax=False, counts=True, drop_empty=True, complete=False):
    """One row per step: time, then each signal (and its min / max, and the number of real
    readings behind it). Returns the number of rows written."""
    ids = list(tab["series"])
    head = ["Timestamp_Epoch_ms", "Time_Local"]
    for sid in ids:
        head.append(sid)
        if minmax and tab["meta"][sid]["kind"] == "readings":
            head += [sid + ".min", sid + ".max"]
        if counts:
            head.append(sid + ".n")
    w = csv.writer(f)
    w.writerow(head)
    rows = 0
    sub_second = tab["step"] % 1000
    for k, t in enumerate(tab["t"]):
        vals = [tab["series"][sid]["mean"][k] for sid in ids]
        if complete and any(v is None for v in vals):
            continue
        if drop_empty and all(v is None for v in vals):
            continue
        stamp = dt.datetime.fromtimestamp(t / 1000)
        row = [t, stamp.strftime("%Y-%m-%d %H:%M:%S") + (f".{t % 1000:03d}" if sub_second else "")]
        for sid, v in zip(ids, vals):
            s = tab["series"][sid]
            row.append(fmt(v))
            if minmax and tab["meta"][sid]["kind"] == "readings":
                row += [fmt(s["min"][k]), fmt(s["max"][k])]
            if counts:
                row.append(s["n"][k])
        w.writerow(row)
        rows += 1
    return rows
