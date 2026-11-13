"""Analysis: the work behind the page's summaries, charts and CSV export.

These run in a separate process (see api.py): a week of recordings can take a few hundred MB once
parsed, and that process exits a minute after the last request, giving all of it back -- the rest
of Bio-dash shares the server process and shouldn't carry it. Plain functions in, plain data out.
"""
import datetime as dt
import io
import threading
import time

import biodata as b
import mcint

MAX_DAYS = 92
MAX_CHART_POINTS = 20_000
MAX_EXPORT_CELLS = 40_000_000


class JobError(Exception):
    """A request that can't be answered as asked: (HTTP status, message for the page)."""

    def __init__(self, status, message):
        super().__init__(message)
        self.status, self.message = status, message

    def __reduce__(self):                  # crosses the process boundary intact
        return (JobError, (self.status, self.message))


def root():
    return b.data_root()


def day_range(start, end):
    """Validated (start_ms, end_ms, start_day, end_day) from 'YYYY-MM-DD' strings."""
    try:
        s = dt.date.fromisoformat(start)
        e = dt.date.fromisoformat(end or start)
    except (TypeError, ValueError):
        raise JobError(400, "Dates must look like 2026-10-07.")
    if e < s:
        s, e = e, s
    if (e - s).days + 1 > MAX_DAYS:
        raise JobError(400, f"Pick at most {MAX_DAYS} days.")
    return b.local_ms(s.isoformat()), b.local_ms(e.isoformat(), end=True), s.isoformat(), e.isoformat()


# signals for a range, kept briefly so the page's several requests read the files once
_sig_cache, _sig_lock = {}, threading.Lock()
def signals_for(start_ms, end_ms, units=""):
    """units: "unit folder|unit folder": only these straps / rings (one person's); "": every unit, merged."""
    key = (root(), start_ms, end_ms, units)
    with _sig_lock:
        hit = _sig_cache.get(key)
        if hit and time.time() - hit[0] < 30:
            return hit[1]
    sigs = b.load_signals(key[0], start_ms, end_ms, units=[u for u in units.split("|") if u] or None)
    with _sig_lock:
        _sig_cache.clear() if len(_sig_cache) > 3 else None
        _sig_cache[key] = (time.time(), sigs)
    return sigs




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


def units(start: str, end: str = ""):
    s, e, _, _ = day_range(start, end)
    return b.worn_units(root(), s, e)


def summary(start: str, end: str = "", units: str = ""):
    s, e, sd, ed = day_range(start, end)
    sigs = signals_for(s, e, units)
    out = b.summarise(root(), s, e, sigs)
    out["nights"] = nights_in(out, sd, ed)
    out["days"] = b.days_between(sd, ed)
    return out


def signal_list(start: str, end: str = "", units: str = ""):
    s, e, _, _ = day_range(start, end)
    sigs = signals_for(s, e, units)
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
        raise JobError(400, str(ex))
    st = st or b.auto_step(span_ms)
    if span_ms // st > max_points:
        raise JobError(400, f"That step gives {span_ms // st:,} points per signal over this range; "
                                 f"pick a larger step or fewer days.")
    return chosen, missing, st


def _r(v, nd=2):
    return None if v is None else round(v, nd)


def series(start: str, end: str = "", ids: str = "", step: str = "auto", max_gap: int = 60, units: str = ""):
    s, e, _, _ = day_range(start, end)
    sigs = signals_for(s, e, units)
    chosen, missing, st = _ids_and_step(sigs, ids, step, e - s + 1, MAX_CHART_POINTS)
    tab = b.table(chosen, s, e, step=st, max_gap=max_gap * 1000)
    return {"step": tab["step"], "first": tab["first"], "n": tab["n"], "missing": missing,
            "meta": tab["meta"],
            "series": {sid: {"mean": [_r(v) for v in r["mean"]],
                             "min": [_r(v) for v in r["min"]] if tab["meta"][sid]["kind"] == "readings" else None,
                             "max": [_r(v) for v in r["max"]] if tab["meta"][sid]["kind"] == "readings" else None,
                             "n": r["n"]}
                       for sid, r in tab["series"].items()}}


def export(start: str, end: str = "", ids: str = "", step: str = "1min", minmax: int = 0, counts: int = 1,
           max_gap: int = 60, units: str = ""):
    s, e, sd, ed = day_range(start, end)
    sigs = signals_for(s, e, units)
    chosen, _, st = _ids_and_step(sigs, ids, step, e - s + 1, 10_000_000)
    if not chosen:
        raise JobError(400, "Pick at least one signal.")
    if (e - s) // st * len(chosen) > MAX_EXPORT_CELLS:
        raise JobError(400, "That export would be too large; pick a larger step, fewer days or fewer signals.")
    tab = b.table(chosen, s, e, step=st, max_gap=max_gap * 1000)
    buf = io.StringIO()
    b.write_csv(buf, tab, minmax=bool(minmax), counts=bool(counts))
    step_txt = f"{st // 60000}min" if st % 60000 == 0 else f"{st / 1000:g}s"
    name = f"bio-dash_{sd}" + (f"_to_{ed}" if ed != sd else "") + f"_{step_txt}.csv"
    return buf.getvalue(), name


def export_mcint(start: str, end: str = "", ids: str = "", step: str = "10s", max_gap: int = 60, hold: int = 120,
                 missing: str = "mark", flags: int = 0, units: str = ""):
    """The selected signals as an integer CSV for ReaLLM-Forge's data/csv_mc_int, zipped with
    its columns.json (scales, ranges) and prepare.sh (get_dataset.sh with every --range)."""
    import time
    import zipfile
    if missing not in ("mark", "drop"):
        raise JobError(400, "missing must be mark or drop")
    s, e, sd, ed = day_range(start, end)
    sigs = signals_for(s, e, units)
    chosen, _, st = _ids_and_step(sigs, ids, step, e - s + 1, 10_000_000)
    if not chosen:
        raise JobError(400, "Pick at least one signal.")
    if (e - s) // st * len(chosen) > MAX_EXPORT_CELLS:
        raise JobError(400, "That export would be too large; pick a larger step, fewer days or fewer signals.")
    tab = b.table(dict(sorted(chosen.items())), s, e, step=st, max_gap=max_gap * 1000)
    try:
        res = mcint.build(tab, chosen, hold_ms=hold * 1000, missing=missing, real_flags=bool(flags))
    except ValueError as ex:
        raise JobError(400, str(ex))
    if not res["rows"]:
        raise JobError(400, "No rows: every row lacked at least one signal. Use 'mark' for missing readings, or fewer signals.")
    step_txt = f"{st // 60000}min" if st % 60000 == 0 else f"{st / 1000:g}s"
    name = f"bio-dash_{sd}" + (f"_to_{ed}" if ed != sd else "") + f"_{step_txt}_mcint"
    csv_buf = io.StringIO()
    mcint.write_csv(csv_buf, res)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{name}/{name}.csv", csv_buf.getvalue())
        z.writestr(f"{name}/{name}.columns.json", mcint.columns_json(res, source=sd if ed == sd else f"{sd} to {ed}"))
        info = zipfile.ZipInfo(f"{name}/{name}.prepare.sh", date_time=time.localtime()[:6])
        info.external_attr = 0o100755 << 16
        info.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(info, mcint.prepare_sh(res, name + ".csv"))
    return buf.getvalue(), name + ".zip"
