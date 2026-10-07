#!/usr/bin/env python3
"""Overnight SpO2 / pulse report from Bio-dash recordings (Viatom O2 band, or Omni).

    ./tools/night_report.py                    the most recent night that has data
    ./tools/night_report.py 2026-10-06         the night starting that evening (18:00 -> 12:00 next day)
    ./tools/night_report.py --files a.csv b.csv   specific _vitals.csv files
Options: --device "Band-WU 0347"   --out report.html   --data-dir <Bio-dash folder>

Joins every _vitals.csv session in the window (dropouts become gaps), cleans the data, and writes
one self-contained HTML report (charts included, opens on any device) to
<Documents>/Bio-dash/Reports/ by default.

Cleaning: 0 / out-of-range values are dropped; stretches where SpO2 *and* pulse stay exactly the
same for 60 s or more are flagged as "frozen" (the band keeps logging its last reading when the
sensor loses contact) and left out of the statistics.

Not a medical device. The numbers are a consumer-oximeter summary, not a sleep study.
Standard library only.
"""
import argparse
import csv
import datetime as dt
import glob
import html
import os
import statistics
import sys

GAP_S = 30                 # no sample for longer than this = a gap
FROZEN_S = 60              # identical SpO2 + pulse for this long = sensor not reading
DESAT_MIN_S = 10           # a desaturation must last at least this long
BASELINE_S = 120           # baseline = median SpO2 over the previous 2 minutes
DESAT_MAX_S = 120          # longer than this is a lower plateau, not a dip
SOURCES = ("Viatom O2", "Omni")


# ---------------------------------------------------------------------------
# Finding and loading recordings
# ---------------------------------------------------------------------------
def data_root(override=None):
    if override:
        return override
    if os.environ.get("BIODASH_DATA_DIR"):
        return os.environ["BIODASH_DATA_DIR"]
    docs = os.path.expanduser("~/Documents")
    try:
        with open(os.path.expanduser("~/.config/user-dirs.dirs")) as f:
            for line in f:
                if line.startswith("XDG_DOCUMENTS_DIR="):
                    docs = os.path.expandvars(line.split("=", 1)[1].strip().strip('"'))
    except OSError:
        pass
    return os.path.join(docs, "Bio-dash")


def all_vitals_files(root, device=None):
    files = []
    for src in SOURCES:
        for f in glob.glob(os.path.join(root, src, "*", "*", "*_vitals.csv")):
            dev = os.path.basename(os.path.dirname(os.path.dirname(f)))
            if device and device.lower() not in dev.lower():
                continue
            if src == "Omni" and "polar" in dev.lower():
                continue
            files.append((src, dev, f))
    return files


def read_rows(path):
    rows = []
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r, None) or []
        try:
            it, isp, ihr = header.index("Timestamp_Epoch_ms"), header.index("SpO2_pct"), header.index("HR_BPM")
        except ValueError:
            return rows
        for line in r:
            try:
                ts = int(float(line[it])) / 1000
                spo2 = float(line[isp]) if line[isp] not in ("", "None") else None
                hr = float(line[ihr]) if line[ihr] not in ("", "None") else None
            except (ValueError, IndexError):
                continue
            rows.append((ts, spo2, hr))
    return rows


def night_window(date):
    start = dt.datetime.combine(date, dt.time(18, 0))
    return start.timestamp(), (start + dt.timedelta(hours=18)).timestamp()


def pick_night(files):
    """Evening date of the most recent night that has readings."""
    latest = None
    for _, _, f in files:
        rows = read_rows(f)
        if rows:
            latest = max(latest or 0, rows[-1][0])
    if latest is None:
        return None
    d = dt.datetime.fromtimestamp(latest)
    return (d - dt.timedelta(days=1)).date() if d.hour < 18 else d.date()


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def analyse(rows):
    rows = sorted(r for r in rows if r[1] is not None)
    # overlapping session files: keep one sample per second
    seen, uniq = set(), []
    for r in rows:
        if int(r[0]) not in seen:
            seen.add(int(r[0])); uniq.append(r)
    rows = uniq
    # 1. drop zeros / impossible values (before the first reading, sensor off)
    clean = [(t, s, h if h and 25 <= h <= 220 else None) for t, s, h in rows if 50 <= s <= 100]
    # 2. frozen stretches: SpO2 and pulse unchanged for >= FROZEN_S
    frozen, i = [], 0
    while i < len(clean):
        j = i
        while j + 1 < len(clean) and clean[j + 1][1] == clean[i][1] and clean[j + 1][2] == clean[i][2] \
                and clean[j + 1][0] - clean[j][0] <= GAP_S:
            j += 1
        if clean[j][0] - clean[i][0] >= FROZEN_S:
            frozen.append((clean[i][0], clean[j][0]))
        i = j + 1
    def is_frozen(t):
        return any(a <= t <= b for a, b in frozen)
    good = [r for r in clean if not is_frozen(r[0])]
    # 3. gaps (between any samples, including dropped ones)
    gaps = [(a[0], b[0]) for a, b in zip(rows, rows[1:]) if b[0] - a[0] > GAP_S]
    # 4. valid time: sum of sample spacing, capped (gaps don't count)
    def duration(sample):
        return sum(min(b[0] - a[0], GAP_S) for a, b in zip(sample, sample[1:]))
    valid_s = duration(good)
    spo2 = [s for _, s, _ in good]
    hr = [h for _, _, h in good if h]
    # 5. time below thresholds (time-weighted)
    below = {}
    for thr in (95, 92, 90, 88):
        below[thr] = sum(min(b[0] - a[0], GAP_S) for a, b in zip(good, good[1:]) if a[1] < thr)
    # 6. desaturation events: drop of >= 3 / 4 points below the 2-minute baseline, lasting >= 10 s
    events, plateaus = {3: [], 4: []}, []
    def close(ev, drop):
        length = ev["end"] - ev["start"]
        if length > DESAT_MAX_S:
            if drop == 3:
                plateaus.append(ev)        # settled at a lower level: reported separately
        elif length >= DESAT_MIN_S:
            events[drop].append(ev)
    for drop in (3, 4):
        k0, in_ev = 0, None
        for k, (t, s, _) in enumerate(good):
            while good[k0][0] < t - BASELINE_S:
                k0 += 1
            if in_ev is None:
                window = [x[1] for x in good[k0:k]]
                # the median ignores normal one-point flicker; needs ~30 s of history
                if len(window) >= 10 and s <= statistics.median(window) - drop:
                    in_ev = {"start": t, "base": statistics.median(window), "nadir": s, "end": t}
            elif s <= in_ev["base"] - drop + 1 and t - good[k - 1][0] <= GAP_S:
                in_ev["end"] = t; in_ev["nadir"] = min(in_ev["nadir"], s)
            else:
                close(in_ev, drop); in_ev = None
        if in_ev:
            close(in_ev, drop)
    hours = valid_s / 3600 if valid_s else 0
    # lowest sustained SpO2: minimum of 10-s rolling medians (ignores single-sample blips)
    sustained = []
    for k in range(len(good)):
        w = [x[1] for x in good[k:] if x[0] - good[k][0] <= 10]
        if len(w) >= 3:
            sustained.append(statistics.median(w))
    return {
        "good": good, "frozen": frozen, "gaps": gaps, "events": events, "plateaus": plateaus, "rhythm": rhythm(good),
        "start": rows[0][0] if rows else None, "end": rows[-1][0] if rows else None,
        "valid_s": valid_s, "hours": hours, "below": below,
        "spo2_mean": statistics.fmean(spo2) if spo2 else None, "spo2_median": statistics.median(spo2) if spo2 else None,
        "spo2_min": min(spo2) if spo2 else None, "spo2_min_sustained": min(sustained) if sustained else None,
        "hr_mean": statistics.fmean(hr) if hr else None, "hr_min": min(hr) if hr else None, "hr_max": max(hr) if hr else None,
        "odi3": len(events[3]) / hours if hours else None, "odi4": len(events[4]) / hours if hours else None,
        "dropped": len(rows) - len(clean),
    }


# ---------------------------------------------------------------------------
# Report (inline SVG, no dependencies)
# ---------------------------------------------------------------------------
def hm(ts):
    return dt.datetime.fromtimestamp(ts).strftime("%H:%M")


def mins(s):
    s = int(round(s / 60))
    return f"{s // 60} h {s % 60:02d} min" if s >= 60 else f"{s} min"


RHYTHM_BLOCK_S = 300       # sleep rhythm: pulse steadiness per 5-minute block


def rhythm(good):
    """Quiet / active / awake stretches from how steady the pulse is. A guess, not sleep staging."""
    blocks = {}
    for t, _, h in good:
        if h:
            blocks.setdefault(int(t // RHYTHM_BLOCK_S), []).append(h)
    blocks = {k: v for k, v in blocks.items() if len(v) >= 30}
    if len(blocks) < 6:
        return []
    sd = {k: statistics.pstdev(v) for k, v in blocks.items()}
    mean = {k: statistics.fmean(v) for k, v in blocks.items()}
    sd_typ, hr_typ = statistics.median(sd.values()), statistics.median(mean.values())
    def classify(k):
        if sd[k] > max(8, 3 * sd_typ) or mean[k] > hr_typ + 12:
            return "awake"
        return "active" if sd[k] > max(3.5, 1.4 * sd_typ) else "quiet"
    keys = sorted(blocks)
    state = {k: classify(k) for k in keys}
    # smooth: a block takes the majority state of the 25 minutes around it (awake always stands)
    smooth = {}
    for k in keys:
        near = [state[j] for j in range(k - 2, k + 3) if j in state]
        smooth[k] = state[k] if state[k] == "awake" else max(("quiet", "active"), key=lambda x: (near.count(x), x == state[k]))
    out = []
    for k in keys:
        t0 = k * RHYTHM_BLOCK_S
        if out and out[-1]["state"] == smooth[k] and out[-1]["end"] == t0:
            out[-1]["end"] = t0 + RHYTHM_BLOCK_S
        else:
            out.append({"state": smooth[k], "start": t0, "end": t0 + RHYTHM_BLOCK_S})
    return out


def svg_rhythm(a, height=30):
    t0, t1 = a["start"], a["end"]
    if not a["rhythm"] or t1 <= t0:
        return ""
    PAD = 14
    pw = max(600, (t1 - t0) / 3600 * PX_PER_HOUR)
    X = lambda t: PAD + (max(t0, min(t1, t)) - t0) / (t1 - t0) * pw
    names = {"quiet": "Quiet (steady pulse)", "active": "Active (jumpy pulse: REM, restless or briefly awake)", "awake": "Likely awake"}
    out = [f"<svg width='{pw + 2 * PAD:.0f}' height='{height}' role='img' aria-label='Sleep rhythm'>"]
    for r in a["rhythm"]:
        out.append(f"<rect x='{X(r['start']):.1f}' y='4' width='{max(1, X(r['end']) - X(r['start']) - 1):.1f}' height='{height - 8}' rx='4' class='r-{r['state']}'>"
                   f"<title>{hm(r['start'])}–{hm(r['end'])} · {names[r['state']]}</title></rect>")
    out.append("</svg>")
    return "".join(out)


PX_PER_HOUR = 360          # chart width: the night scrolls sideways instead of being squeezed
AXIS_W = 46


def svg_series(a, key, lo, hi, color, height=230, ref=None, events=(), unit=""):
    """Returns (fixed y-axis svg, wide plot svg). The plot is PX_PER_HOUR wide per hour."""
    good, t0, t1 = a["good"], a["start"], a["end"]
    if not good or t1 <= t0:
        return "", "<p class='muted'>No data.</p>"
    T, B, PAD = 10, 24, 14
    ph = height - T - B
    pw = max(600, (t1 - t0) / 3600 * PX_PER_HOUR)
    width = pw + 2 * PAD
    X = lambda t: PAD + (t - t0) / (t1 - t0) * pw
    Y = lambda v: T + (hi - max(lo, min(hi, v))) / (hi - lo) * ph
    step = 5 if hi - lo <= 30 else 20
    ticks, v = [], lo - lo % step + step
    while v < hi:
        ticks.append(v); v += step
    axis = [f"<svg class='yaxis' width='{AXIS_W}' height='{height}' aria-hidden='true'>"]
    axis += [f"<text x='{AXIS_W - 6}' y='{Y(v) + 4:.1f}' class='ax' text-anchor='end'>{v}{unit}</text>" for v in ticks]
    axis.append("</svg>")
    out = [f"<svg width='{width:.0f}' height='{height}' role='img' aria-label='{key} over the night'>"]
    for g0, g1 in a["gaps"]:
        out.append(f"<rect x='{X(g0):.1f}' y='{T}' width='{max(1, X(g1) - X(g0)):.1f}' height='{ph}' class='gap'/>")
    for f0, f1 in a["frozen"]:
        out.append(f"<rect x='{X(f0):.1f}' y='{T}' width='{max(1, X(f1) - X(f0)):.1f}' height='{ph}' class='frozen'/>")
    for e in events:
        out.append(f"<rect x='{X(e['start']):.1f}' y='{T}' width='{max(2, X(e['end']) - X(e['start'])):.1f}' height='{ph}' class='event'>"
                   f"<title>{hm(e['start'])} · {e['base']:.0f}% → {e['nadir']:.0f}% · {int(e['end'] - e['start'])} s</title></rect>")
    for v in ticks:
        out.append(f"<line x1='0' x2='{width:.0f}' y1='{Y(v):.1f}' y2='{Y(v):.1f}' class='grid'/>")
    if ref is not None:
        out.append(f"<line x1='0' x2='{width:.0f}' y1='{Y(ref):.1f}' y2='{Y(ref):.1f}' class='ref'/>")
    # a tick every half hour; the full hours are drawn stronger
    h = dt.datetime.fromtimestamp(t0).replace(minute=0, second=0, microsecond=0)
    while h.timestamp() <= t1:
        if h.timestamp() >= t0:
            x, full = X(h.timestamp()), h.minute == 0
            out.append(f"<line x1='{x:.1f}' x2='{x:.1f}' y1='{T}' y2='{T + ph}' class='grid{'' if full else ' half'}'/>"
                       f"<text x='{x:.1f}' y='{height - 6}' class='ax{' hour' if full else ''}' text-anchor='middle'>{h.strftime('%H:%M')}</text>")
        h += dt.timedelta(minutes=30)
    idx = 1 if key == "spo2" else 2
    path, prev = [], None
    for r in good:
        if r[idx] is None:
            prev = None; continue
        cmd = "L" if prev is not None and r[0] - prev <= GAP_S else "M"
        path.append(f"{cmd}{X(r[0]):.1f},{Y(r[idx]):.1f}")
        prev = r[0]
    out.append(f"<path d='{' '.join(path)}' fill='none' stroke='{color}' stroke-width='1.5' stroke-linejoin='round'/>")
    out.append("</svg>")
    return "".join(axis), "".join(out)


def svg_histogram(a, width=460, height=190):
    good = a["good"]
    secs = {}
    for p, q in zip(good, good[1:]):
        secs[int(p[1])] = secs.get(int(p[1]), 0) + min(q[0] - p[0], GAP_S)
    if not secs:
        return ""
    lo, hi = min(min(secs), 85), 100
    total = sum(secs.values())
    L, B, T = 4, 26, 12
    n = hi - lo + 1
    bw = (width - L - 12) / n
    mx = max(secs.values())
    out = [f"<svg class='fit' viewBox='0 0 {width} {height}' role='img' aria-label='Time spent at each SpO2 value'>"]
    for i, v in enumerate(range(lo, hi + 1)):
        s = secs.get(v, 0)
        h = (height - T - B) * s / mx if mx else 0
        cls = "bar low" if v < 90 else ("bar mid" if v < 94 else "bar")
        x = L + i * bw
        out.append(f"<rect x='{x + 1:.1f}' y='{height - B - h:.1f}' width='{bw - 2:.1f}' height='{h:.1f}' class='{cls}'>"
                   f"<title>{v}%: {mins(s)} ({100 * s / total:.1f}% of the night)</title></rect>")
        if v % 2 == 0:
            out.append(f"<text x='{x + bw / 2:.1f}' y='{height - 8}' class='ax' text-anchor='middle'>{v}</text>")
    out.append("</svg>")
    return "".join(out)


def report(a, title, sources):
    f1 = lambda v, d=1: "—" if v is None else f"{v:.{d}f}"
    pct = lambda s: f"{100 * s / a['valid_s']:.1f}%" if a["valid_s"] else "—"
    ev4 = a["events"][4]
    ev_sorted = sorted(ev4, key=lambda e: (e["nadir"], e["start"]))
    row = lambda e: f"<tr><td>{hm(e['start'])}</td><td>{int(e['end'] - e['start'])} s</td><td>{e['base']:.0f}% → {e['nadir']:.0f}%</td></tr>"
    head = "<tr><th>Time</th><th>Length</th><th>Drop</th></tr>"
    ev_rows = "<table>" + head + ("".join(map(row, ev_sorted[:8])) or "<tr><td colspan='3' class='muted'>None</td></tr>") + "</table>"
    if len(ev_sorted) > 8:
        ev_rows += (f"<details><summary>Show all {len(ev_sorted)} dips</summary><table>" + head
                    + "".join(map(row, sorted(ev_sorted[8:], key=lambda e: e["start"]))) + "</table></details>")
    sx, sp = svg_series(a, "spo2", max(70, int((a['spo2_min'] or 85) // 5 * 5) - 5), 100, "var(--spo2)", ref=90, events=ev4, unit="%")
    hx, hp = svg_series(a, "hr", max(30, int((a['hr_min'] or 50) // 10 * 10) - 10), int((a['hr_max'] or 100) // 10 * 10) + 10, "var(--hr)")
    rh = svg_rhythm(a)
    rh_axis = "<div class='lab' style='height:50px;line-height:30px;padding-top:20px'>Rhythm</div>" if rh else ""
    rh_scroll = f"<div class='lab'>&nbsp;</div>{rh}" if rh else ""
    tot = {k: sum(r["end"] - r["start"] for r in a["rhythm"] if r["state"] == k) for k in ("quiet", "active", "awake")}
    act = [r for r in a["rhythm"] if r["state"] == "active" and r["end"] - r["start"] >= 600]
    rh_text = ""
    if rh:
        rh_text = (f"<div class='panel'><h3>Sleep rhythm (estimate from pulse)</h3><p style='margin:0 0 6px'>Quiet {mins(tot['quiet'])} · active {mins(tot['active'])} · "
                   f"likely awake {mins(tot['awake'])}." + (" Longer active stretches: " + ", ".join(f"{hm(r['start'])}–{hm(r['end'])}" for r in act) + "." if act else "")
                   + "</p><p class='sub' style='margin:0;font-size:13px'>Quiet = low, steady pulse (typical of deep sleep). Active = jumpy pulse, which REM sleep, "
                   "restlessness and brief awakenings all produce — pulse alone can't tell them apart, so this is not sleep staging.</p></div>")
    frozen_total = sum(b - a_ for a_, b in a["frozen"])
    gap_total = sum(b - a_ for a_, b in a["gaps"])
    cards = [
        ("Recorded", mins(a["valid_s"]), f"{hm(a['start'])} → {hm(a['end'])}"),
        ("Average SpO₂", f"{f1(a['spo2_mean'])}%", f"median {f1(a['spo2_median'], 0)}%"),
        ("Lowest SpO₂", f"{f1(a['spo2_min_sustained'], 0)}%", f"single reading {f1(a['spo2_min'], 0)}%"),
        ("Time below 90%", mins(a["below"][90]), pct(a["below"][90]) + " of the night"),
        ("Dips ≥ 4% (ODI 4)", f"{f1(a['odi4'])}/h", f"{len(ev4)} events · ODI 3: {f1(a['odi3'])}/h"),
        ("Pulse", f"{f1(a['hr_mean'], 0)} bpm", f"range {f1(a['hr_min'], 0)}–{f1(a['hr_max'], 0)}"),
    ]
    card_html = "".join(f"<div class='card'><div class='k'>{k}</div><div class='v'>{v}</div><div class='s'>{s}</div></div>" for k, v, s in cards)
    quality = []
    if a["gaps"]:
        quality.append(f"{len(a['gaps'])} gap(s) in the recording, {mins(gap_total)} in total (shaded grey).")
    if a["frozen"]:
        quality.append(f"{len(a['frozen'])} frozen stretch(es), {mins(frozen_total)} in total, where SpO₂ and pulse didn't change at all — "
                       "usually the sensor losing contact (shaded amber). Left out of the numbers.")
    if a["dropped"]:
        quality.append(f"{a['dropped']} samples with no valid reading (e.g. before the first measurement) ignored.")
    for e in a["plateaus"]:
        quality.append(f"{hm(e['start'])}–{hm(e['end'])}: SpO₂ settled around {e['nadir']:.0f}–{e['nadir'] + 2:.0f}% for {mins(e['end'] - e['start'])} "
                       "(a lower plateau, not counted as a dip).")
    quality_html = "".join(f"<li>{html.escape(q)}</li>" for q in quality) or "<li>No gaps, frozen stretches or long low plateaus.</li>"
    below_rows = "".join(f"<tr><td>below {t}%</td><td>{mins(a['below'][t])}</td><td>{pct(a['below'][t])}</td></tr>" for t in (95, 92, 90, 88))
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>
:root{{--bg:#f8fafc;--fg:#0f172a;--muted:#64748b;--card:#fff;--line:#e2e8f0;--spo2:#2563eb;--hr:#e11d48;--ev:rgba(234,88,12,.22);--gap:rgba(100,116,139,.18);--frz:rgba(217,119,6,.25);--rq:#1e40af;--ra:#a78bfa;--rw:#f59e0b}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0f172a;--fg:#e2e8f0;--muted:#94a3b8;--card:#1e293b;--line:#334155;--spo2:#60a5fa;--hr:#fb7185}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}}
main{{max-width:1040px;margin:0 auto;padding:28px 16px 48px}}h1{{font-size:26px;margin:0 0 2px;letter-spacing:-.01em}}
h2{{font-size:17px;margin:32px 0 6px}}h3{{font-size:14px;margin:0 0 10px;font-weight:600}}
.muted,.sub{{color:var(--muted)}}.hint{{font-size:12px;margin-top:6px}}
.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:20px}}
@media (min-width:900px){{.cards{{grid-template-columns:repeat(6,1fr)}}}}@media (max-width:480px){{.cards{{grid-template-columns:repeat(2,1fr)}}}}
.card,.panel{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px 16px}}.panel{{margin-top:14px}}
.k{{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}}
.v{{font-size:22px;font-weight:700;margin-top:4px;white-space:nowrap;font-variant-numeric:tabular-nums}}.s{{font-size:12px;color:var(--muted)}}
.chart{{display:flex;padding:12px 0 6px 8px;margin-top:10px}}.axes{{flex:none}}.scroll{{overflow-x:auto;flex:1;min-width:0;scrollbar-width:thin}}
.scroll:focus-visible{{outline:2px solid var(--spo2);outline-offset:2px}}.chart svg{{display:block}}
.lab{{font-size:12px;font-weight:600;color:var(--muted);height:20px;white-space:nowrap;position:relative;z-index:1}}.axes .lab{{width:46px;overflow:visible}}
svg.fit{{width:100%;height:auto;display:block}}
.grid{{stroke:var(--line);stroke-width:1}}.grid.half{{stroke-dasharray:2 4}}.ref{{stroke:var(--hr);stroke-dasharray:5 4;stroke-width:1.2}}
.ax{{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}}.ax.hour{{fill:var(--fg);font-weight:600}}
.r-quiet{{fill:var(--rq)}}.r-active{{fill:var(--ra)}}.r-awake{{fill:var(--rw)}}.gap{{fill:var(--gap)}}.frozen{{fill:var(--frz)}}.event{{fill:var(--ev)}}
.bar{{fill:var(--spo2)}}.bar.mid{{fill:#d97706}}.bar.low{{fill:#dc2626}}
.legend{{font-size:13px}}.legend span{{display:inline-block;width:11px;height:11px;border-radius:3px;margin:0 5px 0 14px;vertical-align:-1px}}
.legend span:first-child{{margin-left:0}}.legend .dash{{background:none;border-top:2px dashed var(--hr);border-radius:0;height:0;vertical-align:3px}}
table{{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}}td,th{{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line)}}
th{{font-size:12px;color:var(--muted);font-weight:600}}tr:last-child td{{border-bottom:0}}
details{{margin-top:8px}}summary{{cursor:pointer;color:var(--spo2);font-size:13px}}ul{{margin:0;padding-left:18px}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}@media (max-width:640px){{.two{{grid-template-columns:1fr}}}}
.note{{border-radius:14px;padding:12px 16px;font-size:13px;color:var(--muted);margin-top:14px;border:1px dashed var(--line)}}
</style></head><body><main>
<h1>{html.escape(title)}</h1><div class="sub">{html.escape(sources)}</div>
<div class="cards">{card_html}</div>
<h2>Through the night</h2>
<div class="legend sub"><span style="background:var(--spo2)"></span>SpO₂<span style="background:var(--hr)"></span>pulse<span style="background:var(--ev)"></span>dip ≥ 4%<span style="background:var(--gap)"></span>no data<span style="background:var(--frz)"></span>sensor not reading<span class="dash"></span>90%<span style="background:var(--rq)"></span>quiet<span style="background:var(--ra)"></span>active<span style="background:var(--rw)"></span>awake</div>
<div class="panel chart"><div class="axes"><div class="lab">SpO₂</div>{sx}<div class="lab">Pulse (bpm)</div>{hx}{rh_axis}</div>
<div class="scroll" tabindex="0"><div class="lab">&nbsp;</div>{sp}<div class="lab">&nbsp;</div>{hp}{rh_scroll}</div></div>
<div class="sub hint">Scroll sideways to move through the night · hover a shaded dip for details</div>
{rh_text}
<div class="two"><div class="panel"><h3>Time at each SpO₂ value</h3>{svg_histogram(a)}</div>
<div class="panel"><h3>Time below</h3><table>{below_rows}</table></div></div>
<div class="panel"><h3>Deepest dips (≥ 4%)</h3>{ev_rows}</div>
<div class="panel"><h3>Recording notes</h3><ul>{quality_html}</ul></div>
<div class="note"><b>Not a medical device or a sleep study.</b> These numbers come from a consumer finger/ring oximeter and simple rules:
a "dip" is SpO₂ falling at least 4 (or 3) points below the typical (median) value of the previous 2 minutes for 10 s to 2 min; ODI is dips per
hour of valid recording. Motion and poor contact cause false dips. If you regularly see many dips per hour, long stretches below 90%,
or you snore heavily or wake up unrefreshed, that's worth discussing with a doctor, who can arrange a proper sleep study.</div>
<p class="sub">Generated by Bio-dash tools/night_report.py · {dt.datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
</main></body></html>"""


def main():
    ap = argparse.ArgumentParser(description="Overnight SpO2 / pulse report from Bio-dash recordings.")
    ap.add_argument("night", nargs="?", help="evening date YYYY-MM-DD (default: most recent night with data)")
    ap.add_argument("--files", nargs="+", help="specific _vitals.csv files instead of searching")
    ap.add_argument("--device", help="only this device folder (substring match)")
    ap.add_argument("--data-dir", help="Bio-dash recordings folder (default: Documents/Bio-dash)")
    ap.add_argument("--out", help="output .html path")
    a = ap.parse_args()
    root = data_root(a.data_dir)

    if a.files:
        chosen = [("files", os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(f)))), f) for f in a.files]
        rows = [r for _, _, f in chosen for r in read_rows(f)]
        if a.night:
            night = dt.date.fromisoformat(a.night)
        else:                      # the night with the most readings in these files
            count = {}
            for r in rows:
                d = dt.datetime.fromtimestamp(r[0])
                d = (d - dt.timedelta(days=1)).date() if d.hour < 18 else d.date()
                count[d] = count.get(d, 0) + 1
            night = max(count, key=count.get) if count else dt.date.today()
        t0, t1 = night_window(night)
        rows = [r for r in rows if t0 <= r[0] < t1]
    else:
        files = all_vitals_files(root, a.device)
        if not files:
            sys.exit(f"No SpO₂ recordings (*_vitals.csv) found under {root}/Viatom O2 or {root}/Omni.")
        night = dt.date.fromisoformat(a.night) if a.night else pick_night(files)
        t0, t1 = night_window(night)
        chosen, rows = [], []
        for src, dev, f in files:
            r = [x for x in read_rows(f) if t0 <= x[0] < t1]
            if r:
                chosen.append((src, dev, f)); rows += r
    if not rows:
        sys.exit(f"No readings for the night of {night} (18:00 → 12:00 next day).")
    devices = sorted({d for _, d, _ in chosen})
    res = analyse(rows)
    if not res["good"]:
        sys.exit("Recordings found, but no valid SpO₂ readings in them.")
    title = f"Night of {night.strftime('%a %d %b %Y')}"
    sources = f"{', '.join(devices)} · {len(chosen)} recording session(s)"
    out = a.out or os.path.join(root, "Reports", f"night_{night.isoformat()}.html")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        f.write(report(res, title, sources))
    print(f"{title}: {mins(res['valid_s'])} recorded from {len(chosen)} session(s)")
    print(f"  SpO₂ average {res['spo2_mean']:.1f}%, lowest sustained {res['spo2_min_sustained']:.0f}%, "
          f"below 90% for {mins(res['below'][90])}; dips ≥4%: {len(res['events'][4])} ({res['odi4']:.1f}/h)")
    print(f"  Report: {out}")


if __name__ == "__main__":
    main()
