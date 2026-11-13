#!/usr/bin/env python3
"""Is the live ECG forecast's beat timing off because of the model, or because of a display/clock offset?

    python3 tools/forecast_timing.py                         newest forecast log and its recording
    python3 tools/forecast_timing.py <Forecasts/<day>/<time>_forecast_ecg.csv> [--ecg <the session's _ecg.csv>]

It finds the R peaks in the recording (the strap's own samples, ~130 Hz) and in the forecast log (what the
dashboard drew, one row per 8 ms, with lead_ms = how far ahead each point was forecast), pairs each forecast
peak with the nearest real one, and reports the timing error by how far ahead it was forecast:

  * the same offset at every lead, even for forecasts made only 0-100 ms ahead -> an alignment problem
    (clock, grid or drawing), worth fixing;
  * small at short leads and growing with the lead -> the model's beat timing (the next beat's exact moment
    isn't in the last 2 s of ECG): nothing is misaligned.

Both the line drawn (one representative sample) and the median of the samples are checked. Standard library
only; reads the files, writes nothing.
"""
import argparse
import csv
import glob
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "polar_dashboard"))
sys.path.insert(1, os.path.dirname(HERE))          # the shared bt_debug.py

LEAD_BINS = [(0, 100), (100, 200), (200, 300), (300, 400), (400, 500), (500, 10 ** 9)]
MATCH_MS = 350          # a forecast peak further than this from every real one is a miss (half a beat at ~85 bpm)
REFRACTORY_MS = 250     # two R peaks are never closer than this
BASELINE_MS = 200       # the moving mean taken off before looking for peaks
MIN_SHORT = 20          # forecast peaks made 0-100 ms ahead needed for a verdict (~5 min of forecasting)
GAP_MS = 60             # a longer gap between rows splits the series (no peaks found across it)


def data_root():
    try:
        import bt_debug
        return bt_debug.data_root()
    except Exception:          # noqa: BLE001  (standard library fallback)
        return os.environ.get("BIODASH_DATA_DIR") or os.path.join(os.path.expanduser("~"), "Documents", "Bio-dash")


def newest_log():
    logs = sorted(glob.glob(os.path.join(data_root(), "Forecasts", "*", "*_forecast_ecg*.csv")), key=os.path.getmtime)
    return logs[-1] if logs else None


def recording_for(log):
    day = os.path.basename(os.path.dirname(log))
    stamp = os.path.basename(log)[:8]
    hits = glob.glob(os.path.join(data_root(), "Polar H10", "*", day, f"{stamp}_ecg.csv"))
    return hits[0] if hits else None


def read_ecg(path):
    t, v = [], []
    with open(path, newline="", errors="replace") as f:
        for row in csv.reader(f):
            try:
                t.append(float(row[0]))
                v.append(float(row[1]))
            except (ValueError, IndexError):
                continue
    return t, v


def read_log(path):
    """-> times, leads, {"drawn": values, "median": values} for the ECG column."""
    t, lead, drawn, med = [], [], [], []
    with open(path, newline="") as f:
        r = csv.reader(f)
        head = next(r)
        i = next(k for k, h in enumerate(head) if h.startswith("ecg") and not h.endswith(("_median", "_lo", "_hi")))
        j = head.index(head[i] + "_median")
        for row in r:
            try:
                t.append(float(row[0]))
                lead.append(float(row[1]))
                drawn.append(float(row[i]))
                med.append(float(row[j]))
            except (ValueError, IndexError):
                continue
    return t, lead, {"drawn": drawn, "median": med}


def pieces(t):
    """Index ranges [a, b) with no gap longer than GAP_MS."""
    a = 0
    for i in range(1, len(t) + 1):
        if i == len(t) or t[i] - t[i - 1] > GAP_MS or t[i] <= t[i - 1]:
            if i - a > 5:
                yield a, i
            a = i


def r_peaks(t, v):
    """R peak times (ms): local maxima of the signal minus its moving mean, above half its 99th percentile,
    at least REFRACTORY_MS apart, refined with a parabola through the peak and its neighbours."""
    peaks = []
    for a, b in pieces(t):
        tt, vv = t[a:b], v[a:b]
        n = len(tt)
        hp, lo, hi, s = [0.0] * n, 0, 0, 0.0
        for i in range(n):                                  # centred moving mean, by time
            while hi < n and tt[hi] <= tt[i] + BASELINE_MS / 2:
                s += vv[hi]
                hi += 1
            while tt[lo] < tt[i] - BASELINE_MS / 2:
                s -= vv[lo]
                lo += 1
            hp[i] = vv[i] - s / (hi - lo)
        srt = sorted(hp)
        thr = 0.5 * srt[min(n - 1, int(0.99 * n))]
        if thr <= 0:
            continue
        cand = [i for i in range(1, n - 1) if hp[i] > thr and hp[i] >= hp[i - 1] and hp[i] > hp[i + 1]]
        cand.sort(key=lambda i: -hp[i])
        taken = []
        for i in cand:
            if all(abs(tt[i] - tt[j]) >= REFRACTORY_MS for j in taken):
                taken.append(i)
        for i in sorted(taken):
            y0, y1, y2 = hp[i - 1], hp[i], hp[i + 1]
            den = y0 - 2 * y1 + y2
            frac = 0.5 * (y0 - y2) / den if den else 0.0
            frac = max(-0.5, min(0.5, frac))
            dt = (tt[i + 1] - tt[i - 1]) / 2
            peaks.append(tt[i] + frac * dt)
    return peaks


def match(real, fc_peaks, fc_t, fc_lead):
    """(lead_ms, offset_ms) per forecast peak with a real peak within MATCH_MS (offset < 0: forecast early),
    and the number of forecast peaks with none."""
    import bisect
    out, missed = [], 0
    for p in fc_peaks:
        k = bisect.bisect_left(real, p)
        near = [real[j] for j in (k - 1, k) if 0 <= j < len(real)]
        if not near:
            missed += 1
            continue
        r = min(near, key=lambda x: abs(x - p))
        if abs(r - p) > MATCH_MS:
            missed += 1
            continue
        i = min(len(fc_t) - 1, bisect.bisect_left(fc_t, p))
        out.append((fc_lead[i], p - r))
    return out, missed


def summary(pairs):
    rows = []
    for lo, hi in LEAD_BINS:
        offs = [o for lead, o in pairs if lo <= lead < hi]
        if len(offs) < 3:
            rows.append((lo, hi, len(offs), None, None, None))
            continue
        q = statistics.quantiles(offs, n=4)
        rows.append((lo, hi, len(offs), statistics.median(offs), q[0], q[2]))
    return rows


def verdict(rows):
    """(kind, text). The shortest leads decide whether there's a constant offset: a beat forecast only 0-100 ms
    ahead is nearly in the context already, so the model can't be far off -- a clock or drawing offset would be."""
    short = rows[0] if rows[0][2] >= MIN_SHORT and rows[0][3] is not None else None
    longer = [r for r in rows[2:5] if r[3] is not None]
    if not short:
        return "unknown", (f"Only {rows[0][2]} forecast peaks made 0-100 ms ahead (needs {MIN_SHORT}): too few to tell "
                           "an offset from the model's timing (further ahead the model's own error dominates). "
                           "Leave the forecast on for 5 minutes or more, then run this again.")
    s_med, s_spread, s_lead = short[3], short[5] - short[4], f"{short[0]}-{short[1]} ms"
    l_spread = statistics.median([r[5] - r[4] for r in longer]) if longer else None
    if abs(s_med) > 24:
        return "alignment", (f"Forecasts made {s_lead} ahead are already {s_med:+.0f} ms off: that's an alignment "
                             "offset (clock, grid or drawing), not the model. Worth fixing.")
    grows = l_spread is not None and l_spread > 1.5 * max(s_spread, 8)
    text = (f"Forecasts made {s_lead} ahead land within {s_med:+.0f} ms of the real R peak "
            f"(middle half spread {s_spread:.0f} ms), so the drawing and clocks are aligned.")
    if grows:
        text += (f" Further ahead the timing spreads out ({l_spread:.0f} ms at 200-500 ms): that's the model "
                 "not knowing exactly when the next beat comes.")
    return "model", text


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("log", nargs="?", help="Forecasts/<day>/<time>_forecast_ecg.csv (default: the newest)")
    ap.add_argument("--ecg", help="the session's _ecg.csv (default: found from the log's day and time)")
    a = ap.parse_args()
    log = a.log or newest_log()
    if not log or not os.path.exists(log):
        sys.exit(f"No forecast log found under {os.path.join(data_root(), 'Forecasts')}. Turn the ECG forecast on "
                 "for a few minutes first (and don't run the service with --no-log).")
    ecg = a.ecg or recording_for(log)
    if not ecg or not os.path.exists(ecg):
        sys.exit(f"Couldn't find the recording for {log}; pass it with --ecg.")
    print(f"forecast log: {log}\nrecording:    {ecg}")
    et, ev = read_ecg(ecg)
    ft, lead, series = read_log(log)
    real = r_peaks(et, ev)
    span = (ft[-1] - ft[0]) / 1000 if ft else 0
    print(f"{len(real)} R peaks in the recording; forecast log covers {span:.0f} s in {len(list(pieces(ft)))} stretch(es)")
    kinds = {}
    for name, vals in series.items():
        pk = r_peaks(ft, vals)
        pairs, missed = match(real, pk, ft, lead)
        rows = summary(pairs)
        label = "line drawn (one sample)" if name == "drawn" else "median of the samples"
        print(f"\n{label}: {len(pk)} forecast R peaks, {len(pairs)} matched, {missed} with no real beat within "
              f"{MATCH_MS} ms")
        print("  forecast made   peaks   timing error (median, middle half)   - = early")
        for lo, hi, n, med, q1, q3 in rows:
            span_s = f"{lo}-{hi} ms ahead" if hi < 10 ** 9 else f"{lo}+ ms ahead"
            if med is None:
                print(f"  {span_s:<15} {n:>5}   -")
            else:
                print(f"  {span_s:<15} {n:>5}   {med:+6.0f} ms   ({q1:+.0f} to {q3:+.0f})")
        kinds[name] = verdict(rows)
        print("  " + kinds[name][1])
    drawn = kinds["drawn"][0]
    print("\nVerdict: " + {"alignment": "an alignment problem; send this output over.",
                           "model": "the model's beat timing, not a misalignment.",
                           "unknown": "not enough data yet."}[drawn])
    return 0 if drawn != "alignment" else 2


if __name__ == "__main__":
    sys.exit(main())
