#!/usr/bin/env python3
"""forecast_service.rhythm_pick: of the forecast samples, the one whose next R peak lands where the recent
beat-to-beat interval puts it is drawn. Made-up ECG on the 8 ms grid (integer µV, as the models' rows)."""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import forecast_service as fs  # noqa: E402

ok = 0


def check(name, cond, detail=""):
    global ok
    print(("  ok  " if cond else "  FAIL ") + name + ("" if cond else f"  {detail}"))
    if not cond:
        raise SystemExit(1)
    ok += 1


def ecg(n, peaks, start=0):
    """n rows (8 ms) from row `start`, R peaks at the given rows: QRS ~1 mV, a T wave 30 rows later."""
    out = []
    for i in range(start, start + n):
        v = 0.0
        for p in peaks:
            d = i - p
            v += 1000 * math.exp(-(d / 1.5) ** 2) - 150 * math.exp(-((d - 3) / 1.2) ** 2) + 220 * math.exp(-((d - 30) / 6) ** 2)
        out.append(int(round(v)))
    return out


RR = 100                                   # rows: 0.8 s, 75 bpm
ring_peaks = [30, 130, 230]                # the 256-row ring: the last beat 25 rows before its end
ring = ecg(256, ring_peaks)
expect = 230 + RR - 255                    # rows after the ring's last row: 75
H = 125


def sample(first):
    return ecg(H, [first + 255, first + 255 + RR], start=256) if first is not None else ecg(H, [], start=256)


early = [sample(f) for f in (60, 55, 62, 50, 75, 58, 52, 64)]       # most of them early, one on time (index 4)
check("picks the sample whose next beat lands where the rhythm says (rows ~75 ahead)", fs.rhythm_pick(ring, early) == 4)
late_first = [sample(f) for f in (90, 76, 100, 110)]
check("…also when the others are late", fs.rhythm_pick(ring, late_first) == 1)
check("too little rhythm in the context (one R peak) -> no pick", fs.rhythm_pick(ecg(256, [200]), early) is None)
check("a flat context -> no pick", fs.rhythm_pick([0] * 256, early) is None)
overdue_ring = ecg(256, [20, 120])         # the next beat (row 220) should already have come
pick = fs.rhythm_pick(overdue_ring, [sample(f) for f in (40, 5, 80)])
check("an overdue beat -> the sample with the soonest beat", pick == 1, pick)
no_beat = [sample(None), sample(75)]
check("a sample with no beat in the horizon loses to the on-time one", fs.rhythm_pick(ring, no_beat) == 1)

lanes = [{"column": "ecg_uv", "int_min": -2500, "int_max": 2500, "scale": 1000, "unit": "mV"}]
vals = [[s] for s in early]
out = fs.summarize(vals, lanes, prefer=4)
check("summarize draws the preferred sample", out["rep_index"] == 4 and out["rep"][0][0] == round(early[4][0] / 1000, 4))
import statistics  # noqa: E402
med = [statistics.median(s[h] for s in early) for h in range(H)]
closest = min(range(len(early)), key=lambda n: sum(abs(early[n][h] - med[h]) for h in range(H)))
check("summarize without a preference: the sample closest to the median, as before",
      fs.summarize(vals, lanes)["rep_index"] == closest)
import subprocess  # noqa: E402
code = "import forecast_service as fs; print(fs.REP_PICK)"
env = {k: v for k, v in os.environ.items() if k != "BIODASH_FORECAST_REP"}
on = subprocess.run([sys.executable, "-c", code], cwd=HERE, env=env, capture_output=True, text=True).stdout.strip()
off = subprocess.run([sys.executable, "-c", code], cwd=HERE, env=dict(env, BIODASH_FORECAST_REP="median"),
                     capture_output=True, text=True).stdout.strip()
check("on by default; BIODASH_FORECAST_REP=median turns it off", on == "rhythm" and off == "median", (on, off))

print(f"{ok} checks passed")
