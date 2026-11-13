#!/usr/bin/env python3
"""tools/forecast_timing.py on made-up recordings: a forecast log with a constant offset is called an alignment
problem; one whose timing error grows with the lead (and a perfect one) is called the model's / aligned. The
newest log and its recording are found in a scratch data folder."""
import math
import os
import random
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ok = 0


def check(name, cond, detail=""):
    global ok
    print(("  ok  " if cond else "  FAIL ") + name + ("" if cond else f"\n{detail}"))
    if not cond:
        raise SystemExit(1)
    ok += 1


def beat(t, peaks):
    """mV at time t: a QRS-ish spike at each peak plus a T wave."""
    v = 0.0
    for p in peaks:
        d = t - p
        if -80 < d < 400:
            v += 1.1 * math.exp(-(d / 12) ** 2) - 0.2 * math.exp(-((d - 25) / 10) ** 2) + 0.25 * math.exp(-((d - 250) / 45) ** 2)
    return v


def make(root, stamp, offset_of_lead, seed, dur=240_000.0):
    rnd = random.Random(seed)
    t0 = 1_700_000_000_000.0
    peaks, p = [], t0 + 300
    while p < t0 + dur:
        peaks.append(p)
        p += rnd.uniform(740, 880)                         # ~75 bpm with beat-to-beat variation
    day = "2026-10-22"
    rec = os.path.join(root, "Polar H10", "Polar H10 TEST0001", day)
    os.makedirs(rec, exist_ok=True)
    with open(os.path.join(rec, f"{stamp}_ecg.csv"), "w") as f:
        f.write("Timestamp_Epoch_ms,ECG_mV,HR_BPM\n")
        t = t0
        while t < t0 + dur:
            f.write(f"{int(t)},{beat(t, peaks) + 0.02 * rnd.gauss(0, 1):.4f},75\n")
            t += 1000 / 130
    out = os.path.join(root, "Forecasts", day)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"{stamp}_forecast_ecg.csv"), "w") as f:
        f.write("t_ms,lead_ms,ecg_mV,ecg_mV_median,ecg_mV_lo,ecg_mV_hi\n")
        start = t0 + 3000
        while start < t0 + dur - 1000:                     # a forecast every 0.5 s, each kept for 0.5 s
            shifted = [pk + offset_of_lead(pk - start, rnd) for pk in peaks if start - 500 < pk < start + 1000]
            for h in range(63):
                t = start + 8 * h
                v = beat(t, shifted)
                f.write(f"{int(t)},{8 * h},{v:.4f},{v:.4f},{v - 0.1:.4f},{v + 0.1:.4f}\n")
            start += 504
    return root


def run(root):
    env = dict(os.environ, BIODASH_DATA_DIR=root)
    return subprocess.run([sys.executable, os.path.join(HERE, "forecast_timing.py")], env=env,
                          capture_output=True, text=True)


r = run(make(tempfile.mkdtemp(), "10-00-00", lambda lead, rnd: 40.0, 1))
check("a constant +40 ms offset -> alignment problem (exit 2)", r.returncode == 2 and "an alignment problem" in r.stdout, r.stdout + r.stderr)
check("…and the short-lead offset is reported near +40 ms", "+40 ms" in r.stdout or "+39 ms" in r.stdout or "+41 ms" in r.stdout, r.stdout)

r = run(make(tempfile.mkdtemp(), "11-00-00", lambda lead, rnd: max(0, lead) * -0.15 + rnd.gauss(0, max(2, 0.2 * max(0, lead))), 2))
check("error growing with the lead, early on average -> the model's timing (exit 0)",
      r.returncode == 0 and "the model's beat timing" in r.stdout and "spreads out" in r.stdout, r.stdout + r.stderr)

r = run(make(tempfile.mkdtemp(), "12-00-00", lambda lead, rnd: 0.0, 3))
check("a perfect forecast -> aligned (exit 0)", r.returncode == 0 and "aligned" in r.stdout, r.stdout + r.stderr)

r = run(make(tempfile.mkdtemp(), "13-00-00", lambda lead, rnd: -0.15 * max(0, lead), 4, dur=20_000.0))
check("20 s of forecasting -> 'not enough data', never an alignment verdict from the model's own bias",
      r.returncode == 0 and "not enough data" in r.stdout and "alignment problem" not in r.stdout, r.stdout)

r = run(tempfile.mkdtemp())
check("no forecast log -> a clear message", r.returncode != 0 and "No forecast log" in (r.stdout + r.stderr))

print(f"{ok} checks passed")
