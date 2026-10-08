#!/usr/bin/env python3
"""Checks for biodata.py with values worked out by hand. Run: python3 test_biodata.py"""
import datetime as dt
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import biodata as b

T0 = int(dt.datetime(2026, 10, 7, 9, 0, 0).timestamp() * 1000)
ok = 0


def check(name, got, want):
    global ok
    if got != want:
        raise AssertionError(f"{name}: got {got!r}, want {want!r}")
    ok += 1


def sig(kind, pts):
    return b.Signal("x", "heart", "x", "", kind, [(T0 + t, v) for t, v in pts])


# readings: mean / min / max of what falls in each step; empty steps interpolated at their middle
s = sig("readings", [(0, 60), (400, 70), (1000, 80), (5000, 100)])
r = b.resample(s, T0, 6, 1000, max_gap=10_000)
check("mean of a step", r["mean"][0], 65)
check("min / max of a step", (r["min"][0], r["max"][0]), (60, 70))
check("count of real readings", r["n"][:2], [2, 1])
# step 2 is empty; its middle (2500 ms) lies between 1000 (80) and 5000 (100): 80 + 20 * 1500/4000
check("interpolated at the middle", r["mean"][2], 87.5)
check("estimate marked", r["n"][2], 0)
# a gap longer than max_gap stays empty
r = b.resample(s, T0, 6, 1000, max_gap=2000)
check("no estimate across a long gap", r["mean"][2], None)
check("nothing after the last reading", b.resample(s, T0, 8, 1000)["mean"][6], None)

# state: last value in the step, carried forward until hold_max
s = sig("state", [(0, 500), (500, 480), (3000, 400)])
r = b.resample(s, T0, 5, 1000, hold_max=1600)
check("state takes the last value", r["mean"][0], 480)
check("carried forward", r["mean"][1], 480)
check("not beyond hold_max", r["mean"][2], None)            # middle of step 2 is 2500, 2000 after 500
check("next reading", r["mean"][3], 400)

# events: summed per step; 0 only while the device was recording
s = sig("events", [(100, 50), (800, 30), (2500, 120)])
r = b.resample(s, T0, 5, 1000, span=(T0, T0 + 2600))
check("events summed", r["mean"][:4], [80, 0.0, 120, None])
check("event count", r["n"][:3], [2, 0, 1])

# daily total: held for the rest of the day, 0 before the first sip while recording
s = sig("daily_total", [(1000, 100), (3000, 250)])
r = b.resample(s, T0, 5, 1000, span=(T0, T0 + 4999))
check("daily total held", r["mean"], [0.0, 100, 100, 250, 250])

# grid: aligned to whole steps of local time
first, n = b.grid_for(T0 + 61_234, T0 + 3_600_000, 300_000)
check("grid starts on a whole step", first, T0)
check("grid covers the range", n, 13)
check("auto step for a day", b.auto_step(b.DAY_MS), 60_000)
check("step parsing", (b.parse_step("1min"), b.parse_step("250ms"), b.parse_step("auto")), (60000, 250, None))

# RMSSD: alternating 800 / 820 ms -> successive differences all 20 ms
rr = [(T0 + i * 810, 800 if i % 2 else 820) for i in range(60)]
rm = b.rmssd_series(rr)
check("RMSSD", rm[0][1], 20.0)

# sips: a replay within 2 s and ~same volume is dropped; a different sip is kept
check("sip de-duplication", b.dedupe_sips([(0, 100), (1500, 104), (1600, 40), (90_000, 100)]),
      [(0, 100), (1600, 40), (90_000, 100)])
check("running total resets at midnight",
      [v for _, v in b.running_daily_total([(T0, 100), (T0 + 1000, 50), (T0 + b.DAY_MS, 30)])], [100, 150, 30])

# time below a limit: each reading counts until the next (at most 30 s)
check("time below", b.time_below([(0, 89), (60_000, 89), (120_000, 95)], 90), 1.0)

# end to end on a folder: Omni R-R without HR becomes heart rate; CSV marks estimates
with tempfile.TemporaryDirectory() as root:
    day = "2026-10-07"
    def w(rel, text):
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").write(text)
    w(f"Omni/Polar H10 0A1B/{day}/09-00-00_rr.csv", "Timestamp_Epoch_ms,RR_ms\n" +
      "".join(f"{T0 + i * 1000},1000\n" for i in range(5)))
    w(f"Omni/O2Ring 4F21/{day}/09-00-00_vitals.csv", f"Timestamp_Epoch_ms,SpO2_pct,HR_BPM\n{T0},0,60\n{T0 + 2000},97,61\n")
    # a recording with a run of NUL bytes in the middle (power cut while writing) still reads
    w(f"Atmos/Mini/{day}/09-00-00_readings.csv", f"ts,co2\n{T0},600\n" + "\0" * 300 + f"{T0 + 1000},610\n{T0 + 2000},620\n")
    sigs = b.load_signals(root, b.local_ms(day), b.local_ms(day, True))
    check("NUL bytes skipped", [v for _, v in sigs["air.co2"].points], [600, 610, 620])
    check("HR derived from R-R", sigs["heart.hr"].points[0][1], 60.0)
    check("SpO2 of 0 dropped", [v for _, v in sigs["oxygen.spo2"].points], [97])
    tab = b.table({"heart.hr": sigs["heart.hr"]}, T0, T0 + 5999, step=2000)
    f = io.StringIO()
    rows = b.write_csv(f, tab)
    lines = f.getvalue().splitlines()
    check("CSV header", lines[0], "Timestamp_Epoch_ms,Time_Local,heart.hr,heart.hr.n")
    check("CSV rows", rows, 3)
    check("CSV values", lines[1].split(",")[2:], ["60", "2"])

# power-cut protection in the shared recorder (bt_debug.py, the same file in every dashboard)
import time
import bt_debug
with tempfile.TemporaryDirectory() as root:
    os.environ["BIODASH_DATA_DIR"] = root
    d = os.path.join(root, "Omni", "Band-WU 0347", "2026-10-06")
    os.makedirs(d)
    bad, fresh = os.path.join(d, "23-30-52_vitals.csv"), os.path.join(d, "23-59-00_vitals.csv")
    for p in (bad, fresh):
        open(p, "wb").write(b"Timestamp_Epoch_ms,SpO2_pct\n1000,97\n" + b"\0" * 273)
    old = time.time() - 3600
    os.utime(bad, (old, old))                       # fresh stays "being written": left alone
    fixed = bt_debug.repair_recordings("Omni", log=lambda m: None)
    check("repair: only the finished file", [os.path.basename(p) for p, _ in fixed], ["23-30-52_vitals.csv"])
    check("repair: zeros removed, readings kept", open(bad, "rb").read(), b"Timestamp_Epoch_ms,SpO2_pct\n1000,97\n")
    check("repair: active file untouched", open(fresh, "rb").read().count(b"\0"), 273)
    s = bt_debug.SessionFiles("Omni", repair=False)
    s.start("Band-WU 0347", None, {"vitals": ["Timestamp_Epoch_ms", "SpO2_pct"]})
    s._synced = 0
    s.writerows("vitals", [(2000, 96)])              # due for a sync: written to disk now, not just buffered
    check("sync after writing", open(s.paths["vitals"]).read().splitlines()[-1], "2000,96")
    s.close()
    del os.environ["BIODASH_DATA_DIR"]

print(f"{ok} checks passed")
