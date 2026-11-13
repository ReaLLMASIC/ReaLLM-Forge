#!/usr/bin/env python3
"""Checks for the grid method of waves.py (one row every grid_ms, ECG and accelerometer both
interpolated) and export_waves.py's default output, with made-up H10 recordings.
Run: python3 tools/test_waves_grid.py   (tools/test_export_waves.py covers the ECG-clock method)"""
import csv
import datetime as dt
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "analysis_dashboard"))
import waves as w  # noqa: E402

ok = 0


def check(name, got, want):
    global ok
    if got != want:
        raise AssertionError(f"{name}: got {got!r}, want {want!r}")
    ok += 1


S = w.Settings()                       # the grid method: 8 ms
E = w.Settings(with_acc=False)

# ---------------------------------------------------------------- settings and grid instants
check("default: grid of 8 ms", (S.method, S.grid_ms, S.sample_rate_hz), ("grid", 8, 125.0))
check("30 s = 3750 rows at 125 Hz", S.min_piece_rows, 3750)
check("settings survive columns.json", w.Settings.from_dict(json.loads(json.dumps(S.as_dict()))), S)
old = w.Settings(grid_ms=None).as_dict()
old.pop("grid_ms")
check("settings saved before the grid (no grid_ms) load as the ECG-clock method",
      w.Settings.from_dict(old).method, "ecg-clock")
check("grid 8.0 from JSON stays a whole number", w.Settings(grid_ms=8.0).grid_ms, 8)
check("grid instants in [17, 41]: 24, 32, 40", w.grid_index_range(17, 41, 8), (3, 5))
check("... in (16, 40]: 24, 32, 40", w.grid_index_range(16, 40, 8, include_t0=False), (3, 5))
check("... in [16, 16]: 16", w.grid_index_range(16, 16, 8), (2, 2))
check("negative times round the right way", w.grid_index_range(-17, -1, 8), (-2, -1))
check("ECG between two samples", w.ecg_between(12, 8, 0.0, 16, 1.0), 0.5)
check("ECG exactly on the right sample", w.ecg_between(16, 8, 0.0, 16, 1.0), 1.0)

# ---------------------------------------------------------------- a made-up strap
T0 = int(dt.datetime(2026, 10, 7, 9, 0, 0).timestamp() * 1000)
T0 -= T0 % 8 - 3                       # the recording doesn't start on a grid instant


def ecg_mv(ms):
    if 30_000 <= ms < 30_010:
        return 6.0
    return round(0.9 * math.exp(-((ms % 800) - 400) ** 2 / (2 * 12 ** 2)) - 0.1 * math.sin(2 * math.pi * ms / 800), 3)


def make_ecg(segments, phase=0):
    out = []
    for start, seconds in segments:
        for i in range(int(seconds * 130)):
            ms = start + phase + int(i * 1000 / 130)
            out.append((T0 + ms, ecg_mv(ms)))
    return out


def make_acc(first, last, holes=()):
    out = []
    for ms in range(first, last, 5):
        if any(a < ms < b for a, b in holes):
            continue
        x = round(300 * math.sin(2 * math.pi * ms / 2000)) + (2600 if ms == 50_000 else 0)
        out.append((T0 + ms, x, round(-950 + 20 * math.cos(2 * math.pi * ms / 3000)), round(80 * math.sin(ms / 700))))
    return out


SEGMENTS = [(0, 60), (60_100, 60)]                 # a 100 ms ECG gap after 60 s
HOLES = [(20_000, 20_060), (70_000, 73_000)]       # a 60 ms accelerometer hole and a 3 s one
ecg = make_ecg(SEGMENTS)
acc = make_acc(-120, 120_300, HOLES)
e_ts, e_mv = [t for t, _ in ecg], [v for _, v in ecg]
acc_pair = ([r[0] for r in acc], [[float(r[k]) for r in acc] for k in (1, 2, 3)])
rep = {}
batch = [(t, r) for pt, rows in w.recording_rows(e_ts, e_mv, acc_pair, S, rep) for t, r in zip(pt, rows)]
times = [t for t, _ in batch]
check("every row on a grid instant", all(t % 8 == 0 for t in times), True)
gaps = sorted({b - a for a, b in zip(times, times[1:])})
check("rows 8 ms apart inside pieces (bigger steps only between pieces)", gaps[0], 8)


def ecg_by_hand(t):
    i = next(k for k, tt in enumerate(e_ts) if tt >= t)
    if e_ts[i] == t:
        return e_mv[i]
    t0, t1 = e_ts[i - 1], e_ts[i]
    return e_mv[i - 1] + (t - t0) / (t1 - t0) * (e_mv[i] - e_mv[i - 1])


def acc_by_hand(t):
    lo = max(r for r in acc if r[0] <= t)
    hi = min(r for r in acc if r[0] >= t)
    f = 0.0 if hi[0] == lo[0] else (t - lo[0]) / (hi[0] - lo[0])
    return [math.floor(lo[k] + f * (hi[k] - lo[k]) + 0.5) for k in (1, 2, 3)]


for k in (0, 1, 777, 5000, len(batch) - 1):
    t, row = batch[k]
    check(f"row at {t - T0} ms worked out by hand", row, [round(ecg_by_hand(t) * 1000)] + acc_by_hand(t))

# pieces: the 60 ms hole and the 3 s hole end pieces, the ECG gap too; short ones left out
starts, ends = [batch[0][0]], []
for (ta, _), (tb, _) in zip(batch, batch[1:]):
    if tb - ta != 8:
        ends.append(ta)
        starts.append(tb)
ends.append(batch[-1][0])
first_inst = lambda ms: T0 + ms + (-(T0 + ms)) % 8      # first grid instant at or after T0 + ms
check("pieces: after the 60 ms hole to the ECG gap, then after the 3 s hole to the end",
      [(a - T0, b - T0) for a, b in zip(starts, ends)],
      [(first_inst(20_060) - T0, max(t for t in range(e_ts[0], e_ts[7799] + 1) if t % 8 == 0) - T0),
       (first_inst(73_000) - T0, max(t for t in range(e_ts[-1] - 8, e_ts[-1] + 1) if t % 8 == 0) - T0)])
check("report: rows, kept, pieces", (rep["kept"], rep["kept_pieces"], rep["gap_pieces"]), (len(batch), 2, 2))
check("clipped: the 6 mV spike; the one-sample accelerometer spike falls between instants (0.4 of it: not clipped)",
      (rep["ecg_clipped"] > 0, rep["acc_clipped"]), (True, [0, 0, 0]))
check("ECG only: two pieces (the ECG gap)", [len(rows) > 0 for _, rows in w.recording_rows(e_ts, e_mv, None, E)],
      [True, True])
check("no accelerometer file: nothing", list(w.recording_rows(e_ts, e_mv, None, S)), [])
check("export_pieces gives the same rows", [r for *_, rows in w.export_pieces(e_ts, e_mv, acc_pair, S) for r in rows],
      [r for _, r in batch])

# a straight-line ECG comes out exact at every instant (2 s, under the 2.5 mV clip)
E1 = w.Settings(with_acc=False, min_piece_s=1)
lin = [(T0 + int(i * 1000 / 130), round(0.001 * int(i * 1000 / 130), 3)) for i in range(130 * 2)]
lin_rows = [(t, r) for pt, rows in w.recording_rows([t for t, _ in lin], [v for _, v in lin], None, E1)
            for t, r in zip(pt, rows)]
check("a 1 uV/ms ramp: ECG value = ms since start at every grid instant",
      (len(lin_rows) > 240, all(r[0] == t - T0 for t, r in lin_rows)), (True, True))


# ---------------------------------------------------------------- live == offline
def live(ecg, acc, s=S, ecg_frame=73, acc_frame=36, acc_lag=0, **kw):
    def frames(rows, size, kind, lag):
        out, newest = [], None
        for n, k in enumerate(range(0, len(rows), size)):
            chunk = rows[k:k + size]
            newest = max([r[0] for r in chunk] + ([newest] if newest is not None else []))
            out.append((newest + 20 + lag, kind, n, chunk))
        return out

    la = w.LiveAligner(s, **kw)
    groups = []

    def take():
        for t, row, new in la.rows():
            if new:
                groups.append([])
            groups[-1].append((t, row))

    for _, kind, _, samples in sorted(frames(acc, acc_frame, 0, acc_lag) + frames(ecg, ecg_frame, 1, 0)):
        for smp in samples:
            la.push_acc(*smp) if kind == 0 else la.push_ecg(*smp)
        take()
    la.finish()
    take()
    return groups


kept = lambda groups, s=S: [x for gr in groups if len(gr) >= s.min_piece_rows for x in gr]
for lag in (0, 300, 900, 1500, -400):
    check(f"live rows == offline rows (accelerometer {lag} ms behind)", kept(live(ecg, acc, acc_lag=lag)), batch)
batch_e = [(t, r) for pt, rows in w.recording_rows(e_ts, e_mv, None, E) for t, r in zip(pt, rows)]
check("live ECG only == offline", kept(live(ecg, [], s=E), E), batch_e)
check("live: accelerometer later than wait_ms loses rows", kept(live(ecg, acc, acc_lag=2500)) == batch, False)
check("... and a longer wait_ms gets them back", kept(live(ecg, acc, acc_lag=2500, wait_ms=3000)), batch)
la = w.LiveAligner(E)
la.push_ecg(T0 + 5, 0.0)                         # T0 + 5 is a grid instant (T0 = 3 mod 8)
la.push_ecg(T0 + 13, 0.8)
check("live: a grid instant comes out once the ECG sample after it arrives",
      [(t - T0, r) for t, r, _ in la.rows()], [(5, [0]), (13, [800])])
la.push_ecg(T0 + 17, 1.2)
check("... the instant between two samples is interpolated", [(t - T0, r) for t, r, _ in la.rows()], [])
la.push_ecg(T0 + 25, 2.0)
check("... (21 ms: 1.2 + 4/8 * 0.8 mV)", [(t - T0, r) for t, r, _ in la.rows()], [(21, [1600])])


def shift_frames(rows, size, every, shift):
    return [(r[0] + (shift if n % every == every - 1 else 0),) + tuple(r[1:])
            for n, k in enumerate(range(0, len(rows), size)) for r in rows[k:k + size]]


def step_back(rows, size, at_ms, back_ms):
    i = next(k for k in range(0, len(rows), size) if rows[k][0] >= T0 + at_ms)
    return rows[:i] + [(r[0] - back_ms,) + tuple(r[1:]) for r in rows[i:]]


def offline(ecg, acc, s=S):
    srt = sorted(acc, key=lambda r: r[0])
    pair = ([r[0] for r in srt], [[float(r[k]) for r in srt] for k in (1, 2, 3)])
    ts, mv = [t for t, _ in ecg], [v for _, v in ecg]
    return [(t, r) for pt, rows in w.recording_rows(ts, mv, pair, s) for t, r in zip(pt, rows)]


ecg2, acc2 = make_ecg([(0, 120)]), make_acc(-120, 121_500)
acc_ov = shift_frames(acc2, 36, 7, -8)
check("live == offline with few-ms accelerometer frame overlaps", kept(live(ecg2, acc_ov)), offline(ecg2, acc_ov))
ecg_back = step_back(ecg2, 73, 60_000, 1200)
want = offline(ecg_back, acc_ov)
check("ECG clock steps back 1.2 s: two pieces offline", len(want) > 0, True)
check("live == offline across an ECG step back", kept(live(ecg_back, acc_ov)), want)
acc_back = step_back(acc_ov, 36, 60_000, 1100)
want, got = offline(ecg2, acc_back), kept(live(ecg2, acc_back))
diff = sum(1 for x, y in zip(got, want) if x != y)
check("accelerometer steps back 1.1 s: only that 1.1 s can differ",
      (len(got) == len(want), 0 < diff <= 1.1 * 125 + 5), (True, True))
check("... a settle_ms longer than the step makes them equal", kept(live(ecg2, acc_back, settle_ms=1200)), want)

# ---------------------------------------------------------------- export_waves.py (default: the grid)
T = tempfile.mkdtemp()
data = os.path.join(T, "data")
day = os.path.join(data, "Polar H10", "Polar H10 0A1B2C3D", "2026-10-07")
os.makedirs(day)
with open(os.path.join(day, "09-00-00_ecg.csv"), "w") as f:
    f.write("Timestamp_Epoch_ms,ECG_mV,HR_BPM\n" + "".join(f"{t},{v:.3f},75\n" for t, v in ecg))
with open(os.path.join(day, "09-00-00_acc.csv"), "w") as f:
    f.write("Timestamp_Epoch_ms,X_mg,Y_mg,Z_mg\n" + "".join(",".join(map(str, r)) + "\n" for r in acc))
out = os.path.join(T, "out")
p = subprocess.run([sys.executable, os.path.join(HERE, "export_waves.py"), "2026-10-07", "--data-dir", data,
                    "--out", out, "--name", "t"], capture_output=True, text=True)
if p.returncode:
    raise AssertionError(p.stderr or p.stdout)
rows = list(csv.reader(open(os.path.join(out, "t.csv"))))
info = json.load(open(os.path.join(out, "t.columns.json")))
check("export: header", rows[0], ["ecg_uv", "acc_x_mg", "acc_y_mg", "acc_z_mg"])
check("export: rows are the grid method's rows", [list(map(int, r)) for r in rows[1:]], [r for _, r in batch])
check("export: format and settings", (info["format"], w.Settings.from_dict(info["settings"])), ("bio-dash waves v2", S))
check("export: 125 Hz recorded for the model config", info["settings"]["sample_rate_hz"], 125.0)
pieces = list(csv.DictReader(open(os.path.join(out, "t_pieces.csv"))))
check("export: pieces index", [int(p["rows"]) for p in pieces], [(b - a) // 8 + 1 for a, b in zip(starts, ends)])
shutil.rmtree(T)
print(f"{ok} checks passed")
