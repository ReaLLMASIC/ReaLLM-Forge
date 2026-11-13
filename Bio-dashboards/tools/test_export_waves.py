#!/usr/bin/env python3
"""Checks for waves.py and export_waves.py with made-up H10 recordings. Run: python3 tools/test_export_waves.py"""
import csv
import datetime as dt
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "analysis_dashboard"))
import waves as w  # noqa: E402

ok = 0


def check(name, got, want):
    global ok
    if got != want:
        raise AssertionError(f"{name}: got {got!r}, want {want!r}")
    ok += 1


S = w.Settings(grid_ms=None)                        # these checks cover the ECG-clock method (v1);
E = w.Settings(with_acc=False, grid_ms=None)        # tools/test_waves_grid.py covers the grid method

# ---------------------------------------------------------------- the method's parts, worked by hand
check("columns", [(c["column"], c["vocab_size"]) for c in w.columns(S)],
      [("ecg_uv", 5001), ("acc_x_mg", 4001), ("acc_y_mg", 4001), ("acc_z_mg", 4001)])
check("ECG-only columns", [c["column"] for c in w.columns(E)], ["ecg_uv"])
check("settings survive columns.json", w.Settings.from_dict(json.loads(json.dumps(S.as_dict()))), S)
check("30 s = 3900 rows", S.min_piece_rows, 3900)

a_ts = [0, 10, 20, 80]
a_cols = [[0.0, 100.0, 50.0, 0.0], [0.0, -10.0, -20.0, 0.0], [1000.0] * 4]
check("straight line between the neighbours", w.acc_at(4, a_ts, a_cols, 50), (40.0, -4.0, 1000.0))
check("exactly on a sample", w.acc_at(10, a_ts, a_cols, 50), (100.0, -10.0, 1000.0))
check("exactly on the last sample", w.acc_at(80, a_ts, a_cols, 50), (0.0, 0.0, 1000.0))
check("before the accelerometer starts", w.acc_at(-1, a_ts, a_cols, 50), None)
check("after it ends", w.acc_at(81, a_ts, a_cols, 50), None)
check("neighbours 60 ms apart: none", w.acc_at(50, a_ts, a_cols, 50), None)
check("neighbours exactly 50 ms apart: fine", w.acc_at(5, [0, 50], [[0.0, 10.0]] * 3, 50), (1.0, 1.0, 1.0))
check("one sample isn't enough", w.acc_at(0, [0], [[1.0]] * 3, 50), None)

S2 = w.Settings(ecg_step_uv=2, grid_ms=None)
check("ECG halves round to even (as the first models' data)", [w.ecg_int(v, S2)[0] for v in (0.001, 0.003, 0.005, -0.001)],
      [0, 2, 2, 0])
check("accelerometer halves round up", [w.acc_int(v, S)[0] for v in (2.5, 1.5, -2.5, -0.5)], [3, 2, -2, 0])
check("H10 ECG in whole uV", w.ecg_int(0.123, S), (123, False))
check("ECG clipped at 2.5 mV", [w.ecg_int(v, S) for v in (2.6, -18.0)], [(2500, True), (-2500, True)])
check("accelerometer clipped at 2 g", [w.acc_int(v, S) for v in (2500.0, -2000.4, -2000.6)],
      [(2000, True), (-2000, False), (-2000, True)])
check("decode", w.decode([123, 5, -7, 1000], S), [0.123, 5, -7, 1000])
check("split at a 51 ms step and a step back", w.split_at_gaps([0, 8, 15, 23, 74, 82, 70], 50),
      [(0, 4), (4, 6), (6, 7)])
check("split where", w.split_where(0, 6, [True, True, False, True, False, True]), [(0, 2), (3, 4), (5, 6)])

# reading: a BOM, a run of NUL bytes from a power cut, blank / unreadable cells, an Omni header
T = tempfile.mkdtemp()
p = os.path.join(T, "x_ecg.csv")
with open(p, "wb") as f:
    f.write("\ufeffTimestamp_Epoch_ms,ECG_mV\n1000,0.100\n10\0\0\0\x0008,0.200\n1015,\n1023,nan\n\n1031,-0.300\n"
            .encode())
ts, mv, bad = w.read_ecg(p)
check("ECG read (NULs dropped, bad rows counted)", (list(ts), list(mv), bad), ([1000, 1008, 1031], [0.1, 0.2, -0.3], 2))
p = os.path.join(T, "x_acc.csv")
with open(p, "w") as f:
    f.write("Timestamp_Epoch_ms,X_mg,Y_mg,Z_mg\n10,1,0,0\n5,2,0,0\n5,3,0,0\n20,4,0,0\n")
ts, cols, bad = w.read_acc(p)
check("accelerometer sorted by time, equal times in file order", (list(ts), list(cols[0])),
      ([5, 5, 10, 20], [2.0, 3.0, 1.0, 4.0]))

# ---------------------------------------------------------------- a made-up strap
T0 = int(dt.datetime(2026, 10, 7, 9, 0, 0).timestamp() * 1000)


def ecg_mv(ms):
    """QRS-like spike every 0.8 s on a slow wave, 3 decimals like the H10; one big spike to clip."""
    if 30_000 <= ms < 30_010:
        return 6.0
    return round(0.9 * math.exp(-((ms % 800) - 400) ** 2 / (2 * 12 ** 2)) - 0.1 * math.sin(2 * math.pi * ms / 800), 3)


def make_ecg(segments, phase=0):
    """[(t, mV)]: 130 Hz with whole-ms timestamps (7 or 8 ms apart), segment by segment."""
    out = []
    for start, seconds in segments:
        for i in range(int(seconds * 130)):
            ms = start + phase + int(i * 1000 / 130)
            out.append((T0 + ms, ecg_mv(ms)))
    return out


def make_acc(first, last, holes=()):
    """[(t, x, y, z)]: 200 Hz (5 ms) whole mg, with holes (no samples strictly inside (a, b))."""
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
first_at = lambda ms: next(i for i, t in enumerate(e_ts) if t >= T0 + ms)
b, g, h2 = first_at(20_060), first_at(60_100), first_at(73_000)
inside = sum(1 for t in e_ts if any(T0 + a < t < T0 + z for a, z in HOLES))
check("made-up strap: 2 x 7800 ECG samples", (len(ecg), g), (15600, 7800))

rep = {}
check("pieces: the 60 ms hole and the 3 s hole end pieces, the ECG gap too; short ones left out",
      w.pieces(e_ts, acc_pair, S, rep), [(b, g), (h2, len(e_ts))])
check("rows without accelerometer counted", rep["ecg_rows_without_acc"], inside)
check("ECG only: just the ECG gap", w.pieces(e_ts, None, E), [(0, g), (g, len(e_ts))])
check("no accelerometer file: nothing (with the accelerometer)", w.pieces(e_ts, None, S), [])
batch = [(t, r) for pt, rows in w.recording_rows(e_ts, e_mv, acc_pair, S, rep) for t, r in zip(pt, rows)]
check("clipped: the 6 mV spike's samples, one accelerometer spike", (rep["ecg_clipped"] > 0, rep["acc_clipped"]),
      (True, [1, 0, 0]))


def by_hand(t):
    """The accelerometer at time t worked out without waves.py."""
    lo = max(r for r in acc if r[0] <= t)
    hi = min(r for r in acc if r[0] >= t)
    f = 0.0 if hi[0] == lo[0] else (t - lo[0]) / (hi[0] - lo[0])
    return [math.floor(lo[k] + f * (hi[k] - lo[k]) + 0.5) for k in (1, 2, 3)]


for i in (b, b + 1, b + 1234, h2 + 77):
    t = e_ts[i]
    row = next(r for tt, r in batch if tt == t)
    check(f"row at {t - T0} ms by hand", row, [round(e_mv[i] * 1000)] + by_hand(t))


# ---------------------------------------------------------------- live == offline
def live(ecg, acc, s=S, ecg_frame=73, acc_frame=36, acc_lag=0, **kw):
    """Feed a recording the way a tail of the two CSV files sees it: each file's frames in file
    order, a frame arriving 20 ms after the newest sample so far in its file (the accelerometer
    acc_lag ms later); then group the rows into pieces as a forecaster would."""
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
groups = live(ecg, acc)
check("live pieces start where offline ones do (before the 30 s filter): start, ACC hole, ECG gap, 3 s hole",
      [gr[0][0] - T0 for gr in groups], [e_ts[i] - T0 for i in (0, b, g, h2)])
check("live: accelerometer later than wait_ms loses rows", kept(live(ecg, acc, acc_lag=2500)) == batch, False)
check("... and a longer wait_ms gets them back", kept(live(ecg, acc, acc_lag=2500, wait_ms=3000)), batch)
batch_e = [(t, r) for pt, rows in w.recording_rows(e_ts, e_mv, None, E) for t, r in zip(pt, rows)]
check("live ECG only == offline", kept(live(ecg, [], s=E), E), batch_e)
la = w.LiveAligner(S)
for t, v in ecg[:200]:
    la.push_ecg(t, v)
check("live: the session's first ECG waits for the accelerometer", (la.rows(), len(la.pending)), ([], 200))
for r in acc[:350]:
    la.push_acc(*r)
got = la.rows()
check("... then all of it comes out, one piece", (len(got), [new for _, _, new in got].count(True)), (200, 1))


# the H10's timestamp glitches seen in real recordings (Oct 8-14): frames that overlap the previous
# one by a few ms, and clock re-syncs that step one stream's timestamps back ~1 s from then on
def shift_frames(rows, size, every, shift):
    """Every `every`-th frame of `size` samples shifted by `shift` ms."""
    return [(r[0] + (shift if n % every == every - 1 else 0),) + tuple(r[1:])
            for n, k in enumerate(range(0, len(rows), size)) for r in rows[k:k + size]]


def step_back(rows, size, at_ms, back_ms):
    """From the first frame starting at or after at_ms on, every timestamp is back_ms earlier."""
    i = next(k for k in range(0, len(rows), size) if rows[k][0] >= T0 + at_ms)
    return rows[:i] + [(r[0] - back_ms,) + tuple(r[1:]) for r in rows[i:]]


def offline(ecg, acc, s=S):
    srt = sorted(acc, key=lambda r: r[0])                 # read_acc's stable sort
    pair = ([r[0] for r in srt], [[float(r[k]) for r in srt] for k in (1, 2, 3)])
    ts, mv = [t for t, _ in ecg], [v for _, v in ecg]
    return [(ts[i], r) for i0, i1 in w.pieces(ts, pair, s) for i, r in zip(range(i0, i1), w.piece_rows(i0, i1, ts, mv, pair, s))]


ecg2, acc2 = make_ecg([(0, 120)]), make_acc(-120, 121_500)
acc_ov = shift_frames(acc2, 36, 7, -8)                    # every 7th frame starts 3 ms before the last one ended
check("glitch data: frames overlap by 3 ms", sum(1 for i in range(1, len(acc_ov)) if acc_ov[i][0] < acc_ov[i - 1][0]) > 50, True)
check("live == offline with few-ms frame overlaps", kept(live(ecg2, acc_ov)), offline(ecg2, acc_ov))
check("... (they'd differ without settle_ms)", kept(live(ecg2, acc_ov, settle_ms=0)) == offline(ecg2, acc_ov), False)
ecg_back = step_back(ecg2, 73, 60_000, 1200)
want = offline(ecg_back, acc_ov)
check("ECG clock steps back 1.2 s: a new piece offline", len(want), len(ecg_back))
check("live == offline across an ECG step back (accelerometer history kept)", kept(live(ecg_back, acc_ov)), want)
acc_back = step_back(acc_ov, 36, 60_000, 1100)
want, got = offline(ecg2, acc_back), kept(live(ecg2, acc_back))
diff = sum(1 for x, y in zip(got, want) if x != y)
check("accelerometer steps back 1.1 s: live keeps the first copy, offline mixes them (only that 1.1 s differs)",
      (len(got) == len(want), 0 < diff <= 1.1 * 130 + 5), (True, True))
check("... a settle_ms longer than the step makes them equal", kept(live(ecg2, acc_back, settle_ms=1200)), want)

# ---------------------------------------------------------------- export_waves.py end to end
data = os.path.join(T, "data")


def write(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(header + "\n" + "".join(",".join(str(v) for v in r) + "\n" for r in rows))


strap = "Polar H10 0A1B2C3D"
day = os.path.join(data, "Polar H10", strap, "2026-10-07")
write(os.path.join(day, "09-00-00_ecg.csv"), "Timestamp_Epoch_ms,ECG_mV,HR_BPM", [(t, f"{v:.3f}", 75) for t, v in ecg])
write(os.path.join(day, "09-00-00_acc.csv"), "Timestamp_Epoch_ms,X_mg,Y_mg,Z_mg", acc)
# Omni recorded the same strap for 35 s meanwhile (its own clock anchor: 3 ms off)
omni = make_ecg([(10_000, 35)], phase=3)
odir = os.path.join(data, "Omni", strap, "2026-10-07")
write(os.path.join(odir, "09-00-10_ecg.csv"), "Timestamp_Epoch_ms,ECG_mV", [(t, f"{v:.3f}") for t, v in omni])
write(os.path.join(odir, "09-00-10_acc.csv"), "Timestamp_Epoch_ms,X_mg,Y_mg,Z_mg", make_acc(9_900, 45_200))
# and a 10 s scrap the next day
write(os.path.join(data, "Omni", strap, "2026-10-08", "07-00-00_ecg.csv"), "Timestamp_Epoch_ms,ECG_mV",
      [(t + 86_400_000, f"{v:.3f}") for t, v in make_ecg([(0, 10)])])


def export(*args):
    out = os.path.join(T, "out_" + str(abs(hash(args))))
    p = subprocess.run([sys.executable, os.path.join(HERE, "export_waves.py"), *args, "--ecg-clock",
                        "--data-dir", data, "--out", out, "--name", "t"], capture_output=True, text=True)
    if p.returncode:
        raise AssertionError(f"export_waves {args}: {p.stderr or p.stdout}")
    rows = list(csv.reader(open(os.path.join(out, "t.csv"))))
    info = json.load(open(os.path.join(out, "t.columns.json")))
    return rows, info, out


rows, info, out = export("2026-10-07")
want = (g - b) + (len(e_ts) - h2)
check("export: header", rows[0], ["ecg_uv", "acc_x_mg", "acc_y_mg", "acc_z_mg"])
check("export: the kept rows (Omni's copy dropped)", (len(rows) - 1, info["rows"], info["overlap_rows_dropped"]),
      (want, want, len(omni)))
check("export: rows are the method's rows", [list(map(int, r)) for r in rows[1:]], [r for _, r in batch])
check("export: recordings used", (info["recordings_used"], info["recordings_total"]), (1, 2))
pieces = list(csv.DictReader(open(os.path.join(out, "t_pieces.csv"))))
check("export: pieces index", [(p["first_row"], p["rows"]) for p in pieces], [("0", str(g - b)), (str(g - b), str(len(e_ts) - h2))])
prep = open(os.path.join(out, "t.prepare.sh")).read()
check("export: prepare.sh ranges", all(f"--range {r}" in prep for r in
                                       ("ecg_uv:-2500:2500", "acc_x_mg:-2000:2000", "acc_z_mg:-2000:2000")), True)
check("export: settings recorded", w.Settings.from_dict(info["settings"]), S)
rows_k, info_k, _ = export("2026-10-07", "--keep-overlaps")
check("export --keep-overlaps: Omni's 35 s copy too", len(rows_k) - 1, want + len(omni))
rows_e, info_e, _ = export("2026-10-07", "--ecg-only")
check("export --ecg-only", (rows_e[0], len(rows_e) - 1), (["ecg_uv"], len(e_ts)))
_, info_d, _ = export()
check("export: default is the most recent day", (info_d["range"]["days"], info_d["rows"]), (["2026-10-08"], 0))
rows_i, _, _ = export("--all", "--input", os.path.join(data, "Omni", strap))
check("export --input folder (Omni's copy on its own)", len(rows_i) - 1, len(omni))
zpath = os.path.join(T, "drive.zip")
with zipfile.ZipFile(zpath, "w") as z:
    for name in ("09-00-00_ecg.csv", "09-00-00_acc.csv"):
        z.write(os.path.join(day, name), f"{strap}/2026-10-07/{name}")
rows_z, _, _ = export("--all", "--input", zpath)
check("export --input zip (a Drive download) == the same recording from the data folder", rows_z, rows)
rows_f, info_f, _ = export("--from", "2026-10-07 09:00:25", "--to", "2026-10-07 09:01:30")
check("export --from/--to clips the samples (25-60 s kept; 60.1-70 s and 73-90 s are under 30 s)",
      [list(map(int, r)) for r in rows_f[1:]], [r for t, r in batch if T0 + 25_000 <= t < T0 + 60_100])

shutil.rmtree(T)
print(f"{ok} checks passed")
