#!/usr/bin/env python3
"""Checks for export_ecg.py with made-up ECG (a test tone and QRS-like spikes). Run: python3 tools/test_export_ecg.py"""
import csv
import datetime as dt
import math
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.argv = [sys.argv[0]]
import export_ecg as e  # noqa: E402

ok = 0


def check(name, got, want):
    global ok
    if got != want:
        raise AssertionError(f"{name}: got {got!r}, want {want!r}")
    ok += 1


# resampling: a QRS-like spike every second, 10 ms wide
spike = lambda ms: math.exp(-((ms % 1000) - 500) ** 2 / (2 * 10 ** 2))
n = 130 * 20
vals = [round(spike(i * 1000 / 130), 3) for i in range(n)]
lin = e.resample(vals, 0, n - 1, 0, (n - 1) * 1000 / 130, 200, "linear")
cub = e.resample(vals, 0, n - 1, 0, (n - 1) * 1000 / 130, 200, "cubic")
check("200 Hz over the stretch", len(lin), int((n - 1) / 130 * 200) + 1)
err = lambda out: max(abs(v - spike(k * 5)) for k, v in enumerate(out))
check("linear within 0.06 mV of a 10 ms spike", err(lin) < 0.06, True)
check("cubic closer than linear", err(cub) < err(lin) / 2, True)
check("stretches split on a gap", e.stretches([0, 8, 15, 23, 100, 108], 1000 / 130), [(0, 3), (4, 5)])

# end to end: two recordings of one strap (other phase) merged, a 2 s gap, a 3 s scrap left out
T = tempfile.mkdtemp()
data = os.path.join(T, "data")
t0 = int(dt.datetime(2026, 10, 7, 9, 0, 0).timestamp() * 1000)
tone = lambda ms: math.sin(2 * math.pi * 1.2 * ms / 1000)


def write(path, segs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("Timestamp_Epoch_ms,ECG_mV,HR_BPM\n")
        for s0, cnt in segs:
            for i in range(cnt):
                ms = s0 + i * 1000 / 130
                f.write(f"{int(t0 + ms)},{tone(ms):.3f},72\n")


write(os.path.join(data, "Polar H10", "Polar H10 0A1B2C3D", "2026-10-07", "09-00-00_ecg.csv"),
      [(0, 130 * 60), (62000, 130 * 60), (200000, 130 * 3)])
write(os.path.join(data, "Omni", "Polar H10 0A1B2C3D", "2026-10-07", "09-00-10_ecg.csv"), [(10003, 130 * 20)])
for extra in ([], ["--uv"]):
    out = os.path.join(T, "out" + "".join(extra))
    p = subprocess.run([sys.executable, os.path.join(HERE, "export_ecg.py"), "2026-10-07", "--data-dir", data, "--out", out, *extra],
                       capture_output=True, text=True)
    check("runs " + " ".join(extra), p.returncode, 0)
    raw = list(csv.reader(open(os.path.join(out, "bio-dash_2026-10-07_ecg_raw.csv"))))
    hz = open(os.path.join(out, "bio-dash_2026-10-07_ecg_200hz.csv")).read().split()
    st = list(csv.DictReader(open(os.path.join(out, "bio-dash_2026-10-07_ecg_200hz_stretches.csv"))))
    unit = "ECG_uV" if extra else "ECG_mV"
    check("raw: timestamp + ECG", raw[0], ["Timestamp_Epoch_ms", unit])
    check("raw: every sample once (the second recording's copies dropped)", len(raw) - 1, 130 * 123)
    check("200 Hz: one column, no timestamp", hz[0], unit)
    check("two stretches, the 3 s scrap left out", [s["rows"] for s in st], ["11999", "11999"])
    check("gap before the second", st[1]["gap_before_s"], "2.008")
    if extra:
        check("whole microvolts", all(v.lstrip("-").isdigit() for v in hz[1:]), True)
    else:
        vals = [float(v) for v in hz[1:]]
        worst = max(abs(vals[k] - tone(k * 5)) for k in range(11999))
        check("200 Hz follows the tone", worst < 0.01, True)
# --with-acc: motion at 200 Hz with a 2 s hole at 30-32 s; the first ECG stretch splits around it
acc = os.path.join(data, "Polar H10", "Polar H10 0A1B2C3D", "2026-10-07", "09-00-00_acc.csv")
wave = lambda ms: 1000 * math.sin(2 * math.pi * 0.5 * ms / 1000)
with open(acc, "w") as f:
    f.write("Timestamp_Epoch_ms,X_mg,Y_mg,Z_mg\n")
    for s0, s1 in ((0, 30000), (32000, 125000)):
        for i in range((s1 - s0) // 5):
            ms = s0 + i * 5
            f.write(f"{t0 + ms},{wave(ms):.0f},-3,{981 + (i % 2)}\n")
out = os.path.join(T, "out_acc")
p = subprocess.run([sys.executable, os.path.join(HERE, "export_ecg.py"), "2026-10-07", "--data-dir", data, "--out", out,
                    "--with-acc", "--uv"], capture_output=True, text=True)
check("--with-acc runs", (p.returncode, p.stderr), (0, ""))
base = os.path.join(out, "bio-dash_2026-10-07_ecg_acc")
araw = list(csv.reader(open(base + "_acc_raw.csv")))
check("motion raw: timestamp + X/Y/Z", araw[0], ["Timestamp_Epoch_ms", "X_mg", "Y_mg", "Z_mg"])
check("motion raw: every sample", len(araw) - 1, (30000 + 93000) // 5)
rows = list(csv.reader(open(base + "_200hz.csv")))
check("200 Hz: ECG + motion columns", rows[0], ["ECG_uV", "ACC_X_mg", "ACC_Y_mg", "ACC_Z_mg"])
check("all integers, no blanks", all(len(r) == 4 and all(c.lstrip("-").isdigit() for c in r) for r in rows[1:]), True)
st = list(csv.DictReader(open(base + "_200hz_stretches.csv")))
check("three stretches: 0-30 s, 32-60 s, 62-122 s", [round(int(s["rows"]) / 200) for s in st], [30, 28, 60])
check("motion lines up with the ECG rows", max(abs(int(r[1]) - wave(k * 5)) for k, r in enumerate(rows[1:6000])) <= 1, True)
check("csv_mc_int ranges printed for motion", "--range ACC_Z_mg:-16000:16000" in p.stdout, True)
p = subprocess.run([sys.executable, os.path.join(HERE, "export_ecg.py"), "2026-10-08", "--data-dir", data, "--out", out,
                    "--with-acc"], capture_output=True, text=True)
check("no motion that day: says so", p.returncode != 0, True)
shutil.rmtree(T)
print(f"{ok} checks passed")
