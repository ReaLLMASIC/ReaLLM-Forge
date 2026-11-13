#!/usr/bin/env python3
"""Checks for replay_session.py and keeping replays out of exports (standard library only, seconds).

    python3 polar_dashboard/test_replay_session.py

  * frames(): accelerometer frames come out as the strap's 36-sample frames, never 1-2-sample scraps, for
    recordings stamped 5 ms apart (the old worker, frames overlapping a few ms) and for stream_clock.py's
    measured spacing (~4.86 ms: steps of 4 and 5 ms, back-to-back frames); gaps and backward steps start a
    new frame; no sample is lost, added or moved
  * the replay refuses to write into the real recordings unless BIODASH_DATA_DIR is set or
    --into-recordings is given, and writes a session + <time>_replay_info.json when it is set
  * biodata.recordings() (used by export_ecg / export_waves / export_mcint), export_waves --input (folder and
    zip) and fleet.py push's rclone rules leave the replayed session out and keep the real one
"""
import json
import os
import subprocess
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, os.path.join(ROOT, "analysis_dashboard"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)
import replay_session as rs  # noqa: E402

ok = 0


def check(name, got, want=True):
    global ok
    if got != want:
        raise AssertionError(f"{name}: got {got!r}, want {want!r}")
    ok += 1
    print(f"  ok  {name}")


def acc_rows(stamps):
    return [(t, ["1", "2", "3"]) for t in stamps]


def old_clock(frames_n, n=36, t0=1_000_000):
    """The old worker: each frame stamped back from its sensor end time 5 ms apart; the sensor really runs
    at ~205.6 Hz, so consecutive frames overlap by a few ms."""
    out, end = [], t0
    for _ in range(frames_n):
        end += n * 4.864
        out += [int(end - (n - 1 - i) * 5.0) for i in range(n)]
    return out


def new_clock(frames_n, n=36, t0=1_000_000, period=4.864):
    """stream_clock.py: samples spaced at the measured period, frames back to back (int truncation)."""
    out, end = [], t0
    for _ in range(frames_n):
        end += n * period
        out += [int(end - (n - 1 - i) * period) for i in range(n)]
    return out


print("frames()")
for label, stamps in (("old 5 ms stamps", old_clock(40)), ("stream_clock.py stamps", new_clock(40))):
    rows = acc_rows(stamps)
    fr = rs.frames("acc", rows)
    sizes = [len(f) for f in fr]
    check(f"{label}: every sample kept, in order", [r for f in fr for r in f], rows)
    check(f"{label}: frames of at most 36", max(sizes) <= 36)
    check(f"{label}: no 1-2-sample scraps", min(sizes[:-1]) >= 3)
    check(f"{label}: about one frame per 36 samples (not one per sample)", len(fr) <= len(rows) // 36 + 3)
# a real gap (a lost frame) and a backward step start a new frame
stamps = new_clock(3) + [t + 1000 for t in new_clock(2, t0=1_000_000 + 600)]
fr = rs.frames("acc", acc_rows(stamps))
check("a 1 s gap starts a new frame", any(f[0][0] - prev[-1][0] > 500 for prev, f in zip(fr, fr[1:])))
check("no frame spans the gap", all(f[-1][0] - f[0][0] < 400 for f in fr))
back = new_clock(2) + [t - 3000 for t in new_clock(1, t0=1_000_000 + 400)]
fr = rs.frames("acc", acc_rows(back))
check("a backward step starts a new frame", all(f[i][0] >= f[i - 1][0] for f in fr for i in range(1, len(f))))
try:
    import stream_clock  # the clock fix, once it's applied to this release
    anchor = stream_clock.ClockAnchor()
    clock = stream_clock.StreamClock(200, anchor)
    stamps, sensor_ns, now = [], 5_000_000_000, 1_700_000_000_000.0
    for _ in range(60):
        sensor_ns += int(36 * 4.864e6)
        now += 36 * 4.864 + 7
        stamps += clock.stamps(sensor_ns, 36, now_ms=now)
    fr = rs.frames("acc", acc_rows(stamps))
    check("stream_clock.StreamClock stamps: frames of 36, no scraps",
          (max(len(f) for f in fr), min(len(f) for f in fr[:-1]) >= 3), (36, True))
except ImportError:
    print("  (stream_clock.py isn't in this release yet: checked with its formula above)")
check("ECG frames stay 73 samples", [len(f) for f in rs.frames("ecg", acc_rows(range(0, 200 * 8, 8)))],
      [73, 73, 54])

print("the replay's data folder")
with tempfile.TemporaryDirectory() as d:
    src_dir = os.path.join(d, "src", "Polar H10", "Polar H10 0A1B2C3D", "2026-10-14")
    os.makedirs(src_dir)
    t0 = 1_792_000_000_000
    with open(os.path.join(src_dir, "09-00-00_ecg.csv"), "w") as f:
        f.write("Timestamp_Epoch_ms,ECG_mV\n" + "".join(f"{t0 + round(i * 1000 / 130)},0.1\n" for i in range(400)))
    with open(os.path.join(src_dir, "09-00-00_acc.csv"), "w") as f:
        f.write("Timestamp_Epoch_ms,X_mg,Y_mg,Z_mg\n" + "".join(f"{t},1,2,3\n" for t in new_clock(15, t0=t0)))
    src = os.path.join(src_dir, "09-00-00_ecg.csv")
    env = {k: v for k, v in os.environ.items() if k != "BIODASH_DATA_DIR"}
    env["HOME"] = os.path.join(d, "home")              # so a mistake couldn't touch the real recordings
    r = subprocess.run([sys.executable, os.path.join(HERE, "replay_session.py"), src, "--duration-s", "1"],
                       env=env, capture_output=True, text=True, timeout=60)
    check("refuses without BIODASH_DATA_DIR", (r.returncode != 0, "BIODASH_DATA_DIR" in r.stderr), (True, True))
    check("... and wrote nothing", os.path.exists(os.path.join(d, "home")), False)
    data = os.path.join(d, "data")
    env["BIODASH_DATA_DIR"] = data
    r = subprocess.run([sys.executable, os.path.join(HERE, "replay_session.py"), src, "--duration-s", "2"],
                       env=env, capture_output=True, text=True, timeout=60)
    check("replays into BIODASH_DATA_DIR", r.returncode, 0)
    rep_day = os.path.join(data, "Polar H10", "Polar H10 REPLAY")
    day = os.path.join(rep_day, sorted(os.listdir(rep_day))[0])
    names = sorted(os.listdir(day))
    stamp = names[0][:8]
    check("session files + replay_info.json", [n[9:] for n in names],
          ["acc.csv", "ecg.csv", "replay_info.json", "rr.csv"])
    info = json.load(open(os.path.join(day, f"{stamp}_replay_info.json")))
    check("replay_info.json says where it came from", (info["source"], info["shift_ms"] % 8), (src, 0))

    print("exports and fleet.py push leave the replay out")
    # a real session next to the replay, in the same data folder
    real_day = os.path.join(data, "Polar H10", "Polar H10 0A1B2C3D", "2026-10-14")
    os.makedirs(real_day)
    for n in ("09-00-00_ecg.csv", "09-00-00_acc.csv"):
        with open(os.path.join(src_dir, n)) as fi, open(os.path.join(real_day, n), "w") as fo:
            fo.write(fi.read())
    import biodata
    got = {os.path.relpath(x.path, data) for x in biodata.recordings(data, include={"ecg", "acc"})}
    check("biodata.recordings(): the real session only",
          got, {os.path.join("Polar H10", "Polar H10 0A1B2C3D", "2026-10-14", n)
                for n in ("09-00-00_ecg.csv", "09-00-00_acc.csv")})
    every = biodata.recordings(data, include={"ecg", "acc"}, skip_replays=False)
    check("biodata.recordings(skip_replays=False): the replay too", len(every), 5)
    import export_waves
    folder = [s.label for s in export_waves.input_sources([data])]
    check("export_waves --input <folder>: the real session only", [os.path.relpath(x, data) for x in folder],
          [os.path.join("Polar H10", "Polar H10 0A1B2C3D", "2026-10-14", "09-00-00_ecg.csv")])
    zpath = os.path.join(d, "export.zip")
    with zipfile.ZipFile(zpath, "w") as z:
        for base, _dirs, files in os.walk(data):
            for n in files:
                z.write(os.path.join(base, n), os.path.relpath(os.path.join(base, n), data))
    zipped = [s.label for s in export_waves.input_sources([zpath])]
    check("export_waves --input <zip>: the real session only", zipped,
          ["export.zip:Polar H10/Polar H10 0A1B2C3D/2026-10-14/09-00-00_ecg.csv"])
    import fleet
    rules = fleet.rclone_rules(data, None)
    excl = [x for x in rules if "REPLAY" in x]
    check("fleet.py push: one exclude for the replay, before the includes",
          (excl, rules.index(excl[0]) < min(i for i, x in enumerate(rules) if x.startswith("+"))),
          ([f"- /Polar H10/Polar H10 REPLAY/{os.path.basename(day)}/{stamp}_*"], True))
    check("fleet.py push: nothing excluded without replays", any("REPLAY" in x or "_*" in x
                                                                 for x in fleet.rclone_rules(os.path.join(d, "src"), None)), False)

print(f"\n{ok} checks passed")
