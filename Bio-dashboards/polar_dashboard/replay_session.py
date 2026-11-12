#!/usr/bin/env python3
"""Replay a recorded Polar H10 session as if the strap were connected: test Polar-dash and its live
forecasts without the strap.

Writes a new session folder like the Bluetooth worker does (<data folder>/Polar H10/<device>/<date>/
<time>_{ecg,acc,rr}.csv) in real time: the ECG in 73-sample frames, the accelerometer in the strap's
36-sample frames, the R-R rows as they came, each frame a little after its last sample (as BLE delivers
them). Timestamps move to "now" by a whole number of 8 ms steps, so the 125 Hz grid rows the forecast
service builds are exactly the offline rows, shifted in time (forecast/check_replay.py checks that). The
shift and the source go to <time>_replay_info.json beside the session, which also keeps replays out of
Bio-dash's exports and fleet.py push.

The data folder must be a test folder: set BIODASH_DATA_DIR (for the dashboard, the forecast service and
this script). Without it the replay refuses to write into the real recordings (<Documents>/Bio-dash),
unless --into-recordings is given.

    export BIODASH_DATA_DIR=/tmp/biodash_test
    python3 replay_session.py "<Documents>/Bio-dash/Polar H10/<device>/<date>/<time>_ecg.csv"
    python3 replay_session.py <Drive export .zip> --member "Polar H10 <strap ID>/<date>/<HH-MM-SS>" \\
        [--start-s 900] [--duration-s 120] [--device-name "Polar H10 REPLAY"]
"""
import argparse
import bisect
import csv
import datetime as dt
import io
import json
import os
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.dirname(HERE))   # the shared bt_debug.py, in the release folder
import bt_debug  # noqa: E402

STREAMS = {   # the worker's columns (advanced_worker.SESSION_STREAMS)
    "ecg": ["Timestamp_Epoch_ms", "ECG_mV", "HR_BPM"],
    "acc": ["Timestamp_Epoch_ms", "X_mg", "Y_mg", "Z_mg"],
    "rr": ["Timestamp_Epoch_ms", "RR_ms", "HR_BPM"],
}
ECG_FRAME = 73          # samples per H10 ECG notification at 130 Hz
ACC_FRAME = 36          # samples per H10 accelerometer notification at 200 Hz
BLE_DELAY_MS = 40       # a frame is written this long after its last sample
GRID_MS = 8             # shift by whole grid steps: the 125 Hz rows stay identical


def _file(src, member, stream):
    """(header names, data rows) of one file of the session, or ([], []) if it isn't there."""
    if src.lower().endswith(".zip"):
        z = zipfile.ZipFile(src)
        name = f"{member}_{stream}.csv"
        if name not in z.namelist():
            return [], []
        fh = io.TextIOWrapper(z.open(name), encoding="utf-8-sig", errors="replace", newline="")
    else:
        path = src[: -len("_ecg.csv")] + f"_{stream}.csv"
        if not os.path.exists(path):
            return [], []
        fh = open(path, encoding="utf-8-sig", errors="replace", newline="")
    with fh:
        rd = csv.reader(line.replace("\0", "") for line in fh)      # NUL bytes: a power cut
        head = [h.strip() for h in next(rd, [])]
        return head, [r for r in rd if r]


def _columns(src, member, stream, cols):
    """[(ts, [cells of cols[1:]]), ...] in file order (a missing column gives ''); rows whose time or first
    value doesn't parse are left out."""
    head, data = _file(src, member, stream)
    idx = [head.index(c) if c in head else None for c in cols]
    if idx[0] is None or idx[1] is None:
        return []
    out = []
    for r in data:
        try:
            ts = int(r[idx[0]])
            cells = [r[i].strip() if i is not None and i < len(r) else "" for i in idx[1:]]
            float(cells[0])
        except (ValueError, IndexError):
            continue                        # a repeated header / garbage
        out.append((ts, cells))
    return out


def _int_str(v):
    try:
        return str(int(round(float(v))))
    except ValueError:
        return "0"


def read_session(src, member):
    """The session in the worker's current layout (STREAMS), file order: {stream: [(ts, [value strings])]}.
    Older recordings have no HR_BPM in the ECG and R-R files; it comes from the strap's 1 Hz HR
    (<time>_hrv.csv) or else 60000 / R-R, so the page's HR tiles work as with the strap."""
    ecg = _columns(src, member, "ecg", STREAMS["ecg"])
    acc = _columns(src, member, "acc", STREAMS["acc"])
    rr = _columns(src, member, "rr", STREAMS["rr"])
    hrv = _columns(src, member, "hrv", ["Timestamp_Epoch_ms", "HR_BPM"])
    for _, v in rr:
        rr_ms = float(v[0])
        v[0] = _int_str(v[0])
        v[1] = _int_str(v[1]) if v[1] else (str(round(60000.0 / rr_ms)) if rr_ms > 0 else "0")
    hr = sorted((ts, _int_str(v[0])) for ts, v in hrv) or sorted((ts, v[1]) for ts, v in rr)
    hr_ts = [t for t, _ in hr]
    for ts, v in ecg:
        if v[1]:
            v[1] = _int_str(v[1])
        else:
            k = bisect.bisect_right(hr_ts, ts) - 1
            v[1] = hr[k][1] if k >= 0 else "0"
    return {"ecg": ecg, "acc": acc, "rr": rr}


def frames(stream, rows):
    """Rows grouped about the way the strap delivers them."""
    if stream == "ecg":
        return [rows[i:i + ECG_FRAME] for i in range(0, len(rows), ECG_FRAME)]
    if stream == "acc":
        # Split where the step is more than 1.5 x the median step (a gap) or goes backwards, then cut the runs
        # into the strap's 36-sample frames. This works for recordings stamped 5 ms apart (the old worker: frames
        # overlap by a few ms) and for stream_clock.py's measured spacing (~4.86 ms, so steps of 4 and 5 ms and
        # back-to-back frames whose edges don't show in the timestamps). The old test "step != 5 ms" would cut a
        # new recording at nearly every sample.
        steps = sorted(b[0] - a[0] for a, b in zip(rows, rows[1:]) if b[0] > a[0])
        gap = 1.5 * (steps[len(steps) // 2] if steps else 5)
        out, cur = [], []
        for r in rows:
            if cur and (r[0] - cur[-1][0] > gap or r[0] < cur[-1][0] or len(cur) >= ACC_FRAME):
                out.append(cur)
                cur = []
            cur.append(r)
        return out + ([cur] if cur else [])
    return [[r] for r in rows]              # R-R: one notification per row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="a recording's *_ecg.csv, or a Drive export .zip (with --member)")
    ap.add_argument("--member", help="in a zip: '<device>/<date>/<HH-MM-SS>' (without _ecg.csv)")
    ap.add_argument("--start-s", type=float, default=0.0, help="skip this far into the recording")
    ap.add_argument("--duration-s", type=float, default=0.0, help="stop after this long (0 = to the end)")
    ap.add_argument("--device-name", default="Polar H10 REPLAY")
    ap.add_argument("--into-recordings", action="store_true",
                    help="write into the real recordings folder even though BIODASH_DATA_DIR isn't set")
    a = ap.parse_args()
    if a.source.lower().endswith(".zip") and not a.member:
        ap.error("--member is needed with a zip")
    if not os.environ.get("BIODASH_DATA_DIR") and not a.into_recordings:
        sys.exit(f"Not replaying into your real recordings ({bt_debug.data_root()}): fleet.py push would upload "
                 "the replay, and an export of that day would copy it into training data. Point BIODASH_DATA_DIR "
                 "at a test folder for the dashboard, the forecast service and this script (e.g. "
                 "export BIODASH_DATA_DIR=/tmp/biodash_test), or add --into-recordings if you really want it there.")

    data = read_session(a.source, a.member)
    if not data["ecg"]:
        sys.exit(f"no ECG rows found in {a.source} {a.member or ''}".rstrip())
    t_begin = data["ecg"][0][0] + int(a.start_s * 1000)
    t_end = t_begin + int(a.duration_s * 1000) if a.duration_s > 0 else None
    for s in data:
        data[s] = [r for r in data[s] if r[0] >= t_begin and (t_end is None or r[0] < t_end)]
    first = min(r[0][0] for r in data.values() if r)
    now_ms = int(time.time() * 1000)
    shift = int(round((now_ms - first) / GRID_MS)) * GRID_MS

    events = []
    for s, rows in data.items():
        for fr in frames(s, rows):
            events.append((fr[-1][0] + shift + BLE_DELAY_MS, s, fr))
    events.sort(key=lambda e: e[0])

    session = bt_debug.SessionFiles("Polar H10", repair=False)
    folder = session.start(a.device_name, None, STREAMS, when=dt.datetime.now())
    with open(os.path.join(folder, f"{session.stamp}_replay_info.json"), "w") as f:
        json.dump({"source": os.path.abspath(a.source), "member": a.member, "start_s": a.start_s,
                   "duration_s": a.duration_s, "t_begin": t_begin, "t_end": t_end, "shift_ms": shift,
                   "stamp": session.stamp, "rows": {s: len(r) for s, r in data.items()}}, f, indent=2)
    total_s = (events[-1][0] - events[0][0]) / 1000.0
    print(f"replaying {sum(len(r) for r in data.values()):,} rows ({total_s:.0f} s) into {folder} "
          f"as {session.stamp}_*.csv; timestamps +{shift} ms", flush=True)
    t_report = time.time() + 10
    try:
        for due, s, fr in events:
            wait = due / 1000.0 - time.time()
            if wait > 0:
                time.sleep(wait)
            session.writerows(s, [[ts + shift] + vals for ts, vals in fr])
            session.flush(s)
            if time.time() >= t_report:
                t_report = time.time() + 10
                print(f"  {(due - events[0][0]) / 1000.0:.0f} / {total_s:.0f} s", flush=True)
    except KeyboardInterrupt:
        print("stopped", flush=True)
    finally:
        session.close()
    print("replay finished", flush=True)


if __name__ == "__main__":
    main()
