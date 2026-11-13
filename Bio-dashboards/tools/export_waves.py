#!/usr/bin/env python3
"""The Polar H10's ECG and accelerometer on one even time grid (default 8 ms = 125 Hz), as whole
numbers for ReaLLM-Forge's csv_mc_int: the training data for the live ECG / accelerometer forecasts.

    ./tools/export_waves.py                        the most recent day with ECG
    ./tools/export_waves.py 2026-10-08 2026-10-14  recordings in those day folders
    ./tools/export_waves.py --all                  every recording
    ./tools/export_waves.py --from "2026-10-08 22:00" --to "2026-10-09 07:00"
    ./tools/export_waves.py --ecg-only             just the ECG (one column)
    ./tools/export_waves.py --all --input "Polar H10 <strap ID>-....zip"   a Drive download (zip or folder)
    ./tools/export_waves.py --all --forge ~/ReaLLM-Forge                 and build the datasets there

Writes a folder (default <Documents>/Bio-dash/Exports/bio-dash_<day>_waves/) with:

  <name>.csv            ecg_uv,acc_x_mg,acc_y_mg,acc_z_mg   one row per grid instant (every 8 ms),
                                                            no timestamp: row order is time
  <name>.columns.json   ranges, units, how to decode, the method's settings, what each recording gave
  <name>.prepare.sh     the get_dataset.sh command with a --range for every column
  <name>_pieces.csv     where each unbroken piece starts in the CSV (row, rows, start time, recording)

The method (analysis_dashboard/waves.py; the forecast service uses the same code live):
  * one row every --grid-ms (8 ms), at whole multiples of it in epoch time
  * every channel interpolated in a straight line at each instant: the ECG between its two samples
    either side, each accelerometer axis between its two samples either side
  * an ECG step over 50 ms, or an instant whose two accelerometer neighbours are more than 50 ms
    apart, ends a piece; pieces shorter than 30 s are left out (--min-piece)
  * ECG in whole microvolts clipped to +-2500 (5,001 values), accelerometer in whole milli-g clipped
    to +-2000 (4,001 values per axis)
--ecg-clock instead gives the first ECG models' rows (one row per ECG sample as recorded, ~130 Hz,
the accelerometer interpolated at each ECG sample's time), byte for byte what ReaLLM-Forge's
data/csv_mc_int/ecg_data_conditioning.py gives.

Recordings: Polar-dash (Polar H10/<strap>/<day>/) and Omni (Omni/<strap>/<day>/) both record the
strap. If both ran at once, the later-starting recording only adds what comes after the earlier one
ends (--keep-overlaps keeps both). --input reads zips / folders instead (any layout with
<day>/<time>_ecg.csv and the _acc.csv beside it).

Other options: --device <part of the strap's name>, --out DIR, --name NAME, --data-dir DIR,
--output-root NAME (the dataset folder under ReaLLM-Forge/data/), and the method's numbers:
--grid-ms 8 --max-gap-ms 50 --acc-max-gap-ms 50 --min-piece 30s --ecg-step-uv 1 --ecg-clip-uv 2500
--acc-step-mg 1 --acc-clip-mg 2000. Standard library only.
"""
import argparse
import array
import csv
import datetime as dt
import json
import os
import subprocess
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "analysis_dashboard"))
sys.path.insert(0, os.path.join(HERE, "tools"))
import biodata as b  # noqa: E402
import mcint  # noqa: E402
import waves as w  # noqa: E402
from interpolate import parse_when, span_arg  # noqa: E402

ECG_SUFFIX = "_ecg.csv"
ACC_SUFFIX = "_acc.csv"
SKIP_DIRS = {"Exports", "Reports", "__MACOSX"}


class Source:
    """One recording: its ECG file and the accelerometer file beside it (None if there isn't one)."""
    __slots__ = ("label", "ecg", "acc", "device", "day")

    def __init__(self, label, ecg, acc, device, day):
        self.label, self.ecg, self.acc, self.device, self.day = label, ecg, acc, device, day


def _where(parts):
    """(strap, day) from a recording's folders: .../<strap>/<YYYY-MM-DD>/<time>_ecg.csv."""
    day = parts[-2] if len(parts) >= 2 and b.DAY_RE.match(parts[-2]) else ""
    return (parts[-3] if day and len(parts) >= 3 else ""), day


def _wanted(parts):
    return parts[-1].endswith(ECG_SUFFIX) and not parts[-1].startswith("._") \
        and not SKIP_DIRS.intersection(parts[:-1])


def root_sources(root, start=None, end=None):
    """Every ECG recording in the Bio-dash recordings folder (Polar-dash and Omni)."""
    out = []
    for r in b.recordings(root, start, end, include={"ecg"}):
        if r.stream == "ecg":
            acc = r.path[: -len(ECG_SUFFIX)] + ACC_SUFFIX
            out.append(Source(os.path.relpath(r.path, root), r.path, acc if os.path.isfile(acc) else None,
                              r.device, r.day))
    return out


def input_sources(items):
    """Every *_ecg.csv in the given zips / folders / files, except sessions written by replay_session.py
    (a <time>_replay_info.json beside them: copies of real sessions, often held-out ones)."""
    out, replays = [], []
    for item in items:
        p = os.path.expanduser(item)
        if os.path.isfile(p) and p.lower().endswith(".zip"):
            zf = zipfile.ZipFile(p)
            names = set(zf.namelist())
            for name in sorted(names):
                parts = name.split("/")
                if not _wanted(parts):
                    continue
                if name[: -len(ECG_SUFFIX)] + b.REPLAY_INFO in names:
                    replays.append(f"{os.path.basename(p)}:{name}")
                    continue
                acc = name[: -len(ECG_SUFFIX)] + ACC_SUFFIX
                out.append(Source(f"{os.path.basename(p)}:{name}", lambda zf=zf, n=name: zf.open(n),
                                  (lambda zf=zf, n=acc: zf.open(n)) if acc in names else None, *_where(parts)))
        elif os.path.isdir(p):
            found = []
            for dirpath, dirnames, filenames in os.walk(p):
                dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
                found += [os.path.join(dirpath, f) for f in filenames]
            for f in sorted(found):
                parts = f.split(os.sep)
                if _wanted(parts):
                    if os.path.isfile(f[: -len(ECG_SUFFIX)] + b.REPLAY_INFO):
                        replays.append(f)
                        continue
                    acc = f[: -len(ECG_SUFFIX)] + ACC_SUFFIX
                    out.append(Source(f, f, acc if os.path.isfile(acc) else None, *_where(parts)))
        elif os.path.isfile(p) and p.endswith(ECG_SUFFIX):
            if os.path.isfile(p[: -len(ECG_SUFFIX)] + b.REPLAY_INFO):
                replays.append(p)
                continue
            acc = p[: -len(ECG_SUFFIX)] + ACC_SUFFIX
            out.append(Source(p, p, acc if os.path.isfile(acc) else None, *_where(os.path.abspath(p).split(os.sep))))
        else:
            sys.exit(f"Not a zip, folder or *_ecg.csv: {p}")
    for r in replays:
        print(f"Skipping a replayed session (not a recording): {r}", file=sys.stderr)
    return out


def first_time(src, start, end):
    """The recording's first good ECG timestamp (within [start, end] if given), or None."""
    for t, _ in w.iter_ecg(src.ecg):
        if start is None or start <= t <= end:
            return t
    return None


def subset(ts, mv, keep):
    """The samples whose time passes keep(t), still as arrays."""
    idx = [i for i, t in enumerate(ts) if keep(t)]
    return array.array("q", (ts[i] for i in idx)), array.array("d", (mv[i] for i in idx))


def local(ms, fmt="%Y-%m-%d %H:%M:%S"):
    return dt.datetime.fromtimestamp(ms / 1000).strftime(fmt)


def main():
    ap = argparse.ArgumentParser(description="The H10's ECG + accelerometer on an even time grid (default "
                                             "8 ms = 125 Hz), as an integer CSV for ReaLLM-Forge csv_mc_int.",
                                 epilog="Full notes: the top of this file.")
    ap.add_argument("days", nargs="*", help="YYYY-MM-DD day folders (default: the most recent day with ECG)")
    ap.add_argument("--all", action="store_true", help="every recording")
    ap.add_argument("--from", dest="start", type=parse_when)
    ap.add_argument("--to", dest="end", type=parse_when)
    ap.add_argument("--ecg-only", action="store_true", help="just the ECG column")
    ap.add_argument("--input", nargs="+", help="zip(s) / folder(s) / *_ecg.csv to read instead of the data folder")
    ap.add_argument("--device", help="only the strap whose name contains this")
    ap.add_argument("--keep-overlaps", action="store_true",
                    help="keep both copies when two dashboards recorded the strap at once")
    ap.add_argument("--out", help="output folder (default <data>/Exports/<name>)")
    ap.add_argument("--name", help="file name stem (default bio-dash_<day>_waves)")
    ap.add_argument("--data-dir")
    ap.add_argument("--forge", help="your ReaLLM-Forge folder: also build the datasets there")
    ap.add_argument("--output-root", help="dataset folder under ReaLLM-Forge/data/ (default h10_waves[_ecg])")
    ap.add_argument("--grid-ms", type=float, default=8.0, help="one row every this many ms (default 8 = 125 Hz)")
    ap.add_argument("--ecg-clock", action="store_true",
                    help="the first ECG models' method instead: one row per ECG sample (~130 Hz)")
    ap.add_argument("--max-gap-ms", type=float, default=50.0, help="an ECG step over this ends a piece (50)")
    ap.add_argument("--acc-max-gap-ms", type=float, default=50.0,
                    help="max spacing of the two accelerometer samples around a row's time (50)")
    ap.add_argument("--min-piece", type=span_arg, default=30_000, help="leave out shorter pieces (default 30s)")
    ap.add_argument("--ecg-step-uv", type=int, default=1)
    ap.add_argument("--ecg-clip-uv", type=int, default=2500)
    ap.add_argument("--acc-step-mg", type=int, default=1)
    ap.add_argument("--acc-clip-mg", type=int, default=2000)
    a = ap.parse_args()
    t_run = time.time()
    try:
        s = w.Settings(with_acc=not a.ecg_only, max_gap_ms=a.max_gap_ms, acc_max_gap_ms=a.acc_max_gap_ms,
                       min_piece_s=a.min_piece / 1000.0, ecg_step_uv=a.ecg_step_uv, ecg_clip_uv=a.ecg_clip_uv,
                       acc_step_mg=a.acc_step_mg, acc_clip_mg=a.acc_clip_mg,
                       grid_ms=None if a.ecg_clock else a.grid_ms)
    except ValueError as e:
        sys.exit(str(e))
    root = b.data_root(a.data_dir)

    # which recordings
    start = end = None
    if a.start is not None or a.end is not None:
        if a.start is None or a.end is None or a.end <= a.start:
            sys.exit("Give both --from and --to, --to later than --from.")
        start, end = a.start, a.end
    for d in a.days:
        if not b.DAY_RE.match(d):
            sys.exit(f"'{d}' isn't a date (YYYY-MM-DD).")
    if sum(map(bool, (a.days, a.all, start is not None))) > 1:
        sys.exit("Use one of: day(s), --all, --from/--to.")
    sources = input_sources(a.input) if a.input else root_sources(root, start, end)
    if a.device:
        sources = [x for x in sources if a.device.lower() in (x.device or x.label).lower()]
    days = set(a.days)
    if not (days or a.all or start is not None):
        known = sorted({x.day for x in sources if x.day})
        days = {known[-1]} if known else set()
    if days:
        sources = [x for x in sources if x.day in days]
    if not sources:
        where = " ".join(a.input) if a.input else root
        sys.exit(f"No ECG recordings found in {where}" + (" for those days" if days else "")
                 + (f" for '{a.device}'" if a.device else "") + ".")
    if a.all:
        tag = "all"
    elif start is not None:
        tag = local(start, "%Y-%m-%d_%H%M")
    else:
        tag = min(days) if len(days) == 1 else f"{min(days)}_to_{max(days)}"
    name = a.name or f"bio-dash_{b.clean_name(tag)}_waves" + ("_ecg" if a.ecg_only else "")
    out = a.out or os.path.join(root, "Exports", name)
    os.makedirs(out, exist_ok=True)

    # in time order (by first sample, as ecg_data_conditioning.py orders them), one at a time
    order, report = [], []
    for x in sources:
        try:
            t0 = first_time(x, start, end)
        except ValueError as e:
            report.append({"source": x.label, "rows": 0, "kept": 0, "note": str(e)})
            continue
        if t0 is None:
            report.append({"source": x.label, "rows": 0, "kept": 0, "note": "empty"})
            continue
        order.append((t0, x))
    order.sort(key=lambda r: r[0])

    cols = w.columns(s)
    csv_name = name + ".csv"
    totals = {"rows": 0, "considered": 0, "kept": 0, "ecg_clipped": 0, "acc_clipped": [0, 0, 0], "no_acc": 0,
              "overlap": 0}
    kept_ms, pieces, last_end = 0, [], {}
    with open(os.path.join(out, csv_name), "w", newline="") as f:
        f.write(",".join(c["column"] for c in cols) + "\n")
        for t0, x in order:
            ts, mv, bad = w.read_ecg(x.ecg)
            if start is not None:
                ts, mv = subset(ts, mv, lambda t: start <= t <= end)
            entry = {"source": x.label, "strap": x.device, "start_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0 / 1000)), "rows": len(ts), "skipped_bad_rows": bad}
            strap = x.device.lower()
            if not a.keep_overlaps and strap in last_end and len(ts) and ts[0] <= last_end[strap]:
                n0, after = len(ts), last_end[strap]
                ts, mv = subset(ts, mv, lambda t: t > after)
                entry["overlap_rows_dropped"] = n0 - len(ts)
                totals["overlap"] += n0 - len(ts)
            if len(ts):
                last_end[strap] = max(last_end.get(strap, ts[0]), max(ts))
            totals["rows"] += entry["rows"]
            if len(ts) < 2:
                entry.update({"kept": 0, "note": "empty" if "overlap_rows_dropped" not in entry
                              else "recorded by another dashboard at the same time"})
                report.append(entry)
                continue
            acc = None
            if s.with_acc:
                if x.acc is None:
                    entry["note"] = "no accelerometer file"
                else:
                    try:
                        a_ts, a_cols, a_bad = w.read_acc(x.acc)
                        acc = (a_ts, a_cols)
                        entry.update({"acc_rows": len(a_ts), "acc_skipped_bad_rows": a_bad})
                    except ValueError as e:
                        entry["note"] = str(e)
            rep = {}
            for first, last, n_rows, rows in w.export_pieces(ts, mv, acc, s, rep):
                pieces.append({"first_row": totals["kept"], "rows": n_rows, "start": first, "end": last,
                               "source": x.label})
                kept_ms += last - first
                f.writelines(",".join(map(str, r)) + "\n" for r in rows)
                totals["kept"] += n_rows
            totals["considered"] += rep.get("rows", 0)
            totals["ecg_clipped"] += rep.get("ecg_clipped", 0)
            totals["acc_clipped"] = [p + q for p, q in zip(totals["acc_clipped"], rep.get("acc_clipped", [0, 0, 0]))]
            entry.update({"segments": rep["gap_pieces"], "kept": rep["kept"], "kept_segments": rep["kept_pieces"]})
            if s.grid_ms:
                entry["grid_rows"] = rep["rows"]
            if s.with_acc:
                entry["ecg_rows_without_acc"] = rep["ecg_rows_without_acc"]
                totals["no_acc"] += rep["ecg_rows_without_acc"]
            report.append(entry)

    n = totals["kept"]
    if not n:
        print(f"Nothing long enough to keep (pieces of at least {s.min_piece_s:g} s); "
              f"see {os.path.join(out, name + '.columns.json')}")
    hours = round(n / s.sample_rate_hz / 3600.0, 3)
    rate = round(n / (kept_ms / 1000.0), 2) if kept_ms else None
    with open(os.path.join(out, name + "_pieces.csv"), "w", newline="") as f:
        wr = csv.writer(f, lineterminator="\n")
        wr.writerow(["piece", "first_row", "rows", "seconds", "start_epoch_ms", "start_local", "source"])
        for k, p in enumerate(pieces, 1):
            wr.writerow([k, p["first_row"], p["rows"], round((p["end"] - p["start"]) / 1000.0, 3), p["start"],
                         local(p["start"], "%Y-%m-%d %H:%M:%S.%f")[:-3], p["source"]])
    if s.grid_ms:
        fmt, method = "bio-dash waves v2", (
            f"one row every {s.grid_ms:g} ms ({s.sample_rate_hz:g} Hz) at whole multiples of {s.grid_ms:g} ms in "
            "epoch time; the ECG and each accelerometer axis interpolated linearly at every instant from their "
            "two neighbouring samples; pieces end at ECG steps over max_gap_ms and where the accelerometer "
            "neighbours are over acc_max_gap_ms apart; pieces under min_piece_s left out (Bio-dash "
            "analysis_dashboard/waves.py, grid method)")
    else:
        fmt, method = "bio-dash waves v1", (
            "one row per Polar H10 ECG sample (~130 Hz, not resampled); accelerometer interpolated "
            "linearly at each ECG sample's time; pieces end at ECG steps over max_gap_ms and where the "
            "accelerometer neighbours are over acc_max_gap_ms apart; pieces under min_piece_s left out "
            "(Bio-dash analysis_dashboard/waves.py = ReaLLM-Forge ecg_data_conditioning.py)")
    summary = {
        "format": fmt,
        "made_for": "ReaLLM-Forge data/csv_mc_int (token_id = value - int_min)",
        "method": method,
        "settings": s.as_dict(),
        "csv": csv_name,
        "rows": n,
        "hours": hours,
        "measured_rate_hz": rate,
        "pieces": len(pieces),
        "rows_left_out": (totals["considered"] if s.grid_ms else totals["rows"]) - n,
        "rows_clipped": dict(zip([c["column"] for c in cols], [totals["ecg_clipped"]] + (
            totals["acc_clipped"] if s.with_acc else []))),
        "ecg_rows_without_acc": totals["no_acc"] if s.with_acc else None,
        "overlap_rows_dropped": totals["overlap"],
        "decode": "; ".join(f"{c['column']}: {c['decode']}" for c in cols),
        "columns": cols,
        "range": {"days": sorted(days) or None, "from": start, "to": end, "device": a.device},
        "inputs": a.input or [root],
        "recordings_used": sum(1 for r in report if r.get("kept")),
        "recordings_total": len(report),
        "recordings": sorted(report, key=lambda r: r.get("start_utc", "")),
    }
    with open(os.path.join(out, name + ".columns.json"), "w") as f:
        f.write(json.dumps(summary, indent=2) + "\n")
    output_root = a.output_root or ("h10_waves_ecg" if a.ecg_only else "h10_waves")
    prep = os.path.join(out, name + ".prepare.sh")
    with open(prep, "w") as f:
        f.write(mcint.prepare_sh({"columns": cols}, csv_name, output_root))
    os.chmod(prep, 0o755)

    if order:
        print(f"{n:,} rows ({hours} h) from {summary['recordings_used']} of {len(report)} ECG recording(s), "
              f"{local(order[0][0])} -> {local(order[-1][0])} (first samples)")
    print(f"  {len(pieces):,} unbroken piece(s); left out {summary['rows_left_out']:,} rows (gaps, pieces under "
          f"{s.min_piece_s:g} s" + (", no accelerometer" if s.with_acc else "") + ")"
          + (f"; {totals['overlap']:,} rows recorded twice dropped" if totals["overlap"] else ""))
    clip = f"  clipped: {totals['ecg_clipped']:,} ECG values to +-{s.ecg_clip_uv} uV"
    if s.with_acc:
        clip += (f", {sum(totals['acc_clipped']):,} accelerometer values to +-{s.acc_clip_mg} mg; "
                 f"{totals['no_acc']:,} ECG rows had no accelerometer")
    print(clip + (f"; measured rate {rate} Hz" if rate else ""))
    print(f"  {os.path.join(out, csv_name)}")
    print(f"  + {name}.columns.json, {name}.prepare.sh, {name}_pieces.csv   [{time.time() - t_run:.1f} s]")
    print(f"\ncsv_mc_int:  bash {prep} <ReaLLM-Forge folder>   (datasets in data/{output_root})")
    if a.forge:
        if not n:
            sys.exit("Not building datasets from an empty CSV.")
        cmd = ["bash", prep, os.path.expanduser(a.forge), output_root]
        print("Running: " + " ".join(cmd))
        sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
