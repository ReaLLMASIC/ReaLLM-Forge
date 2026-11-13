#!/usr/bin/env python3
"""Bio-dash recordings as an integer CSV, ready for ReaLLM-Forge's data/csv_mc_int.

The same table as interpolate.py (every device on one clock), written the way csv_mc_int
wants it: a whole number in every cell, a fixed range per column, nothing blank.

    ./tools/export_mcint.py                       the most recent day, one row a second
    ./tools/export_mcint.py 2026-10-07 --step 10s
    ./tools/export_mcint.py --from "2026-10-07 22:00" --to "2026-10-08 07:00"
    ./tools/export_mcint.py 2026-10-07 --forge ~/ReaLLM-Forge     ...and build the datasets too

It writes a folder (default <Documents>/Bio-dash/Exports/<day>_<step>_mcint/) with:
    <name>.csv           minute_of_day[,second], then one column per signal
    <name>.columns.json  scale, range and "no reading" value of every column (for decoding)
    <name>.prepare.sh    runs data/csv_mc_int/get_dataset.sh with a --range for each column

Columns and their fixed ranges (so different days share vocabularies):
    hr_bpm 0-250, rr_ms 0-2500, hrv_rmssd_ms 0-500, spo2_pct 0-100, pulse_bpm 0-250,
    bottle_ml 0-5000, sip_ml 0-2000, drunk_today_ml 0-10000,
    co2_ppm 0-10000, pm*_ugm3_x10 0-10000 (0.1 µg/m³), temp_c_x10 -400-850 (0.1 °C),
    rh_pct_x10 0-1000, voc_index / nox_index 0-500, hcho_ppb 0-2000.
Each range has one extra value below it meaning "no reading" (-1 for most): a device that
wasn't recording. Gaps up to --hold are bridged with the last value first.

Options:
    --step 1s          spacing of the rows (1s, 5s, 10s, 1min ...)             (default 1s)
    --max-gap 60s      no straight line across a pause longer than this
    --hold 2min        then carry the last value across gaps up to this long (0 = never)
    --missing mark     mark|drop: what to do with what's still empty (drop = leave the row out)
    --only hr,spo2     keep only signals whose name contains one of these words
    --raw              also every other numeric column (raw.*), ranged from the data
    --real-flags       add <column>_real (1 = a real reading in this step, 0 = filled in)
    --no-time          leave out minute_of_day / second
    --forge DIR        then run the prepare script against this ReaLLM-Forge clone
    --root NAME        output_root under the Forge's data/ (default bio_dash_mcint)
    --out DIR          where to write the folder
    --data-dir DIR     the recordings folder, if it isn't <Documents>/Bio-dash
    --list             show what would go in, and stop

Back to real units: tools/mcint_decode.py <sample.csv> <name>.columns.json
Standard library only.
"""
import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "analysis_dashboard"))
sys.path.insert(0, os.path.join(HERE, "tools"))
import biodata as b  # noqa: E402
import mcint  # noqa: E402
from interpolate import parse_when, span_arg  # noqa: E402


def step_text(step):
    if step < 1000:
        return f"{step}ms"
    return f"{step // 60000}min" if step % 60000 == 0 else f"{step / 1000:g}s"


def main():
    ap = argparse.ArgumentParser(description="Bio-dash recordings as an integer CSV for ReaLLM-Forge csv_mc_int.",
                                 epilog="Full notes: the top of this file.")
    ap.add_argument("day", nargs="?", help="YYYY-MM-DD (local time); default: the most recent day with recordings")
    ap.add_argument("--from", dest="start", type=parse_when)
    ap.add_argument("--to", dest="end", type=parse_when)
    ap.add_argument("--step", type=span_arg, default=1000)
    ap.add_argument("--max-gap", type=span_arg, default=60_000)
    ap.add_argument("--hold", type=span_arg, default=120_000)
    ap.add_argument("--missing", choices=["mark", "drop"], default="mark")
    ap.add_argument("--only", default="")
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--real-flags", action="store_true")
    ap.add_argument("--no-time", action="store_true")
    ap.add_argument("--forge")
    ap.add_argument("--root", default="bio_dash_mcint")
    ap.add_argument("--out")
    ap.add_argument("--data-dir")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if not a.step:
        sys.exit("--step must be more than 0")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", a.root):
        sys.exit("--root: letters, numbers, _ and - only")
    root = b.data_root(a.data_dir)

    if a.start is not None or a.end is not None:
        if a.start is None or a.end is None:
            sys.exit("Give both --from and --to (or a day).")
        if a.end <= a.start:
            sys.exit("--to must be later than --from.")
        start, end = a.start, a.end
        import datetime as dt
        tag = dt.datetime.fromtimestamp(start / 1000).strftime("%Y-%m-%d_%H%M")
    else:
        days = b.days_with_data(root)
        day = a.day or (max(days) if days else None)
        if not day:
            sys.exit(f"No recordings found under {root}. (Use --data-dir if they live elsewhere.)")
        if not re.fullmatch(r"\d{4}-\d\d-\d\d", day):
            sys.exit(f"'{day}' isn't a date (YYYY-MM-DD).")
        start, end, tag = b.local_ms(day), b.local_ms(day, end=True), day

    signals = b.load_signals(root, start, end, extras=a.raw)
    only = [w.strip().lower() for w in a.only.split(",") if w.strip()]
    if only:
        signals = {k: v for k, v in signals.items() if any(w in k.lower() for w in only)}
    if not signals:
        sys.exit("Nothing in that range" + (" matching --only" if only else "") + ".")
    first = min(s.points[0][0] for s in signals.values())
    last = max(s.points[-1][0] for s in signals.values())
    rows = (last - first) // a.step + 1
    if a.list:
        print(f"{len(signals)} signals, up to {rows:,} rows of {step_text(a.step)}")
        for sid, s in sorted(signals.items()):
            col, scale, lo, hi, how = mcint.spec_for(sid, None, s.points)
            print(f"  {sid:36s} -> {col:24s} x{scale:<3d} {lo}..{hi} ({how})")
        return
    if rows * len(signals) > 60_000_000:
        sys.exit(f"That would be {rows:,} rows x {len(signals)} signals. Use a larger --step, a shorter range or --only.")

    tab = b.table(dict(sorted(signals.items())), first, last, step=a.step, max_gap=a.max_gap)
    res = mcint.build(tab, signals, hold_ms=a.hold, missing=a.missing,
                      time_cols=[] if a.no_time else "auto", real_flags=a.real_flags)
    if not res["rows"]:
        sys.exit("No rows left (with --missing drop, every row lacked at least one signal; try --missing mark or --only).")

    name = f"bio-dash_{b.clean_name(tag)}_{step_text(a.step)}_mcint"
    out = a.out or os.path.join(root, "Exports", name)
    os.makedirs(out, exist_ok=True)
    csv_path = os.path.join(out, name + ".csv")
    with open(csv_path, "w", newline="") as f:
        mcint.write_csv(f, res)
    with open(os.path.join(out, name + ".columns.json"), "w") as f:
        f.write(mcint.columns_json(res, source=tag))
    sh = os.path.join(out, name + ".prepare.sh")
    with open(sh, "w") as f:
        f.write(mcint.prepare_sh(res, name + ".csv", a.root))
    os.chmod(sh, 0o755)

    print(f"\n{'column':26s} {'real':>8s} {'estimated':>10s} {'held':>7s} {'no reading':>11s} {'clipped':>8s}")
    for r in res["report"]:
        print(f"  {r['column']:24s} {r['real']:>8,} {r['estimated']:>10,} {r['held']:>7,} {r['no_reading']:>11,} {r['clipped']:>8,}")
    print(f"\nWrote {len(res['rows']):,} rows x {len(res['header'])} columns to {csv_path}")
    if res["dropped"]:
        print(f"Left out {res['dropped']:,} rows where a signal had no reading (--missing drop).")
    if any(r["clipped"] for r in res["report"]):
        print("'clipped': readings outside the column's fixed range, set to its nearest end.")
    print(f"Next: bash {sh} path/to/ReaLLM-Forge")

    if a.forge:
        print()
        sys.exit(subprocess.call(["bash", sh, a.forge, a.root]))


if __name__ == "__main__":
    main()
