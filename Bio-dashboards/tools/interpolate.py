#!/usr/bin/env python3
"""One evenly spaced CSV from Bio-dash recordings: every device on one clock.

Every device records at its own pace (heart rate about once a second, the O2 ring every
second or so, the air monitor and the bottle every few seconds, sips whenever you drink).
This puts them on one clock: one row per time step, one column per signal, each filled the
way that suits it (the same code as the analysis dashboard's export):

    readings (heart rate, SpO2, CO2 ...)  the mean of the real readings in each step; a step
                                          with none gets a straight line between the readings
                                          either side (not across pauses longer than --max-gap)
    state (bottle fill level, battery)    the last value, carried forward
    events (sips)                         mL drunk in each step; plus "water.total", drunk so
                                          far that day

Each signal also gets a ".n" column: how many real readings went into that step. 0 means
the value is an estimate.

    ./tools/interpolate.py                         the most recent day with recordings, 1 s steps
    ./tools/interpolate.py 2026-10-07 --step 1min
    ./tools/interpolate.py --from "2026-10-07 09:00" --to "2026-10-07 12:30"
    ./tools/interpolate.py --list                  what would go in, and stop

Options:
    --step 1s          spacing of the rows: 250ms, 1s, 5s, 1min, 5min ...        (default 1s)
    --max-gap 60s      no straight line across a pause longer than this (0 = always)
    --minmax           add .min / .max columns (what the mean hides inside each step)
    --no-counts        leave out the .n columns
    --only hr,spo2     keep only signals whose name contains one of these words
    --include ecg,acc  also read the waveforms (130 Hz / 50 Hz; use a matching --step)
    --curated          only the named signals (heart.*, oxygen.*, air.*, water.*), no raw.* columns
    --complete         keep only rows where every signal has a value
    --out file.csv     default: <Documents>/Bio-dash/Exports/
    --data-dir DIR     the recordings folder, if it isn't <Documents>/Bio-dash

Standard library only.
"""
import argparse
import datetime as dt
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis_dashboard"))
import biodata as b  # noqa: E402


def parse_when(text):
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            return int(dt.datetime.strptime(text.strip(), fmt).timestamp() * 1000)
        except ValueError:
            pass
    raise argparse.ArgumentTypeError(f"can't read '{text}' as a date/time (try \"2026-10-07 09:00\")")


def span_arg(text):
    if str(text).strip() == "0":
        return 0
    try:
        return b.parse_step(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))


def main():
    ap = argparse.ArgumentParser(description="One evenly spaced CSV from Bio-dash recordings.",
                                 epilog="Full notes: the top of this file.")
    ap.add_argument("day", nargs="?", help="YYYY-MM-DD (local time); default: the most recent day with recordings")
    ap.add_argument("--from", dest="start", type=parse_when)
    ap.add_argument("--to", dest="end", type=parse_when)
    ap.add_argument("--step", type=span_arg, default=1000)
    ap.add_argument("--max-gap", type=span_arg, default=60_000)
    ap.add_argument("--minmax", action="store_true")
    ap.add_argument("--no-counts", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--include", default="")
    ap.add_argument("--curated", action="store_true")
    ap.add_argument("--complete", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--data-dir")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    root = b.data_root(a.data_dir)
    if not a.step:
        sys.exit("--step must be more than 0")

    if a.start is not None or a.end is not None:
        if a.start is None or a.end is None:
            sys.exit("Give both --from and --to (or a day).")
        if a.end <= a.start:
            sys.exit("--to must be later than --from.")
        start, end = a.start, a.end
        tag = dt.datetime.fromtimestamp(start / 1000).strftime("%Y-%m-%d_%H%M")
    else:
        day = a.day or (max(b.days_with_data(root)) if b.days_with_data(root) else None)
        if not day:
            sys.exit(f"No recordings found under {root}. (Use --data-dir if they live elsewhere.)")
        if not re.fullmatch(r"\d{4}-\d\d-\d\d", day):
            sys.exit(f"'{day}' isn't a date (YYYY-MM-DD).")
        start, end, tag = b.local_ms(day), b.local_ms(day, end=True), day

    include = {s.strip().lower() for s in a.include.split(",") if s.strip()}
    signals = b.load_signals(root, start, end, include=include, extras=not a.curated)
    only = [w.strip().lower() for w in a.only.split(",") if w.strip()]
    if only:
        signals = {k: v for k, v in signals.items() if any(w in k.lower() for w in only)}
    if not signals:
        sys.exit("Nothing in that range" + (" matching --only" if only else "") + ".")
    first = min(s.points[0][0] for s in signals.values())
    last = max(s.points[-1][0] for s in signals.values())
    step_txt = f"{a.step} ms" if a.step < 1000 else (f"{a.step // 60000} min" if a.step % 60000 == 0 else f"{a.step / 1000:g} s")
    when = lambda ms: dt.datetime.fromtimestamp(ms / 1000).strftime("%Y-%m-%d %H:%M:%S")
    rows = (last - first) // a.step + 1
    print(f"{len(signals)} signals, {when(first)} -> {when(last)}, one row every {step_txt} (up to {rows:,} rows)")
    if a.list:
        for sid, s in sorted(signals.items()):
            print(f"  {sid:40s} {s.kind:9s} {len(s.points):>9,} readings   {s.label} [{s.unit}]")
        return
    if rows * len(signals) > 60_000_000:
        sys.exit(f"That would be {rows:,} rows x {len(signals)} signals. Use a larger --step, a shorter range or --only.")

    tab = b.table(dict(sorted(signals.items())), first, last, step=a.step, max_gap=a.max_gap)
    out = a.out or os.path.join(root, "Exports", f"bio-dash_{b.clean_name(tag)}_{step_txt.replace(' ', '')}.csv")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", newline="") as f:
        written = b.write_csv(f, tab, minmax=a.minmax, counts=not a.no_counts, complete=a.complete)

    print(f"\n{'signal':40s} {'readings':>9s} {'real steps':>11s} {'estimated':>10s} {'empty':>8s}")
    for sid, s in tab["series"].items():
        real = sum(1 for c in s["n"] if c)
        est = sum(1 for c, v in zip(s["n"], s["mean"]) if not c and v is not None)
        empty = sum(1 for v in s["mean"] if v is None)
        print(f"  {sid:38s} {len(signals[sid].points):>9,} {real:>11,} {est:>10,} {empty:>8,}")
    print(f"\nWrote {written:,} rows to {out}")
    print("'estimated' steps had no reading of their own (interpolated, carried forward, or no sip = 0);"
          " the .n columns mark them row by row.")


if __name__ == "__main__":
    try:
        import signal
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except (ImportError, AttributeError, ValueError):
        pass
    main()
