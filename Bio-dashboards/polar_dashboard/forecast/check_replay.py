#!/usr/bin/env python3
"""Check the live forecast path on a replay: were the rows the service fed the models the training rows?

Replay a recording (replay_session.py) while the service records its contexts:

    python3 forecast_service.py --record /tmp/contexts.jsonl --seed 1
    python3 replay_session.py <Drive export .zip> --member "Polar H10 <strap ID>/<date>/<HH-MM-SS>" \\
        --start-s 900 --duration-s 90

then

    python3 forecast/check_replay.py /tmp/contexts.jsonl \\
        "<Documents>/Bio-dash/Polar H10/Polar H10 REPLAY/<date>/<time>_replay_info.json"

It rebuilds the rows offline with waves.py from the same stretch of the source recording (the grid method
with each model's settings, as for the training data, but no minimum piece length) and compares every
context row the service used, after undoing the replay's time shift. ECG-only rows should be identical;
ECG + accelerometer rows can differ in a few hundredths of a percent (the live aligner settles the
accelerometer after 10 ms; see waves.LiveAligner). It also scores the recorded forecast medians against
what followed (mean |error|), next to repeating the last value. Standard library only.
"""
import argparse
import json
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "analysis_dashboard"))
import waves  # noqa: E402

MODEL_DIRS = {"ecg": "ecg", "acc": "ecg_acc"}


def opener(src, member, stream):
    """Something waves.iter_ecg / iter_acc can read, or None if the recording has no such file."""
    if src.lower().endswith(".zip"):
        z = zipfile.ZipFile(src)
        name = f"{member}_{stream}.csv"
        return (lambda: z.open(name)) if name in z.namelist() else None
    path = src[: -len("_ecg.csv")] + f"_{stream}.csv"
    return path if os.path.exists(path) else None


def to_real(lane, v):
    return v * float(lane.get("step", 1)) / float(lane.get("scale") or 1)


def offline_rows(info, settings):
    """{grid instant (source clock): integer row} for the replayed stretch, as the exporter builds them."""
    t0, t1 = info["t_begin"], info.get("t_end")
    keep = (lambda t: t >= t0 and (t1 is None or t < t1))
    ecg = [(t, v) for t, v in waves.iter_ecg(opener(info["source"], info.get("member"), "ecg")) if keep(t)]
    acc = None
    if settings.with_acc:
        src = opener(info["source"], info.get("member"), "acc")
        rows = [r for r in waves.iter_acc(src) if keep(r[0])] if src else []
        rows.sort(key=lambda r: r[0])                         # stable, like waves.read_acc
        acc = ([r[0] for r in rows], [[r[1 + k] for r in rows] for k in range(3)])
    table = {}
    for times, rows in waves.grid_recording_rows([t for t, _ in ecg], [v for _, v in ecg], acc, settings, {}):
        table.update(zip(times, rows))
    return table


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("record", help="the JSONL file forecast_service.py --record wrote")
    ap.add_argument("info", help="the replay's <time>_replay_info.json")
    ap.add_argument("--models", default=os.path.join(HERE, "models"))
    a = ap.parse_args()
    info = json.load(open(a.info))
    if "t_begin" not in info:
        sys.exit("replay_info.json has no t_begin (made by an older replay_session.py)")
    shift = info["shift_ms"]

    recs = {}
    for line in open(a.record):
        line = line.strip()
        if line:
            r = json.loads(line)
            recs.setdefault(r["kind"], []).append(r)
    if not recs:
        sys.exit(f"no forecasts in {a.record}")

    ok = True
    for kind, items in sorted(recs.items()):
        cfg = json.load(open(os.path.join(a.models, MODEL_DIRS[kind], "config.json")))
        st = dict(cfg["source"]["settings"], min_piece_s=0)
        settings = waves.Settings.from_dict(st)
        lanes = cfg["lanes"]
        dt = settings.grid_ms
        table = offline_rows(info, settings)

        live = {}
        for r in items:
            for t, row in zip(r["times"], r["rows"]):
                live[t] = row
        same = diff = missing = 0
        worst = [0] * len(lanes)
        examples = []
        for t, row in sorted(live.items()):
            off = table.get(t - shift)
            if off is None:
                missing += 1
            elif list(off) == list(row):
                same += 1
            else:
                diff += 1
                worst = [max(w, abs(x - y)) for w, x, y in zip(worst, row, off)]
                if len(examples) < 3:
                    examples.append((t - shift, row, list(off)))
        n = len(live)
        print(f"{kind}: {len(items)} forecasts, {n:,} distinct context rows: {same:,} identical to the offline "
              f"rows ({same / max(1, n):.3%}), {diff:,} different, {missing:,} not in the offline rows")
        if diff:
            print("    largest differences (integer steps): " +
                  ", ".join(f"{ln['column']} {w}" for ln, w in zip(lanes, worst)))
            for t, row, off in examples:
                print(f"    at {t}: live {row} vs offline {off}")

        # The forecasts themselves: median vs what followed, and repeating the last real value
        err = [0.0] * len(lanes)
        hold = [0.0] * len(lanes)
        cnt = 0
        for r in items:
            last = r["rows"][-1]
            med = r["median"]
            for h in range(len(med[0])):
                actual = table.get(r["t0"] + h * dt - shift)
                if actual is None:
                    continue
                for k, ln in enumerate(lanes):
                    v = to_real(ln, actual[k])
                    err[k] += abs(med[k][h] - v)
                    hold[k] += abs(to_real(ln, last[k]) - v)
                cnt += 1
        if cnt:
            parts = [f"{ln['column']} {e / cnt:.4g} {ln.get('unit', '')} (hold {hd / cnt:.4g})"
                     for ln, e, hd in zip(lanes, err, hold)]
            print(f"    forecast median vs what followed, mean |error| over {cnt:,} rows: " + "; ".join(parts))

        if missing or (kind == "ecg" and diff) or diff > 0.005 * max(1, n):
            ok = False
    print("OK: the live rows are the training rows" if ok else "CHECK FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
