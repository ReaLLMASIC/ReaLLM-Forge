#!/usr/bin/env python3
"""Score the forecast model on recordings, offline: is it beating "hold the last value", and does
centering the context on its median (removing a baseline offset) help?

    PY=~/.local/share/bio-dash/forecast-venv/bin/python        (the forecast Python: tools/orin_setup.sh)
    $PY tools/score_forecast.py ~/Documents/Bio-dash/"Polar H10"/*/2026-10-22/*_ecg.csv
    $PY tools/score_forecast.py <*_ecg.csv ...> --model polar_dashboard/forecast/models/ecg_acc --windows 300

For each recording it builds the model's rows exactly like the live service (waves.py, the model folder's
settings), takes evenly spaced windows (the live context length, then the horizon), forecasts each one
twice -- as the dashboard does today ("plain") and with the ECG context shifted to a median of 0 and the
forecast shifted back ("centred") -- and reports the mean |actual - median forecast| over the first
0.25 / 0.5 / 1 s, next to "hold". Windows are also split by the size of the baseline offset (the context
median). A GPU is used when there is one. Nothing is written anywhere.
"""
import argparse
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))                       # <release>/tools
FC_DIR = os.path.join(os.path.dirname(HERE), "polar_dashboard", "forecast")
DEFAULT_MODEL = os.path.join(FC_DIR, "models", "ecg")


def use_release(model_dir):
    """Import the forecast code and waves.py from the release the model folder is in
    (<release>/polar_dashboard/forecast/models/<model>), so this file can run from anywhere."""
    fc_dir = os.path.dirname(os.path.dirname(os.path.abspath(model_dir)))
    if not os.path.exists(os.path.join(fc_dir, "modeling_csv_mc_int.py")):
        fc_dir = FC_DIR
    rel = os.path.dirname(os.path.dirname(fc_dir))
    for p in (os.path.join(rel, "analysis_dashboard"), rel, fc_dir):
        if p not in sys.path:
            sys.path.insert(0, p)


CONTEXT = 132        # rows of context the live service uses (forecast_service: block - horizon + 1)
TOP_K, TEMPERATURE = 100, 1.0


def load_rows(ecg_path, settings):
    """[(times, rows)] pieces of model rows for one recording (with the accelerometer if the model uses it)."""
    ts, mv, _ = waves.read_ecg(ecg_path)
    acc = None
    if settings.with_acc:
        acc_path = ecg_path[: -len("_ecg.csv")] + "_acc.csv"
        if not os.path.exists(acc_path):
            return []
        a_ts, a_cols, _ = waves.read_acc(acc_path)
        acc = (a_ts, a_cols)
    return list(waves.grid_recording_rows(ts, mv, acc, settings, {}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recordings", nargs="+", help="*_ecg.csv files (the _acc.csv beside each is used too)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="model folder (default: polar_dashboard/forecast/models/ecg)")
    ap.add_argument("--windows", type=int, default=200, help="windows per recording (evenly spaced; default 200)")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--horizon-s", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    a = ap.parse_args()
    use_release(a.model)
    global waves
    import waves
    import torch
    from modeling_csv_mc_int import CsvMcIntForCausalLM
    from fast_forecast import FastForecaster

    cfg = json.load(open(os.path.join(a.model, "config.json")))
    lanes = cfg["lanes"]
    settings = waves.Settings.from_dict(cfg["source"]["settings"])
    dt = float(settings.grid_ms)
    H = int(round(a.horizon_s * 1000 / dt))
    ctx_len = min(CONTEXT, int(cfg["block_size"]) - H + 1)
    dev = torch.device("cuda" if a.device == "cuda" or (a.device == "auto" and torch.cuda.is_available()) else "cpu")
    model = CsvMcIntForCausalLM.load_dir(a.model, map_location=dev)
    fc = FastForecaster(model, num_samples=a.samples, top_k=TOP_K, temperature=TEMPERATURE, dtype=torch.bfloat16)
    lo0, hi0 = lanes[0]["int_min"], lanes[0]["int_max"]
    scale0 = float(lanes[0].get("scale") or 1) / float(lanes[0].get("step", 1))
    marks = [(f"{s:g} s", max(1, int(round(s * 1000 / dt)))) for s in (0.25, 0.5, 1.0) if s <= a.horizon_s + 1e-9]
    print(f"model {os.path.basename(os.path.normpath(a.model))} on {dev}; {ctx_len} rows of context, "
          f"{H} rows ahead, {a.samples} samples; ECG errors in mV (lane {lanes[0]['column']})")

    totals = {"hold": [], "plain": [], "centred": []}
    by_off = {}
    import time
    for ri, path in enumerate(a.recordings, 1):
        t0 = time.time()
        print(f"\n[{ri}/{len(a.recordings)}] reading {os.path.basename(path)} ...", end="", flush=True)
        pieces = load_rows(path, settings)
        print(f" {sum(len(r) for _, r in pieces)} rows in {time.time() - t0:.0f} s", flush=True)
        starts = [(pi, p) for pi, (_, rows) in enumerate(pieces) for p in range(ctx_len, len(rows) - H + 1, 25)]
        if not starts:
            print(f"{path}: too short (or no accelerometer file for a motion model)")
            continue
        step = max(1, len(starts) // a.windows)
        chosen = starts[::step][: a.windows]
        errs = {k: [] for k in totals}
        for wi, (pi, p) in enumerate(chosen, 1):
            if wi % 20 == 0 or wi == len(chosen):
                print(f"\r  forecasting window {wi}/{len(chosen)} ({time.time() - t0:.0f} s)", end="", flush=True)
            rows = pieces[pi][1]
            ctx = [r for r in rows[p - ctx_len: p]]
            truth = [r[0] / scale0 for r in rows[p: p + H]]
            last = ctx[-1][0] / scale0
            off = int(statistics.median(r[0] for r in ctx))
            out = {}
            for name, shift in (("plain", 0), ("centred", off)):
                ids = torch.tensor([[min(hi0, max(lo0, r[0] - shift)) - lo0 for r in ctx]] +
                                   [[r[k] - lanes[k]["int_min"] for r in ctx] for k in range(1, len(lanes))],
                                   dtype=torch.long)
                toks = fc.forecast(ids, H, generator_seed=a.seed)            # (N, K, H)
                ecg = (toks[:, 0, :].float().cpu() + lo0 + shift) / scale0   # (N, H) mV
                out[name] = ecg.median(dim=0).values.tolist()
            row_err = {"hold": [abs(t - last) for t in truth],
                       "plain": [abs(t - m) for t, m in zip(truth, out["plain"])],
                       "centred": [abs(t - m) for t, m in zip(truth, out["centred"])]}
            for k, e in row_err.items():
                errs[k].append(e)
                totals[k].append(e)
            bucket = "|offset| < 0.05 mV" if abs(off / scale0) < 0.05 else "|offset| >= 0.05 mV"
            by_off.setdefault(bucket, {"hold": [], "plain": [], "centred": [], "off": []})
            for k, e in row_err.items():
                by_off[bucket][k].append(e)
            by_off[bucket]["off"].append(off / scale0)
        offs = sorted(statistics.median(r[0] for r in pieces[pi][1][p - ctx_len: p]) / scale0 for pi, p in chosen)
        print(f"\n{os.path.basename(path)}: {len(chosen)} windows; context median offset "
              f"{offs[len(offs) // 2]:+.3f} mV (10-90%: {offs[len(offs) // 10]:+.3f} .. {offs[len(offs) * 9 // 10]:+.3f})")
        report(errs, marks)

    if len(a.recordings) > 1 and totals["hold"]:
        print("\nall recordings")
        report(totals, marks)
    for bucket in sorted(by_off):
        b = by_off[bucket]
        print(f"\nwindows with {bucket} ({len(b['hold'])}, mean offset {statistics.mean(b['off']):+.3f} mV)")
        report(b, marks)


def report(errs, marks):
    print("  " + "".ljust(10) + "".join(m.rjust(10) for m, _ in marks))
    for k in ("hold", "plain", "centred"):
        if not errs[k]:
            return
        vals = [statistics.mean(statistics.mean(e[:n]) for e in errs[k]) for _, n in marks]
        print("  " + k.ljust(10) + "".join(f"{v:10.4f}" for v in vals))


if __name__ == "__main__":
    main()
