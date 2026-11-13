"""Bio-dash recordings as an integer CSV for ReaLLM-Forge's `data/csv_mc_int`.

csv_mc_int turns every CSV column into its own multicontext dataset with its own vocabulary:
    token_id = value - int_min        vocab_size = int_max - int_min + 1
so it needs a header row, a whole number in every cell (no blanks, no decimals) and an
inclusive range for every column. This module makes exactly that from the same table the
analysis dashboard charts and exports:

    * each signal becomes one column with a fixed scale and range (SPEC below), so files from
      different days share the same vocabulary and can be trained together. Decimals are kept
      by scaling: SpO2 98 -> 98, temperature 21.4 °C -> 214 (column "temp_c_x10").
    * time becomes minute_of_day (0-1439), plus second (0-59) when the rows are less than a
      minute apart: a model can learn the time of day; an epoch timestamp would need a
      vocabulary of billions.
    * a step with no reading of its own is first filled the usual way (straight line between
      readings, level carried forward); a short gap after that is held at the last value
      (hold_ms); anything still empty gets the column's "no reading" value, one below its
      range (-1 for most columns), so no row is lost and the model can tell "not recording"
      from a real value. missing="drop" leaves those rows out instead.

Alongside the CSV:
    <name>.columns.json   column -> signal, unit, scale, range, missing value (what decode needs)
    <name>.prepare.sh     the get_dataset.sh command with a --range for every column

Standard library only.
"""
import datetime as dt
import json
import math
import re
import shlex

# signal id -> (column, scale, lo, hi): the range is in the column's own (scaled) units.
SPEC = {
    "heart.hr":     ("hr_bpm", 1, 0, 250),
    "heart.rr":     ("rr_ms", 1, 0, 2500),
    "heart.rmssd":  ("hrv_rmssd_ms", 1, 0, 500),
    "oxygen.spo2":  ("spo2_pct", 1, 0, 100),
    "oxygen.pulse": ("pulse_bpm", 1, 0, 250),
    "water.fill":   ("bottle_ml", 1, 0, 5000),
    "water.sips":   ("sip_ml", 1, 0, 2000),
    "water.total":  ("drunk_today_ml", 1, 0, 10000),
}
# Atmos metrics -> (column stem, scale, lo, hi)
AIR_SPEC = {
    "co2":  ("co2_ppm", 1, 0, 10000),
    "pm1":  ("pm1_ugm3_x10", 10, 0, 10000),
    "pm25": ("pm25_ugm3_x10", 10, 0, 10000),
    "pm4":  ("pm4_ugm3_x10", 10, 0, 10000),
    "pm10": ("pm10_ugm3_x10", 10, 0, 10000),
    "temp": ("temp_c_x10", 10, -400, 850),
    "rh":   ("rh_pct_x10", 10, 0, 1000),
    "voc":  ("voc_index", 1, 0, 500),
    "nox":  ("nox_index", 1, 0, 500),
    "hcho": ("hcho_ppb", 1, 0, 2000),
}
TIME_COLUMNS = {"minute_of_day": (0, 1439), "second": (0, 59)}
MAX_VOCAB = 65536            # above this csv_mc_int stores uint32 and the embedding gets huge


def column_name(text):
    """The folder name csv_mc_int will use (it keeps only A-Z a-z 0-9 _ -)."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text.strip()).strip("_").lower() or "col"


def spec_for(sid, sig_meta=None, points=None):
    """(column, scale, lo, hi, how) for a signal. Curated and Atmos signals have fixed specs;
    anything else (raw.*) gets one worked out from its readings (how="from data")."""
    if sid in SPEC:
        return (*SPEC[sid], "fixed")
    if sid.startswith("air."):
        m = re.match(r"([A-Za-z0-9]+)(.*)", sid[4:])
        metric, tail = (m.group(1).lower(), m.group(2).strip("@._")) if m else ("", "")
        if metric in AIR_SPEC:
            stem, scale, lo, hi = AIR_SPEC[metric]
            extra = column_name(tail) if tail else ""     # sensor (and device, when several)
            name = stem if not extra else f"{stem.split('_')[0]}_{extra}_{'_'.join(stem.split('_')[1:])}"
            return (name, scale, lo, hi, "fixed")
    # from the data: whole numbers stay as they are, anything else keeps one decimal (x10)
    vals = [v for _, v in (points or [])]
    whole = all(abs(v - round(v)) < 1e-9 for v in vals)
    scale = 1 if whole else 10
    lo = math.floor(min(vals) * scale) if vals else 0
    hi = math.ceil(max(vals) * scale) if vals else 0
    if hi - lo + 2 > MAX_VOCAB and scale == 10:
        scale, lo, hi = 1, math.floor(min(vals)), math.ceil(max(vals))
    name = column_name(sid[4:] if sid.startswith("raw.") else sid)
    if scale != 1:
        name += f"_x{scale}"
    return (name, scale, 0 if lo >= 0 else lo, hi, "from data")


def build(tab, signals=None, hold_ms=120_000, missing="mark", time_cols="auto", real_flags=False):
    """Integer rows from a biodata.table(). Returns a dict:
        header, rows (lists of int), columns (what decode needs), report (per-column counts)."""
    if missing not in ("mark", "drop"):
        raise ValueError("missing must be 'mark' or 'drop'")
    step, ts, ids = tab["step"], tab["t"], list(tab["series"])
    if time_cols == "auto":
        time_cols = ["minute_of_day"] + (["second"] if step < 60_000 else [])
    time_cols = list(time_cols or [])
    hold_steps = max(0, hold_ms // step) if hold_ms else 0

    cols, used = [], set()
    for sid in ids:
        pts = signals[sid].points if signals and sid in signals else None
        name, scale, lo, hi, how = spec_for(sid, tab["meta"].get(sid), pts)
        base, k = name, 2
        while name in used or name in TIME_COLUMNS:
            name, k = f"{base}_{k}", k + 1
        used.add(name)
        if hi - lo + 2 > MAX_VOCAB:
            raise ValueError(f"{sid}: range {lo}..{hi} is too wide for one vocabulary")
        meta = tab["meta"].get(sid, {})
        cols.append({"column": name, "signal": sid, "label": meta.get("label", sid), "unit": meta.get("unit", ""),
                     "kind": meta.get("kind", "readings"), "scale": scale, "int_min": lo - 1, "int_max": hi,
                     "missing": lo - 1, "range_from": how,
                     "decode": f"value / {scale}" if scale != 1 else "value",
                     "real": 0, "estimated": 0, "held": 0, "no_reading": 0, "clipped": 0})

    # one column of ints per signal, None where nothing is known
    n = len(ts)
    values = []
    for c, sid in zip(cols, ids):
        s = tab["series"][sid]
        out, last, last_k = [None] * n, None, None
        hold = hold_steps if c["kind"] == "readings" else 0   # never repeat a sip; levels already carry
        first_k = next((k for k in range(n) if s["mean"][k] is not None), None)
        for k in range(n):
            v = s["mean"][k]
            if v is not None:
                iv = int(round(v * c["scale"]))
                lo, hi = c["int_min"] + 1, c["int_max"]
                if iv < lo or iv > hi:
                    c["clipped"] += 1
                    iv = min(max(iv, lo), hi)
                out[k], last, last_k = iv, iv, k
                c["real" if s["n"][k] else "estimated"] += 1
            elif hold and last is not None and k - last_k <= hold:
                out[k] = last
                c["held"] += 1
            elif hold and last is None and first_k is not None and first_k - k <= hold:
                out[k] = int(round(s["mean"][first_k] * c["scale"]))   # just before the first reading
                out[k] = min(max(out[k], c["int_min"] + 1), c["int_max"])
                c["held"] += 1
        values.append(out)

    header = list(time_cols) + [c["column"] for c in cols]
    if real_flags:
        header += [c["column"] + "_real" for c in cols]
    rows, dropped, empty = [], 0, 0
    for k, t in enumerate(ts):
        row_vals = [v[k] for v in values]
        if all(v is None for v in row_vals):
            empty += 1                          # nothing recording at all: never a row
            continue
        if missing == "drop" and any(v is None for v in row_vals):
            dropped += 1
            continue
        when = dt.datetime.fromtimestamp(t / 1000)
        row = []
        for tc in time_cols:
            row.append(when.hour * 60 + when.minute if tc == "minute_of_day" else when.second)
        for c, v in zip(cols, row_vals):
            if v is None:
                c["no_reading"] += 1
                v = c["missing"]
            row.append(v)
        if real_flags:
            row += [1 if tab["series"][sid]["n"][k] else 0 for sid in ids]
        rows.append(row)

    columns = [{"column": tc, "signal": "time", "label": tc.replace("_", " "), "unit": "", "scale": 1,
                "int_min": TIME_COLUMNS[tc][0], "int_max": TIME_COLUMNS[tc][1], "missing": None,
                "range_from": "fixed", "decode": "value"} for tc in time_cols]
    for c in cols:
        columns.append({k: c[k] for k in ("column", "signal", "label", "unit", "kind", "scale", "int_min",
                                          "int_max", "missing", "range_from", "decode")})
    if real_flags:
        columns += [{"column": c["column"] + "_real", "signal": c["signal"], "label": "1 = a real reading in this step",
                     "unit": "", "scale": 1, "int_min": 0, "int_max": 1, "missing": None, "range_from": "fixed",
                     "decode": "value"} for c in cols]
    report = [{k: c[k] for k in ("column", "signal", "real", "estimated", "held", "no_reading", "clipped")} for c in cols]
    return {"header": header, "rows": rows, "columns": columns, "report": report, "step": step,
            "first": ts[0] if ts else None, "dropped": dropped, "empty": empty, "missing": missing}


def ranges(result):
    return [f"{c['column']}:{c['int_min']}:{c['int_max']}" for c in result["columns"]]


def write_csv(f, result):
    f.write(",".join(result["header"]) + "\n")
    for row in result["rows"]:
        f.write(",".join(str(v) for v in row) + "\n")
    return len(result["rows"])


def columns_json(result, source=""):
    return json.dumps({
        "format": "bio-dash csv_mc_int v1",
        "made_for": "ReaLLM-Forge data/csv_mc_int (token_id = value - int_min)",
        "source": source,
        "step_ms": result["step"],
        "rows": len(result["rows"]),
        "missing_rows": result["missing"],
        "decode": "real value = value / scale; a value equal to 'missing' means no reading",
        "columns": result["columns"],
    }, indent=2) + "\n"


def prepare_sh(result, csv_name, output_root="bio_dash_mcint"):
    rng = " \\\n  ".join(f"--range {shlex.quote(r)}" for r in ranges(result))
    return f"""#!/usr/bin/env bash
# Turn {csv_name} into one ReaLLM-Forge multicontext dataset per column.
# Run from anywhere; FORGE is your ReaLLM-Forge clone (default: the current folder).
#   bash {csv_name.rsplit('.', 1)[0]}.prepare.sh [path/to/ReaLLM-Forge] [output_root]
# The ranges are fixed per signal, so exports from different days share vocabularies.
set -euo pipefail
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
FORGE="${{1:-.}}"
ROOT="${{2:-{output_root}}}"
[ -f "$FORGE/data/csv_mc_int/get_dataset.sh" ] || {{ echo "No data/csv_mc_int in $FORGE - pass your ReaLLM-Forge folder."; exit 1; }}
bash "$FORGE/data/csv_mc_int/get_dataset.sh" "$HERE/{csv_name}" \\
  --output_root "$ROOT" \\
  --train_ratio 0.9 \\
  {rng}
echo
echo "Train and sample:  CSV_MC_OUTPUT_ROOT=$ROOT bash $FORGE/demos/csv_mc_int_demo.sh $HERE/{csv_name}"
"""


def decode_row(columns, row):
    """Integer row -> real values (None for 'no reading')."""
    out = []
    for c, v in zip(columns, row):
        v = int(v)
        if c.get("missing") is not None and v == c["missing"]:
            out.append(None)
        else:
            out.append(v / c["scale"] if c["scale"] != 1 else v)
    return out
