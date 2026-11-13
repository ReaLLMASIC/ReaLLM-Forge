#!/usr/bin/env python3
"""forecast_service.ForecastLog: the inference CSV keeps each forecast up to where the next one starts (the rule
the page uses), in Forecasts/<day>/<session time>_forecast_<kind>[_replay].csv, outside the recordings layout.
Standard library only (torch isn't needed)."""
import csv
import os
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
TMP = tempfile.mkdtemp()
os.environ["BIODASH_DATA_DIR"] = os.path.join(TMP, "Bio-dash")
for p in (HERE, os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "analysis_dashboard")):
    sys.path.insert(0, p)
import forecast_service as fs  # noqa: E402
import biodata  # noqa: E402

ok = 0


def check(name, got, want):
    global ok
    good = got == want
    print(("  ok  " if good else "  FAIL ") + name + ("" if good else f": got {got!r}, want {want!r}"))
    if not good:
        sys.exit(1)
    ok += 1


day = os.path.join(TMP, "Bio-dash", "Polar H10", "Polar H10 TEST", "2026-10-22")
os.makedirs(day)
session = os.path.join(day, "09-30-00_ecg.csv")
open(session, "w").write("Timestamp_Epoch_ms,ECG_mV,HR_BPM\n")
ov = types.SimpleNamespace(kind="ecg", lanes=[{"column": "ecg_uv", "unit": "mV"}])
log = fs.ForecastLog(session, ov)
check("file in Forecasts/<day>/", os.path.relpath(log.path, os.path.join(TMP, "Bio-dash")),
      os.path.join("Forecasts", "2026-10-22", "09-30-00_forecast_ecg.csv"))


def forecast(t0, n=125, base=0.0):
    vals = [round(base + k / 1000, 3) for k in range(n)]
    return {"t0": t0, "dt": 8.0, "from": t0 - 8, "rep": [vals], "median": [vals], "lo": [[v - 0.1 for v in vals]],
            "hi": [[v + 0.1 for v in vals]]}


f1 = forecast(1000)
log.write(f1, 1000 + 8 * 62)        # the next forecast starts 62 rows later: rows 0..61 of f1 are kept
log.close()
rows = list(csv.reader(open(log.path)))
check("header", rows[0], ["t_ms", "lead_ms", "ecg_mV", "ecg_mV_median", "ecg_mV_lo", "ecg_mV_hi"])
check("rows kept: up to the next forecast's start", len(rows) - 1, 62)
check("first row: t0, 8 ms ahead", rows[1][:3], ["1000", "8", "0.0"])
check("last row: 62 rows on", rows[-1][:2], ["1488", "496"])

log2 = fs.ForecastLog(session, ov)                  # the same session again: appended, header once
log2.write(forecast(1496), 1496 + 8 * 10)
log2.close()
rows = list(csv.reader(open(log.path)))
check("appended, one header", (len(rows) - 1, sum(r[0] == "t_ms" for r in rows)), (72, 1))

open(os.path.join(day, "09-30-00_replay_info.json"), "w").write("{}")
check("a replayed session is marked", os.path.basename(fs.ForecastLog(session, ov).path), "09-30-00_forecast_ecg_replay.csv")

acc = types.SimpleNamespace(kind="acc", lanes=[{"column": c, "unit": u} for c, u in
                                                (("ecg_uv", "mV"), ("acc_x_mg", "mg"), ("acc_y_mg", "mg"), ("acc_z_mg", "mg"))])
la = fs.ForecastLog(session, acc)
la.close()
check("accelerometer model: every lane", list(csv.reader(open(la.path)))[0][2::4], ["ecg_mV", "acc_x_mg", "acc_y_mg", "acc_z_mg"])

recs = [r.path for r in biodata.recordings(os.path.join(TMP, "Bio-dash"), include=("ecg",), skip_replays=False)]
check("Analysis / the exporters don't see Forecasts/", [p for p in recs if "Forecasts" in p], [])
# a new release folder has no weights (they stay out of git): the same model's from an older folder is used
import json, shutil  # noqa: E402
top = os.path.join(TMP, "Bio-dash-clone")
for v in ("V5.5_Dashboard", "V5.6_Dashboard", "V5.3_Dashboard"):
    os.makedirs(os.path.join(top, v, "polar_dashboard", "forecast", "models", "ecg"))
cfg = {"block_size": 256, "lanes": [{"column": "ecg_uv"}], "transformers_version": "5.20.0"}
for v, c in (("V5.5_Dashboard", dict(cfg, transformers_version="4.53.3")), ("V5.6_Dashboard", cfg),
             ("V5.3_Dashboard", dict(cfg, block_size=512))):
    json.dump(c, open(os.path.join(top, v, "polar_dashboard", "forecast", "models", "ecg", "config.json"), "w"))
for v in ("V5.5_Dashboard", "V5.3_Dashboard"):
    open(os.path.join(top, v, "polar_dashboard", "forecast", "models", "ecg", "model.safetensors"), "wb").write(b"x")
saved = fs.HERE
fs.HERE = os.path.join(top, "V5.6_Dashboard", "polar_dashboard")
mine = os.path.join(fs.HERE, "forecast", "models", "ecg")
found = fs.weights_elsewhere(mine, cfg)
check("weights from the older folder with the same model", os.path.relpath(found, top) if found else None,
      os.path.join("V5.5_Dashboard", "polar_dashboard", "forecast", "models", "ecg"))
check("not from a different model (another config)", fs.weights_elsewhere(mine, dict(cfg, n_layer=12)), None)
fs.HERE = saved
print(f"{ok} checks passed")
