#!/usr/bin/env python3
"""Checks for mcint.py (the ReaLLM-Forge csv_mc_int export), values worked out by hand.
Run: python3 test_mcint.py   (optionally FORGE=path/to/ReaLLM-Forge to also run its real prepare script)"""
import csv
import datetime as dt
import io
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import biodata as b
import mcint

T0 = int(dt.datetime(2026, 10, 7, 9, 0, 0).timestamp() * 1000)
ok = 0


def check(name, got, want):
    global ok
    if got != want:
        raise AssertionError(f"{name}: got {got!r}, want {want!r}")
    ok += 1


def sig(sid, fam, kind, unit, pts):
    return b.Signal(sid, fam, sid, unit, kind, [(T0 + t * 1000, v) for t, v in pts])


# six one-second rows from 09:00:00
signals = {
    "heart.hr": sig("heart.hr", "heart", "readings", "bpm", [(0, 60.4), (1, 61.6), (5, 70)]),
    "air.temp@sen55": sig("air.temp@sen55", "air", "readings", "°C", [(0, 21.44), (1, 21.46), (2, -3.0), (3, 99)]),
    "water.sips": sig("water.sips", "water", "events", "mL", [(2, 150)]),
    "oxygen.spo2": sig("oxygen.spo2", "oxygen", "readings", "%", [(4, 97)]),
}
tab = b.table(signals, T0, T0 + 5000, step=1000, max_gap=2000)
res = mcint.build(tab, signals, hold_ms=1000)
check("header: time, then each signal in table order", res["header"],
      ["minute_of_day", "second", "hr_bpm", "temp_sen55_c_x10", "sip_ml", "spo2_pct"])
cols = {c["column"]: c for c in res["columns"]}
check("fixed range plus one 'no reading' value below it", (cols["hr_bpm"]["int_min"], cols["hr_bpm"]["int_max"]), (-1, 250))
check("temperature kept to 0.1 by scaling", (cols["temp_sen55_c_x10"]["scale"], cols["temp_sen55_c_x10"]["int_min"]), (10, -401))
check("time columns", (cols["minute_of_day"]["int_max"], cols["second"]["int_max"]), (1439, 59))
rows = res["rows"]
check("six rows", len(rows), 6)
check("09:00:00 -> minute 540, second 0", rows[0][:2], [540, 0])
check("hr rounded to whole bpm", [r[2] for r in rows[:2]], [60, 62])
# hr: 1 s (61.6) then nothing until 5 s; 4 s gap > max_gap 2 s so 2-4 s are empty; hold 1 s fills 2 s only
check("short gap held, longer gap marked -1", [r[2] for r in rows[2:]], [62, -1, -1, 70])
check("temperature 21.44 -> 214, -3.0 -> -30", [r[3] for r in rows[:3]], [214, 215, -30])
check("out of range clipped to the top (99 C -> 850)", rows[3][3], 850)
check("clip counted", next(r["clipped"] for r in res["report"] if r["column"] == "temp_sen55_c_x10"), 1)
# sips: events inside the water family's recording span are 0, outside it unknown; never held
check("one sip, nothing repeated", [r[4] for r in rows], [-1, -1, 150, -1, -1, -1])
check("spo2 back-filled 1 s before its first reading only", [r[5] for r in rows], [-1, -1, -1, 97, 97, 97])
check("every cell a whole number", all(isinstance(v, int) for r in rows for v in r), True)

# CSV text, ranges and decoding
f = io.StringIO()
mcint.write_csv(f, res)
lines = f.getvalue().splitlines()
check("csv header", lines[0], "minute_of_day,second,hr_bpm,temp_sen55_c_x10,sip_ml,spo2_pct")
check("csv first row", lines[1], "540,0,60,214,-1,-1")
check("every value fits its declared range",
      all(c["int_min"] <= int(v) <= c["int_max"] for line in lines[1:] for c, v in zip(res["columns"], line.split(","))), True)
check("ranges for get_dataset.sh", mcint.ranges(res)[2:4], ["hr_bpm:-1:250", "temp_sen55_c_x10:-401:850"])
check("decode: scale back, -1 -> None", mcint.decode_row(res["columns"], lines[1].split(",")), [540, 0, 60, 21.4, None, None])
spec = json.loads(mcint.columns_json(res))
check("columns.json lists every column", [c["column"] for c in spec["columns"]], res["header"])
check("names are what csv_mc_int folders will be", all(mcint.column_name(h) == h for h in res["header"]), True)

# missing="drop" leaves rows out instead; real flags; no time
res2 = mcint.build(tab, signals, hold_ms=1000, missing="drop", time_cols=[], real_flags=True)
check("drop: only rows where every signal has a value", len(res2["rows"]) + res2["dropped"], 6)
check("drop: no -1 left", any(v == -1 for r in res2["rows"] for v in r[:4]), False)
check("real flags appended", res2["header"][-1], "spo2_pct_real")
# a row with nothing at all from any device is never written
far = {"heart.hr": sig("heart.hr", "heart", "readings", "bpm", [(0, 60), (10, 60)])}
res3 = mcint.build(b.table(far, T0, T0 + 10_000, step=1000, max_gap=1000), far, hold_ms=0)
check("empty rows skipped", (len(res3["rows"]), res3["empty"]), (2, 9))

# step of a minute or more: no 'second' column
res4 = mcint.build(b.table(far, T0, T0 + 10_000, step=60_000, max_gap=0), far)
check("minute steps: minute_of_day only", res4["header"][:2], ["minute_of_day", "hr_bpm"])

# spec for unknown (raw) columns comes from the data
check("raw whole numbers", mcint.spec_for("raw.Omni.vitals.PI", None, [(0, 3), (1, 12)])[:4], ("omni_vitals_pi", 1, 0, 12))
check("raw decimals kept x10", mcint.spec_for("raw.Polar.acc.X_mg", None, [(0, -1.25), (1, 2.5)])[:4], ("polar_acc_x_mg_x10", 10, -13, 25))
check("air with a device", mcint.spec_for("air.co2@scd30.Atmos-1")[0], "co2_scd30_atmos-1_ppm")

# the prepare script, run against ReaLLM-Forge's own get_dataset.sh when FORGE is set
forge = os.environ.get("FORGE")
if forge and os.path.isfile(os.path.join(forge, "data", "csv_mc_int", "get_dataset.sh")):
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "t.csv"), "w").write(f.getvalue())
        open(os.path.join(d, "t.prepare.sh"), "w").write(mcint.prepare_sh(res, "t.csv", "bio_dash_mcint_test"))
        r = subprocess.run(["bash", os.path.join(d, "t.prepare.sh"), forge], capture_output=True, text=True)
        check("ReaLLM-Forge accepts it", r.returncode, 0)
        man = json.load(open(os.path.join(forge, "data", "bio_dash_mcint_test", "manifest.json")))
        check("one dataset per column", len(man["multicontext_datasets"]), 6)
        import shutil
        shutil.rmtree(os.path.join(forge, "data", "bio_dash_mcint_test"))

print(f"{ok} checks passed")
