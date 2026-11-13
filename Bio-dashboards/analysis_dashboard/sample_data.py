#!/usr/bin/env python3
"""Write made-up but realistic Bio-dash recordings, to try the analysis dashboard (or the
interpolation tool) without wearing anything.

    python3 sample_data.py /tmp/biodash-demo              two days ending today
    python3 sample_data.py /tmp/biodash-demo --days 7
    BIODASH_DATA_DIR=/tmp/biodash-demo ./run_dashboard.sh  then look at it

The folder layout and columns are exactly what the dashboards record. Every value is
generated; none of it is anyone's data. Standard library only.
"""
import argparse
import csv
import datetime as dt
import math
import os
import random


def write(path, head, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(head)
        w.writerows(rows)


def ms(d, hour):
    return int((dt.datetime.combine(d, dt.time()) + dt.timedelta(hours=hour)).timestamp() * 1000)


def stamp(d, hour):
    return (dt.datetime.combine(d, dt.time()) + dt.timedelta(hours=hour)).strftime("%H-%M-%S")


def day(root, d, rnd):
    ds = d.isoformat()
    # --- Polar H10 worn 08:30 - 12:30 (desk work, a walk at 11:00)
    rows, t, end = [], ms(d, 8.5), ms(d, 12.5)
    while t < end:
        h = (t - ms(d, 0)) / 3.6e6
        hr = 68 + 6 * math.sin(h * 3) + (28 * math.exp(-((h - 11.1) / 0.18) ** 2)) + rnd.gauss(0, 1.5)
        rr = 60000 / hr + rnd.gauss(0, 18)
        rows.append((t, round(rr), round(hr)))
        t += int(rr)
    write(f"{root}/Polar H10/Polar H10 0A1B2C3D/{ds}/{stamp(d, 8.5)}_rr.csv", ["Timestamp_Epoch_ms", "RR_ms", "HR_BPM"], rows)
    # --- Night (Omni): strap + O2 ring 23:00 -> 07:00 next day, recorded under the evening's date
    rr_rows, vit_rows, t, end = [], [], ms(d, 23), ms(d, 31)
    next_vit = t
    while t < end:
        h = (t - ms(d, 23)) / 3.6e6
        hr = 58 - 4 * math.sin(h / 8 * math.pi) + 3 * math.sin(h * 1.6) + rnd.gauss(0, 1.2)
        rr = 60000 / hr + rnd.gauss(0, 30)
        rr_rows.append((t, round(rr)))
        if t >= next_vit:
            dip = 4 * max(0, math.sin(h * 2.3)) ** 12
            spo2 = round(96.5 - dip + rnd.gauss(0, 0.6))
            vit_rows.append((t, min(100, spo2), round(hr + rnd.gauss(0, 1))))
            next_vit = t + 1000
        t += int(rr)
    write(f"{root}/Omni/Polar H10 0A1B2C3D/{ds}/{stamp(d, 23)}_rr.csv", ["Timestamp_Epoch_ms", "RR_ms"], rr_rows)
    write(f"{root}/Omni/O2Ring 4F21/{ds}/{stamp(d, 23)}_vitals.csv", ["Timestamp_Epoch_ms", "SpO2_pct", "HR_BPM"], vit_rows)
    # --- Atmos all day, every 5 s: CO2 builds up in the evening, drops with a window at 18:00
    rows = []
    for k in range(0, 24 * 720):
        h = k / 720
        co2 = 520 + 380 * max(0, math.sin((h - 7) / 16 * math.pi)) - (250 if 18 <= h < 18.5 else 0) + rnd.gauss(0, 12)
        temp = 21 + 1.8 * math.sin((h - 9) / 24 * 2 * math.pi) + rnd.gauss(0, 0.05)
        rh = 48 - 6 * math.sin((h - 9) / 24 * 2 * math.pi) + rnd.gauss(0, 0.3)
        pm25 = max(0.5, 4 + (22 if 19 <= h < 19.6 else 0) + rnd.gauss(0, 0.8))
        rows.append((ms(d, h), round(co2), round(temp, 2), round(rh, 1), round(pm25, 1), round(100 + rnd.gauss(0, 6))))
    write(f"{root}/Atmos/Atmos-Sphere-S4 (XX-XX-XX-XX-XX-07)/{ds}/{stamp(d, 0)}_readings.csv",
          ["ts", "co2@scd30", "temp@sen55", "rh@sen55", "pm25@sen55", "voc@sen55"], rows)
    # --- HidrateSpark 08:00 - 22:00: sips, refills, fill level every 5 s
    cap, fill, sips, status, events = 621, 600.0, [], [], []
    sip_times = sorted(rnd.uniform(8.2, 21.8) for _ in range(rnd.randint(18, 26)))
    for k in range(0, 14 * 720):
        h = 8 + k / 720
        while sip_times and sip_times[0] <= h:
            sip_times.pop(0)
            ml = round(rnd.uniform(25, 140))
            if fill - ml < 40:
                events.append((ms(d, h - 0.01), "refill", f"weight {round(fill)} -> {cap}"))
                fill = cap
            fill -= ml
            sips.append((ms(d, h), ml, round(100 * ml / cap), "", "live", "pro2", "", "", "", "", "", "weights"))
        status.append((ms(d, h), round(fill), round(100 * fill / cap), "", "", 100 - round(k / 720 * 2), "closed"))
    base = f"{root}/HidrateSpark/h2o0000demo/{ds}/{stamp(d, 8)}"
    write(base + "_sips.csv", ["Timestamp_Epoch_ms", "Volume_ml", "Pct", "Total_Reported", "Source", "Layout", "Raw",
                               "Weight_before", "Weight_after", "Cal_min", "Cal_max", "Volume_from"], sips)
    write(base + "_status.csv", ["Timestamp_Epoch_ms", "Fill_ml", "Fill_pct", "Weight_raw", "Weight_stable_raw",
                                 "Battery_pct", "Cap"], status)
    write(base + "_events.csv", ["Timestamp_Epoch_ms", "Event", "Detail"], events)


def main():
    ap = argparse.ArgumentParser(description="Write made-up Bio-dash recordings for trying things out.")
    ap.add_argument("folder", help="where to write them (use an empty folder, not your real Bio-dash data)")
    ap.add_argument("--days", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    today = dt.date.today()
    for i in range(a.days - 1, -1, -1):
        day(a.folder, today - dt.timedelta(days=i), rnd)
    print(f"Wrote {a.days} day(s) of made-up recordings to {a.folder}")


if __name__ == "__main__":
    main()
