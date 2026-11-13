#!/usr/bin/env python3
"""Pretend to be a device's Bluetooth worker: made-up readings, written exactly like the real
worker writes them (recordings, status.json, scan results). `server.py --demo` runs one per
device so the app can be tried, tested and screenshotted without wearing anything.

    python3 tools/demo_feed.py heart|oxygen|brain|air|water [person 0-2]
    python3 tools/demo_feed.py forecast <heart API url>       made-up forecasts, like forecast_service.py posts

Recordings go to BIODASH_DATA_DIR and the state files to BIODASH_STATE_DIR (server.py --demo points both
at a scratch folder). Every value is generated; none of it is anyone's data. Addresses are fake on purpose.
"""
import json
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import bt_debug  # noqa: E402

KEY = sys.argv[1] if len(sys.argv) > 1 else "heart"
WHO = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 0      # which person's (they differ a little)
rnd = random.Random(11 + WHO)
TAG = f"{WHO + 1:04d}"


def state(folder, *parts):
    """A state file's path: in BIODASH_STATE_DIR, else the device's own folder (as the real workers)."""
    return os.path.join(bt_debug.state_dir(os.path.join(HERE, folder)), *parts)


ADAPTERS = [{"name": "hci0", "address": "DEMO-RADIO-0", "alias": "demo", "powered": True, "vendor": "Intel",
             "placement": "internal"},
            {"name": "hci1", "address": "DEMO-RADIO-1", "alias": "demo", "powered": True, "vendor": "Realtek",
             "usb_id": "2357:0604", "placement": "external"}]


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f)
    os.replace(path + ".tmp", path)


def debug(name, addr, hci="hci0", session=None):
    return {"device": {"name": name, "address": addr, "rssi": rnd.randint(-70, -52)},
            "radio": {"config": "auto", "hci": hci, "reason": "auto: demo radio"}, "adapters": ADAPTERS,
            "connected_at": START, "recording": session and session.dir, "files": session and dict(session.paths), "rates": {}, "versions": {"python": sys.version.split()[0]}}


START = time.time()


def now_ms():
    return int(time.time() * 1000)


# ---------------------------------------------------------------------------------- heart
def ecg_shape(phase):
    """One heartbeat (phase 0..1) as a sum of bumps: P, Q, R, S, T. mV."""
    g = lambda c, w, a: a * math.exp(-((phase - c) / w) ** 2)
    return g(0.18, 0.035, 0.12) + g(0.30, 0.010, -0.15) + g(0.32, 0.012, 1.25) + g(0.345, 0.012, -0.3) + g(0.56, 0.06, 0.3)


def stuck_once(logs):
    """BIODASH_DEMO_STUCK=heart-1: this worker (person index 1) gets stuck in its first search, once, like a BlueZ
    scan that never answers, so test_server.py can check that server.py notices and starts it again."""
    if os.environ.get("BIODASH_DEMO_STUCK") != f"{KEY}-{WHO}":
        return
    marker = os.path.join(logs, "demo-was-stuck")
    if os.path.exists(marker):
        return
    write_json(os.path.join(logs, "status.json"), {"state": "scanning", "debug": {}})
    write_json(marker, {"at": time.time()})
    print("Radar active. Sweeping for Polar units... (demo: stuck)", flush=True)
    while True:
        time.sleep(60)


def heart():
    logs = state("polar_dashboard", "logs_advanced")
    stuck_once(logs)
    name, addr = f"Polar H10 DEMO{TAG}", f"DEMO-HEART-{WHO + 1}"
    base = 62 + 14 * WHO                                             # each person's own resting rate
    s = bt_debug.SessionFiles("Polar H10", repair=False)
    s.start(name, addr, {"ecg": ["Timestamp_Epoch_ms", "ECG_mV", "HR_BPM"],
                         "acc": ["Timestamp_Epoch_ms", "X_mg", "Y_mg", "Z_mg"],
                         "rr": ["Timestamp_Epoch_ms", "RR_ms", "HR_BPM"]})
    write_json(os.path.join(logs, "devices.json"), [{"name": name, "address": addr, "rssi": -58}])
    bt_debug.LastDevice(state("polar_dashboard", "last_device.json")).remember(addr, name)
    # the last half hour of beats, so the trends have something to show
    t = now_ms() - 30 * 60_000
    rows = []
    while t < now_ms():
        hr = base + 4 * math.sin(t / 240_000 + WHO) + rnd.gauss(0, 1.2)
        rr = int(60000 / hr)
        t += rr
        rows.append([t, rr, int(hr)])
    s.writerows("rr", rows)
    s.flush()
    next_beat, phase_start, rr, hr = now_ms(), now_ms(), 900, 66
    t_ecg = now_ms()
    while True:
        frame_end = now_ms()
        ecg, acc, beats = [], [], []
        while t_ecg < frame_end:
            if t_ecg >= next_beat:
                hr = base + 4 * math.sin(t_ecg / 240_000 + WHO) + rnd.gauss(0, 1.2)
                rr = int(60000 / hr)
                phase_start, next_beat = next_beat, next_beat + rr
                beats.append([next_beat, rr, int(hr)])
            ph = (t_ecg - phase_start) / rr
            ecg.append([t_ecg, round(ecg_shape(ph) + rnd.gauss(0, 0.015) + 0.05 * math.sin(t_ecg / 1800), 3), int(hr)])
            t_ecg += 8                                          # ~130 Hz
        for k in range(0, len(ecg), 5):
            ts = ecg[k][0]
            acc.append([ts, int(30 * math.sin(ts / 900) + rnd.gauss(0, 8)), int(-990 + rnd.gauss(0, 6)),
                        int(80 + 20 * math.sin(ts / 3000))])
        s.writerows("ecg", ecg)
        s.writerows("acc", acc)
        s.writerows("rr", beats)
        s.flush()
        write_json(os.path.join(logs, "status.json"),
                   {"state": "streaming", "address": addr, "attempt": None, "max_attempts": 3,
                    "last_device": {"address": addr, "name": name, "auto": True},
                    "debug": debug(name, addr, "hci1", s)})
        time.sleep(0.5)


# ---------------------------------------------------------------------------------- oxygen
def oxygen():
    folder = state("viatom_dashboard")
    name, addr = f"O2Ring {TAG}", f"DEMO-O2-{WHO + 1}"
    s = bt_debug.SessionFiles("Viatom O2", repair=False)
    s.start(name, addr, {"vitals": ["Timestamp_Epoch_ms", "SpO2_pct", "HR_BPM", "Battery_pct"]})
    write_json(os.path.join(folder, "devices.json"), [{"name": name, "address": addr, "rssi": -61}])
    bt_debug.LastDevice(os.path.join(folder, "last_device.json")).remember(addr, name)
    t = now_ms() - 30 * 60_000
    rows = []
    while t < now_ms():
        rows.append([t, int(96.5 + 1.5 * math.sin(t / 400_000)), int(62 + 14 * WHO + 3 * math.sin(t / 250_000 + WHO)), 88])
        t += 5000
    s.writerows("vitals", rows)
    s.flush()
    last_row, base, pulse = 0, 97.0 - WHO, 62.0 + 14 * WHO
    while True:
        base = min(99, max(94, base + rnd.choice([0, 0, 0, 0, 0.5, -0.5])))
        pulse = min(67 + 14 * WHO, max(57 + 14 * WHO, pulse + rnd.gauss(0, 0.6)))
        dip = 3 if ((time.time() + 200 * WHO) % 600) < 30 else 0
        spo2 = int(round(base - dip))
        hr = round(pulse)
        write_json(os.path.join(folder, "data.json"), {"spo2": spo2, "hr": hr, "battery": 87, "status": "Connected"})
        d = debug(name, addr, "hci0", s)
        d["state"] = "Connected"
        write_json(os.path.join(folder, "debug.json"), d)
        if time.time() - last_row >= 5:
            last_row = time.time()
            s.writerows("vitals", [[now_ms(), spo2, hr, 87]])
            s.flush()
        time.sleep(2)


# ---------------------------------------------------------------------------------- air
AIR_COLS = ["co2@scd30", "temp@scd30", "rh@scd30", "pm1@sen44", "pm25@sen44", "pm4@sen44", "pm10@sen44",
            "voc@sen44", "rh@sen44", "temp@sen44", "hcho@sfa3x", "rh@sfa3x", "temp@sfa3x"]


def air_row(t):
    m = t / 60000
    co2 = 640 + 380 * (0.5 + 0.5 * math.sin(m / 23)) + rnd.gauss(0, 3)
    pm = max(0.3, 4 + 3 * math.sin(m / 11) + rnd.gauss(0, 0.15))
    temp = 22.4 + 0.8 * math.sin(m / 40)
    rh = 41 + 4 * math.sin(m / 33)
    v = [co2, temp + 1.2, rh - 3, pm * 0.7, pm, pm * 1.1, pm * 1.2, 100 + 40 * math.sin(m / 17) + rnd.gauss(0, 1),
         rh, temp, 12 + 3 * math.sin(m / 29), rh + 1, temp + 0.3]
    return [t] + [round(x, 1) for x in v]


def air():
    logs = state("atmos_dashboard", "logs_atmos")
    name, addr = "Atmos-Sphere-S4", "DEMO-AIR"
    s = bt_debug.SessionFiles("Atmos", repair=False)
    s.start(name, addr, {"readings": ["ts"] + AIR_COLS})
    write_json(os.path.join(logs, "devices.json"), [{"name": name, "address": addr, "rssi": -66}])
    t = now_ms() - 60 * 60_000
    rows = []
    while t < now_ms():
        rows.append(air_row(t))
        t += 2000
    s.writerows("readings", rows)
    s.flush()
    while True:
        s.writerows("readings", [air_row(now_ms())])
        s.flush()
        d = debug(name, addr, "hci0", s)
        d["parser"] = {"profile": "auto", "effective_profile": "s4"}
        write_json(os.path.join(logs, "status.json"), {"state": "streaming", "address": addr, "attempt": None,
                                                       "max_attempts": 3, "last_device": {"address": addr, "name": name,
                                                                                          "auto": True}, "debug": d})
        time.sleep(1)


# ---------------------------------------------------------------------------------- water
def water():
    logs = state("hydro_dashboard", "logs_hydro")
    name, addr = "h2o00009999", "DEMO-BOTTLE"
    s = bt_debug.SessionFiles("HidrateSpark", repair=False)
    s.start(name, addr, {
        "sips": ["Timestamp_Epoch_ms", "Volume_ml", "Pct", "Total_Reported", "Source", "Layout", "Raw",
                 "Weight_before", "Weight_after", "Cal_min", "Cal_max", "Volume_from"],
        "status": ["Timestamp_Epoch_ms", "Fill_ml", "Fill_pct", "Weight_raw", "Weight_stable_raw", "Battery_pct", "Cap"],
        "events": ["Timestamp_Epoch_ms", "Event", "Detail"]})
    write_json(os.path.join(logs, "devices.json"), [{"name": name, "address": addr, "rssi": -63, "bottle": True}])
    cap, fill = 621, 621
    # the last week of sips (and today's so far), at waking hours
    day0 = time.mktime(time.localtime()[:3] + (0, 0, 0, 0, 0, -1))
    rows = []
    for back in range(6, -1, -1):
        t = day0 - back * 86400 + 8 * 3600 + rnd.randint(0, 2400)
        end = min(day0 - back * 86400 + 22 * 3600, time.time() - 60)
        while t < end:
            ml = rnd.choice([40, 55, 70, 90, 120, 150])
            rows.append([int(t * 1000), ml, "", "", "replayed", "demo", "", "", "", "", "", "demo"])
            t += rnd.randint(900, 3600)
    s.writerows("sips", rows)
    s.flush()
    next_sip, live = time.time() + 20, None
    while True:
        if time.time() >= next_sip:
            ml = rnd.choice([40, 60, 85, 110])
            fill = fill - ml if fill - ml > 60 else cap
            s.writerows("sips", [[now_ms(), ml, "", "", "live", "demo", "", "", "", "", "", "demo"]])
            s.flush()
            live_sip = {"ts": time.time(), "ml": ml}
            next_sip = time.time() + rnd.randint(40, 90)
        else:
            live_sip = live["last_sip"] if live else None
        live = {"fill_ml": fill, "fill_pct": round(100 * fill / cap), "battery": 76, "cap_open": False,
                "last_sip": live_sip, "fill_from": "live weight", "capacity_bottle": cap, "can_glow": True,
                "can_upload_glow": True, "synced_at": START, "reminders_set": 8, "firmware": "112.3", "family": "telink"}
        s.writerows("status", [[now_ms(), fill, live["fill_pct"], "", "", 76, "closed"]])
        write_json(os.path.join(logs, "status.json"), {"state": "streaming", "address": addr, "attempt": None,
                                                       "max_attempts": 4, "live": live,
                                                       "last_device": {"address": addr, "name": name, "auto": True},
                                                       "debug": debug(name, addr, "hci0", s)})
        time.sleep(2)


# ---------------------------------------------------------------------------------- brain
BRAIN_EEG = ["TP9", "AF7", "AF8", "TP10"]
BRAIN_OPTICS = ["LO_NIR", "RO_NIR", "LO_IR", "RO_IR", "LI_NIR", "RI_NIR", "LI_IR", "RI_IR"]


def brain():
    """A Muse S Athena with eyes mostly closed: alpha comes and goes; a pulse on the forehead; small head movements."""
    logs = state("muse_dashboard", "logs_muse")
    name, addr = "MuseS-DEMO", "DEMO-MUSE"
    s = bt_debug.SessionFiles("Muse", repair=False)
    bands = ["delta", "theta", "alpha", "beta", "gamma"]
    s.start(name, addr, {
        "eeg": ["Timestamp_Epoch_ms"] + [f"{c}_uV" for c in BRAIN_EEG],
        "optics": ["Timestamp_Epoch_ms", *BRAIN_OPTICS],
        "imu": ["Timestamp_Epoch_ms", "ACC_X_g", "ACC_Y_g", "ACC_Z_g", "GYRO_X_dps", "GYRO_Y_dps", "GYRO_Z_dps"],
        "bands": ["Timestamp_Epoch_ms"] + [f"{c}_{b}" for c in BRAIN_EEG for b in bands] + [f"{c}_contact" for c in BRAIN_EEG],
        "heart": ["Timestamp_Epoch_ms", "IBI_ms", "HR_BPM"],
        "fnirs": ["Timestamp_Epoch_ms", "L_HbO_uM", "L_HbR_uM", "R_HbO_uM", "R_HbR_uM"],
        "battery": ["Timestamp_Epoch_ms", "Battery_pct"]})
    write_json(os.path.join(logs, "devices.json"), [{"name": name, "address": addr, "rssi": -57}])
    two_pi = 2 * math.pi
    t_eeg = t_opt = t_imu = now_ms()
    next_beat, last_second, battery_at = now_ms() + 900, 0, 0
    while True:
        end = now_ms()
        eeg, optics, imu, beats = [], [], [], []
        while t_eeg < end:                                          # 256 Hz
            sec = t_eeg / 1000
            alpha = 14 * (0.55 + 0.45 * math.sin(sec / 7))          # waxes and wanes
            eeg.append([int(t_eeg)] + [round(alpha * (0.6 if c in (1, 2) else 1.0) * math.sin(two_pi * 10 * sec + c)
                                            + 6 * math.sin(two_pi * 2.2 * sec + 2 * c) + 3 * math.sin(two_pi * 21 * sec + c)
                                            + rnd.gauss(0, 4), 2) for c in range(4)])
            t_eeg += 1000 / 256
        while t_opt < end:                                          # 64 Hz
            sec = t_opt / 1000
            pulse = math.sin(two_pi * 1.1 * sec) + 0.35 * math.sin(two_pi * 2.2 * sec + 1)
            optics.append([int(t_opt)] + [round(52000 + 900 * k - 160 * pulse + rnd.gauss(0, 12), 1) for k in range(8)])
            t_opt += 1000 / 64
        while t_imu < end:                                          # 52 Hz
            sec = t_imu / 1000
            imu.append([int(t_imu), round(0.05 * math.sin(sec / 3), 3), round(0.02 * math.sin(sec / 5), 3),
                        round(0.99 + rnd.gauss(0, 0.002), 3), round(rnd.gauss(0, 0.6), 2), round(rnd.gauss(0, 0.6), 2),
                        round(4 * math.sin(sec / 4), 2)])
            t_imu += 1000 / 52
        while next_beat <= end:
            ibi = int(905 + 40 * math.sin(next_beat / 9000) + rnd.gauss(0, 12))
            beats.append([next_beat, ibi, round(60000 / ibi)])
            next_beat += ibi
        s.writerows("eeg", eeg)
        s.writerows("optics", optics)
        s.writerows("imu", imu)
        s.writerows("heart", beats)
        if end // 1000 != last_second:
            last_second = end // 1000
            sec = end / 1000
            a = 0.55 + 0.45 * math.sin(sec / 7)
            power = [1.25, 0.85, 0.55 + 0.9 * a, 0.45, 0.05]        # log10 uV^2: delta .. gamma
            s.writerows("bands", [[end] + [round(v + rnd.gauss(0, 0.04), 3) for _c in range(4) for v in power] + [1, 1, 1, 2]])
            drift = math.sin(sec / 40)
            s.writerows("fnirs", [[end, round(0.8 * drift + rnd.gauss(0, 0.05), 3), round(-0.3 * drift + rnd.gauss(0, 0.05), 3),
                                   round(0.6 * drift + rnd.gauss(0, 0.05), 3), round(-0.25 * drift + rnd.gauss(0, 0.05), 3)]])
        if time.time() - battery_at > 30:
            battery_at = time.time()
            s.writerows("battery", [[end, 83]])
        s.flush()
        d = debug(name, addr, "hci0", s)
        d.update(preset="p1034", layout={"eeg": 4, "optics": 8}, info={"fw": "demo"},
                 signals={"ppg": {"channel": "LI_NIR", "quality": 0.8, "hr": 66},
                          "fnirs": {"optodes": {"L": "inner", "R": "inner"}, "baseline_s": 10, "of_s": 10}})
        write_json(os.path.join(logs, "status.json"), {"state": "streaming", "address": addr, "attempt": None, "max_attempts": 3,
                                                       "last_device": {"address": addr, "name": name, "auto": True},
                                                       "preset": "p1034", "debug": d})
        time.sleep(0.25)


# ---------------------------------------------------------------------------------- forecasts
def forecast(url):
    """Stands in for forecast_service.py: once a second its heartbeat, and for each overlay that is switched on
    a forecast in the same shape (band, median, one sample on the 8 ms grid), so the overlays can be tried
    without PyTorch. The "model" just repeats the demo heartbeat's shape."""
    import urllib.request

    def post(path, obj):
        req = urllib.request.Request(url + path, data=json.dumps(obj).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3) as r:
            return json.loads(r.read() or b"null")

    dt, n, count = 8, 125, 0
    RR_SCORE = {"n": 20, "model_ms": 14.0, "hold_ms": 16.5}       # as forecast_service.py's Overlay.rr_score()
    while True:
        try:
            on = post("/api/forecast/heartbeat", {
                "state": "running", "detail": "demo", "device": "Polar H10 DEMO0001",
                "overlays": {k: {"on": True, "model": f"demo {k}", "loaded": True, "error": None, "compute_ms": 40.0,
                                 "forecasts": count, **({"beats": 30, "rr_score": RR_SCORE} if k == "rr" else {})}
                             for k in ("ecg", "acc", "rr")}})
            t0 = now_ms() + dt
            count += 1
            common = {"t0": t0, "dt_ms": dt, "samples": 8, "context_rows": 900, "compute_ms": 40.0, "made_at": now_ms()}
            rr = 900
            wave = [round(ecg_shape(((t0 + k * dt) % rr) / rr), 3) for k in range(n)]
            if on.get("ecg"):
                post("/api/forecast/live", {**common, "kind": "ecg", "model": "demo ecg", "lanes": ["ecg_uv"], "units": ["mV"],
                                            "last": [wave[0]], "median": [wave], "rep": [wave],
                                            "lo": [[round(v - 0.08, 3) for v in wave]], "hi": [[round(v + 0.08, 3) for v in wave]]})
            if on.get("acc"):
                lanes = [[int(30 * math.sin((t0 + k * dt) / 900)) for k in range(n)], [-990] * n,
                         [int(80 + 20 * math.sin((t0 + k * dt) / 3000)) for k in range(n)]]
                post("/api/forecast/live", {**common, "kind": "acc", "model": "demo ecg_acc",
                                            "lanes": ["ecg_uv", "acc_x_mg", "acc_y_mg", "acc_z_mg"], "units": ["mV", "mg", "mg", "mg"],
                                            "last": [wave[0]] + [l[0] for l in lanes], "median": [wave] + lanes, "rep": [wave] + lanes,
                                            "lo": [wave] + [[v - 12 for v in l] for l in lanes],
                                            "hi": [wave] + [[v + 12 for v in l] for l in lanes]})
            if on.get("rr"):
                post("/api/forecast/live", {**common, "kind": "rr", "model": "demo ecg_rr", "lanes": ["ecg_uv", "rr_ms"],
                                            "units": ["mV", "ms"], "last": [wave[0], rr], "median": [wave, [rr] * n], "rep": [wave, [rr] * n],
                                            "lo": [wave, [rr - 30] * n], "hi": [wave, [rr + 30] * n],
                                            "beats": [{"t": now_ms() + 600, "rr": rr + 8, "lo": rr - 25, "hi": rr + 40, "of": 8}],
                                            "rr_score": RR_SCORE})
        except OSError:
            pass                                                    # the server is starting or stopping
        time.sleep(0.5)


if __name__ == "__main__":
    print(f"demo {KEY}: made-up readings into {bt_debug.data_root()}", flush=True)
    try:
        if KEY == "forecast":
            forecast(sys.argv[2].rstrip("/"))
        {"heart": heart, "oxygen": oxygen, "brain": brain, "air": air, "water": water}[KEY]()
    except KeyboardInterrupt:
        pass
