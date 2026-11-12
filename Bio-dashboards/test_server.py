#!/usr/bin/env python3
"""End-to-end check of the one-app server, with made-up devices (no Bluetooth needed):
starts `server.py --demo` on a spare port in a scratch folder, then checks the app API, every
device's API, the websocket (status, the once-a-second summary, a device's live stream), a second
and third person (their own workers, files and history; moving a strap between people; Analysis
keeping them apart), the forecast overlays' plumbing, turning a device off and on, the setup call,
and that the recordings are written in the usual format.

    python3 test_server.py          (needs the packages in requirements.txt)
"""
import asyncio
import csv
import glob
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ok = 0


def check(name, cond, detail=""):
    global ok
    if not cond:
        raise SystemExit(f"FAIL {name} {detail}")
    ok += 1


def port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def get(path, body=None, method=None):
    req = urllib.request.Request(BASE + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method=method or ("GET" if body is None else "POST"))
    with urllib.request.urlopen(req, timeout=20) as r:
        raw = r.read()
        return json.loads(raw) if r.headers.get("content-type", "").startswith("application/json") else raw.decode()


def status_of(path, body=None, method=None):
    try:
        get(path, body, method)
        return 200
    except urllib.error.HTTPError as e:
        return e.code


def people_checks():
    """A second person: their own strap and ring, kept apart from the first person's everywhere."""
    info = get("/api/app")
    check("one person to start with", [p["id"] for p in info["people"]] == ["p1"] and "heart-p2" not in info["instances"])
    check("add a person", get("/api/app/people", {"name": "  Sam  "})["id"] == "p2")
    time.sleep(4)
    info = get("/api/app")
    check("their strap and ring exist", info["instances"]["heart-p2"] == {"kind": "heart", "person": "p2", "enabled": True}
          and info["people"][1]["name"] == "Sam" and info["people"][1]["devices"]["oxygen"]["key"] == "oxygen-p2")
    a, b2 = get("/api/d/heart/api/status"), get("/api/d/heart-p2/api/status")
    check("two straps streaming, not the same one", a["state"] == b2["state"] == "streaming" and a["address"] != b2["address"])
    fa, fb = get("/api/d/heart/api/session")["ecg"], get("/api/d/heart-p2/api/session")["ecg"]
    check("each follows its own recording", fa and fb and fa != fb and "DEMO0001" in fa and "DEMO0002" in fb, (fa, fb))
    ha = get("/api/d/heart/api/history?metric=hr&minutes=30")["hr"]
    hb = get("/api/d/heart-p2/api/history?metric=hr&minutes=30")["hr"]
    mean = lambda pts: sum(p["y"] for p in pts) / len(pts)
    check("their histories differ", len(hb) > 100 and mean(hb) - mean(ha) > 4, (mean(ha), mean(hb)))
    check("second ring's history", len(get("/api/d/oxygen-p2/api/history?minutes=30")["spo2"]) > 100)
    check("the app knows whose unit is whose", info["people"][1]["devices"]["heart"]["unit"] in fb)
    today = time.strftime("%Y-%m-%d")
    units = get(f"/api/d/analysis/api/units?start={today}")
    check("Analysis sees both straps", len(units["heart"]) == 2 and len(units["oxygen"]) == 2, units)
    one = get(f"/api/d/analysis/api/series?start={today}&ids=heart.hr&step=1min&units=" + urllib.parse.quote(units["heart"][0]))
    two = get(f"/api/d/analysis/api/series?start={today}&ids=heart.hr&step=1min&units=" + urllib.parse.quote(units["heart"][1]))
    m1 = [v for v in one["series"]["heart.hr"]["mean"] if v is not None]
    m2 = [v for v in two["series"]["heart.hr"]["mean"] if v is not None]
    check("Analysis keeps two people apart", m1 and m2 and abs(sum(m1) / len(m1) - sum(m2) / len(m2)) > 4)
    # a strap belongs to one person: connecting person 2 to person 1's makes person 1 let go
    out = get("/api/app/claim", {"device": "heart-p2", "address": a["address"]})
    check("moving a strap releases its owner", out["released"] == ["heart"], out)
    check("rename", get("/api/app/people/p2", {"name": "Samira"})["people"][1]["name"] == "Samira")
    check("a third person", get("/api/app/people", {"name": ""})["id"] == "p3")
    check("no fourth", status_of("/api/app/people", {"name": "x"}) == 409)
    check("the first person stays", status_of("/api/app/people/p1", method="DELETE") == 409)
    check("turn one person's ring off", get("/api/app/device", {"device": "oxygen-p3", "enabled": False})["ok"]
          and get("/api/app")["instances"]["oxygen-p3"]["enabled"] is False and get("/api/app")["instances"]["oxygen"]["enabled"])
    get("/api/app/people/p3", method="DELETE")
    info = get("/api/app")
    check("remove a person", [p["id"] for p in info["people"]] == ["p1", "p2"] and "heart-p3" not in info["instances"])


def forecast_checks():
    """The overlays' plumbing, with the demo's made-up forecasts standing in for forecast_service.py."""
    check("forecast service seen", get("/api/app")["forecast"] and get("/api/d/heart/api/forecast/settings")["service"]["online"])
    on = get("/api/d/heart/api/forecast/settings", {"ecg": True, "acc": True, "rr": True})["settings"]
    check("three overlays switch on", on["ecg"] and on["acc"] and on["rr"])
    check("the lab page", "/api/d/heart" in get("/lab"))
    check("the lab's API is passed on (no service in the demo: 503)", status_of("/api/d/heart/api/models") == 503)


async def ws_checks():
    import websockets
    async with websockets.connect(BASE.replace("http", "ws") + "/ws") as ws:
        hello = json.loads(await ws.recv())
        check("ws hello", hello["t"] == "hello" and "heart" in hello["status"])
        await ws.send(json.dumps({"sub": ["heart", "air"]}))
        seen, latest, checked, kinds = set(), None, set(), set()
        end = time.time() + 15
        while time.time() < end and not ({"heart", "air"} <= seen and latest and len(kinds) >= 6):
            m = json.loads(await asyncio.wait_for(ws.recv(), 10))
            if m["t"] == "live":
                seen.add(m["d"])
                if m["d"] == "heart":
                    kinds.add(m["m"]["type"])
                if m["d"] == "heart" and "heart" not in checked and m["m"]["type"] in ("ecg", "acc", "ppi"):
                    checked.add("heart")
                    check("heart live rows", len(m["m"]["rows"][0]) >= 3)
            elif m["t"] == "latest":
                latest = m["v"]
        check("live streams for the subscribed devices", {"heart", "air"} <= seen, seen)
        check("latest summary", latest is not None and "water" in latest)
        check("latest: the headband and the second person", latest["brain"].get("alpha") and latest["heart-p2"].get("hr"), latest.get("brain"))
        check("forecasts arrive on the heart stream", {"ecg_forecast", "acc_forecast", "rr_forecast"} <= kinds, kinds)
        await ws.send(json.dumps({"sub": []}))
        await asyncio.sleep(1.0)
        while True:                                          # drain what was already queued
            try:
                await asyncio.wait_for(ws.recv(), 0.3)
            except asyncio.TimeoutError:
                break
        got_live = False
        end = time.time() + 3
        while time.time() < end:
            try:
                m = json.loads(await asyncio.wait_for(ws.recv(), 1.5))
            except asyncio.TimeoutError:
                continue
            got_live |= m["t"] == "live" and m["d"] == "heart"
        check("unsubscribed: no heart stream", not got_live)


def main():
    global BASE
    tmp = tempfile.mkdtemp(prefix="biodash-test-")
    env = dict(os.environ, BIODASH_DATA_DIR=os.path.join(tmp, "data"), BIODASH_APP_CONFIG=os.path.join(tmp, "app.json"),
               XDG_CONFIG_HOME=os.path.join(tmp, "config"), BIODASH_DEMO="1", BIODASH_POWER_DRY_RUN="1")
    p = port()
    BASE = f"http://127.0.0.1:{p}"
    log = open(os.path.join(tmp, "server.log"), "w+")
    proc = subprocess.Popen([sys.executable, os.path.join(HERE, "server.py"), "--port", str(p), "--host", "127.0.0.1"],
                            cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    try:
        for _ in range(60):
            try:
                get("/api/app")
                break
            except Exception:
                time.sleep(0.25)
        info = get("/api/app")
        check("app info", info["demo"] and set(info["devices"]) == {"heart", "oxygen", "brain", "air", "water"})
        check("page", "<title>Bio-dash</title>" in get("/") and "{{V}}" not in get("/"))
        check("static", "class Plot" in get("/static/js/plot.js"))
        time.sleep(4)                                        # the made-up devices connect
        for k in ("heart", "oxygen", "brain", "air", "water"):
            st = get(f"/api/d/{k}/api/status")
            check(f"{k} streaming", st.get("state") == "streaming", st.get("state"))
            check(f"{k} radio api", get(f"/api/d/{k}/api/radio")["radio"] == ("any" if k == "water" else "auto"))   # the bottle isn't pinned
            check(f"{k} last-device api", "device" in get(f"/api/d/{k}/api/last-device")["devices"])
        check("heart history", len(get("/api/d/heart/api/history?metric=hr&minutes=30")["hr"]) > 100)
        check("oxygen history", len(get("/api/d/oxygen/api/history?minutes=30")["spo2"]) > 100)
        check("air history", len(get("/api/d/air/api/history?minutes=30")["rows"]) > 100)
        s = get("/api/d/water/api/summary?days=7")
        check("water summary", len(s["daily"]) == 7 and s["goal_ml"] > 0)
        check("water settings", get("/api/d/water/api/settings", {"goal_ml": 2700})["goal_ml"] == 2700)
        today = time.strftime("%Y-%m-%d")
        check("analysis days", today in get("/api/d/analysis/api/days")["days"])
        check("analysis summary", "heart" in get(f"/api/d/analysis/api/summary?start={today}")["cards"])
        with urllib.request.urlopen(BASE + f"/api/d/analysis/api/export_mcint.zip?start={today}&ids=heart.hr,oxygen.spo2&step=10s",
                                    timeout=60) as r:
            import io as _io, zipfile as _zip
            z = _zip.ZipFile(_io.BytesIO(r.read()))
            csv_name = next(n for n in z.namelist() if n.endswith(".csv"))
            head = z.read(csv_name).decode().splitlines()
        check("ReaLLM-Forge export", head[0] == "minute_of_day,second,hr_bpm,spo2_pct" and
              all(c.lstrip("-").isdigit() for c in head[1].split(",")), head[:2])
        check("power info", get("/api/system/power")["allowed"])
        check("restart info", get("/api/system/restart")["allowed"])
        check("device names", get("/api/app/device-names", {"key": "Polar H10 TEST", "name": "Mine"})["names"].get("Polar H10 TEST") == "Mine"
              and "Polar H10 TEST" not in get("/api/app/device-names", {"key": "Polar H10 TEST", "name": ""})["names"])
        check("radios", "adapters" in get("/api/app/radios"))
        check("brain history", len(get("/api/d/brain/api/history?metric=alpha&minutes=15")["alpha"]) > 2)
        check("brain presets", get("/api/d/brain/api/preset")["preset"] in get("/api/d/brain/api/preset")["presets"])
        people_checks()
        forecast_checks()
        asyncio.run(ws_checks())
        # off and on again
        get("/api/app/device", {"device": "air", "enabled": False})
        time.sleep(1.5)
        check("air off", get("/api/app")["devices"]["air"]["enabled"] is False)
        check("air worker stopped", get("/api/app/log/air")["worker"]["state"] == "off")
        get("/api/app/device", {"device": "air", "enabled": True})
        time.sleep(1.5)
        check("air back on", get("/api/app/log/air")["worker"]["state"] == "running")
        get("/api/app/setup", {"devices": ["heart", "water"], "done": True})
        cfg = json.load(open(os.path.join(tmp, "app.json")))
        check("setup saved", cfg["devices"] == {"heart": True, "oxygen": False, "brain": False, "air": False, "water": True} and cfg["setup_done"]
              and [p["id"] for p in cfg["people"]] == ["p1", "p2"])
        # recordings: same folders and headers as ever
        ecg = glob.glob(os.path.join(tmp, "data", "Polar H10", "*", "*", "*_ecg.csv"))
        check("ecg recorded", ecg)
        with open(ecg[0]) as f:
            check("ecg header", next(csv.reader(f)) == ["Timestamp_Epoch_ms", "ECG_mV", "HR_BPM"])
        vit = glob.glob(os.path.join(tmp, "data", "Viatom O2", "*", "*", "*_vitals.csv"))
        with open(vit[0]) as f:
            check("vitals header", next(csv.reader(f)) == ["Timestamp_Epoch_ms", "SpO2_pct", "HR_BPM", "Battery_pct"])
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=25)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log.seek(0)
        out = log.read()
        if ok < 60:
            print(out[-3000:])
        check("workers stopped with Ctrl+C, not restarted", "starting it again" not in out)
        check("server stopped cleanly", proc.returncode is not None)
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"{ok} checks passed")


if __name__ == "__main__":
    main()
