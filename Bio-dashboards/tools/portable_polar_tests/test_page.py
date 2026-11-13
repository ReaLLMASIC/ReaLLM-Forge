#!/usr/bin/env python3
"""The page in Chromium with a stand-in H10 (fake_h10.js in place of navigator.bluetooth): connects, shows HR and
the ECG, records Bio-dash's three CSVs (in the browser's storage: no folder picked) with the right headers, rates
and rising timestamps, disconnects. Also: a browser without Web Bluetooth gets the explanation, and the monitor
is the dashboards' own copy. Needs Playwright with Chromium; skips otherwise."""
import filecmp
import os
import re
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from locate import page_dir  # noqa: E402
WEB = page_dir()
ok = 0


def check(name, cond, detail=""):
    global ok
    print(("  ok  " if cond else "  FAIL ") + name + ("" if cond else f"  {detail}"))
    if not cond:
        raise SystemExit(1)
    ok += 1


DASH = os.path.join(os.path.dirname(os.path.dirname(HERE)), "polar_dashboard", "static", "js", "ecg_monitor.js")
check("ecg_monitor.js is the same as this Bio-dash's", filecmp.cmp(os.path.join(WEB, "ecg_monitor.js"), DASH, shallow=False),
      "copy polar_dashboard/static/js/ecg_monitor.js over portable_polar/ecg_monitor.js (or the other way)")
try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("skipped the browser part: Playwright isn't installed")
    sys.exit(0)

s = socket.socket(); s.bind(("127.0.0.1", 0)); PORT = s.getsockname()[1]; s.close()
srv = subprocess.Popen([sys.executable, "-m", "http.server", str(PORT), "--bind", "127.0.0.1"], cwd=WEB,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(1)
exe = "/opt/pw-browsers/chromium" if os.path.exists("/opt/pw-browsers/chromium") else None
try:
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=exe)
        ctx = b.new_context(viewport={"width": 1300, "height": 1000})
        ctx.add_init_script(path=os.path.join(HERE, "fake_h10.js"))
        pg = ctx.new_page()
        errors = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto(f"http://127.0.0.1:{PORT}/")
        time.sleep(1)
        pg.click("#connect")
        time.sleep(8)
        check("connects and names the strap", "Polar H10 TEST0001" in pg.inner_text("#status"), pg.inner_text("#status"))
        check("heart rate shown", pg.inner_text("#hr-val") == "73" and pg.inner_text("#mon-hr") == "73")
        check("the recording's path is shown", "Polar H10/Polar H10 TEST0001/" in pg.inner_text("#file"))

        # the debug panel (D), measured in the page
        pg.keyboard.press("d")
        time.sleep(1.5)
        dbg = pg.inner_text("#debug-panel") if pg.is_visible("#debug-panel") else ""
        check("D opens the debug panel: link, stream rates, forecast & versions",
              all(t in dbg.upper() for t in ("LINK", "STREAM RATES", "FORECAST & VERSIONS", "BLUETOOTH RADIO")), dbg[:200])
        m = re.search(r"ECG\s+([\d.]+) / 130 Hz", dbg)
        check(f"debug: ECG rate measured ({m.group(1) if m else '?'} / 130 Hz)", m and 115 < float(m.group(1)) < 145, dbg[:400])
        check("debug: streaming, the strap, nothing dropped", "streaming" in dbg and "Polar H10 TEST0001" in dbg)

        # a name for the strap, typed in place, shown next to the status and kept by the browser
        pg.click("#status-nick .nick")
        pg.fill("#status-nick .nick-input", "Mine")
        pg.press("#status-nick .nick-input", "Enter")
        time.sleep(0.3)
        check("nickname: typed in place, shown next to the status", "Mine" in pg.inner_text("#status-nick"), pg.inner_text("#status-nick"))
        check("nickname: kept by the browser", "Mine" in (pg.evaluate("localStorage.getItem('portable-polar-names')") or ""))
        pg.keyboard.press("d")
        time.sleep(0.3)
        check("D closes it", not pg.is_visible("#debug-panel"))
        pg.click("#disconnect")
        time.sleep(1.5)
        check("disconnects", pg.inner_text("#status") == "Not connected" and pg.is_visible("#connect"))
        check("Connect then uses the strap's name", pg.inner_text("#connect") == "Connect Mine", pg.inner_text("#connect"))
        files = pg.evaluate("""async () => { const r = await navigator.storage.getDirectory(), out = {};
            async function walk(d, p) { for await (const [n, h] of d.entries()) {
                if (h.kind === 'directory') await walk(h, p + n + '/'); else out[p + n] = await (await h.getFile()).text(); } }
            await walk(r, ''); return out; }""")
        by = {k.rsplit("_", 1)[-1][:-4]: v for k, v in files.items()}
        check("three files in Bio-dash's layout", sorted(by) == ["acc", "ecg", "rr"] and
              all(k.startswith("Polar H10/Polar H10 TEST0001/") for k in files), list(files))
        heads = {k: v.split("\n", 1)[0] for k, v in by.items()}
        check("Bio-dash's headers", heads == {"ecg": "Timestamp_Epoch_ms,ECG_mV,HR_BPM", "acc": "Timestamp_Epoch_ms,X_mg,Y_mg,Z_mg",
                                              "rr": "Timestamp_Epoch_ms,RR_ms,HR_BPM"}, heads)
        rows = {k: [ln.split(",") for ln in v.strip().split("\n")[1:]] for k, v in by.items()}
        n_ecg, n_acc = len(rows["ecg"]), len(rows["acc"])
        span = lambda k: (int(rows[k][-1][0]) - int(rows[k][0][0])) / 1000
        hz_ecg, hz_acc = (n_ecg - 1) / span("ecg"), (n_acc - 1) / span("acc")
        check(f"ECG at ~130 Hz ({n_ecg} rows, {hz_ecg:.1f} Hz)", n_ecg > 900 and 125 < hz_ecg < 136, n_ecg)
        check(f"accelerometer at ~205 Hz ({n_acc} rows, {hz_acc:.1f} Hz)", n_acc > 1400 and 195 < hz_acc < 215, n_acc)
        t = [int(r[0]) for r in rows["ecg"]]
        check("ECG timestamps rise, ~7.7 ms apart, near the computer's clock",
              all(b2 >= a for a, b2 in zip(t, t[1:])) and abs((t[-1] - t[0]) / (len(t) - 1) - 1000 / 130) < 0.3
              and abs(t[-1] / 1000 - time.time()) < 30)
        mv = [float(r[1]) for r in rows["ecg"]]
        check("ECG in mV (the beats reach ~0.94 mV)", 0.85 < max(mv) < 1.0 and min(mv) > -0.2, (min(mv), max(mv)))
        check("R-R in ms with the heart rate", rows["rr"] and rows["rr"][0][1:] == ["819", "73"], rows["rr"][:1])
        check("no script errors", not errors, errors)

        p2 = b.new_context().new_page()
        p2.goto(f"http://127.0.0.1:{PORT}/")
        time.sleep(1)
        check("no Web Bluetooth -> the explanation, and Connect is off", p2.is_visible("#unsupported") and p2.is_disabled("#connect"))
        b.close()
finally:
    srv.terminate()
print(f"{ok} checks passed")
