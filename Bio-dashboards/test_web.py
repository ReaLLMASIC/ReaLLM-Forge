#!/usr/bin/env python3
"""The web app in a real browser, against `server.py --demo` (made-up devices, no Bluetooth): every
page loads without a script error and shows live numbers; a second person can be added from
Settings and then appears on Compare and in the Heart page's person switcher; the forecast
overlays switch on; the layout fits a phone.

    python3 test_web.py          (needs Playwright + Chromium; skips without them)
"""
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("skipped: Playwright isn't installed")
    sys.exit(0)

HERE = os.path.dirname(os.path.abspath(__file__))
ok = 0


def check(name, cond, detail=""):
    global ok
    if not cond:
        raise SystemExit(f"FAIL {name} {detail}")
    ok += 1


def main():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    base = f"http://127.0.0.1:{port}"
    tmp = tempfile.mkdtemp(prefix="biodash-web-")
    env = dict(os.environ, BIODASH_DATA_DIR=os.path.join(tmp, "data"), BIODASH_APP_CONFIG=os.path.join(tmp, "app.json"),
               XDG_CONFIG_HOME=os.path.join(tmp, "config"), BIODASH_DEMO="1", BIODASH_POWER_DRY_RUN="1")
    proc = subprocess.Popen([sys.executable, os.path.join(HERE, "server.py"), "--port", str(port), "--host", "127.0.0.1"],
                            cwd=HERE, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(80):
            try:
                urllib.request.urlopen(base + "/api/app", timeout=2)
                break
            except OSError:
                time.sleep(0.25)
        time.sleep(4)                                            # the made-up devices connect
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 2000})
            errors = []
            page.on("pageerror", lambda e: errors.append(f"{e} {(e.stack or str())[:400]} @ {page.url}"))
            page.on("console", lambda m: m.type == "error" and errors.append(m.text))
            number = "el => /^\\d/.test(el.textContent.trim())"

            def go(hash_, wait_for):
                page.goto(f"{base}/#/{hash_}")
                page.wait_for_selector(wait_for, timeout=15000)

            go("", '[data-card="heart"]')
            page.wait_for_function("document.querySelector('[data-card=\"heart\"] .main').textContent.trim().match(/^\\d/)", timeout=15000)
            cards = page.eval_on_selector_all("[data-card]", "els => els.map(e => e.dataset.card)")
            check("Home: a card per device", cards == ["heart", "oxygen", "brain", "air", "water"], cards)
            check("no Compare with one person", page.query_selector('#nav [data-page="compare"]') is None)

            go("heart", "[data-ecg] canvas")
            page.wait_for_function(f"({number})(document.querySelector('[data-v=\"hr\"]'))", timeout=15000)
            check("Heart: live heart rate", True)
            check("Heart: no person switcher with one person", page.query_selector(".people-seg") is None)
            for kind in ("ecg", "acc", "rr"):
                page.click(f'[data-fc="{kind}"]')
            page.wait_for_function("['ecg','acc'].every(k => document.querySelector(`[data-fc-status=\"${k}\"]`).textContent.includes('ahead'))"
                                   " && document.querySelector('[data-fc-status=\"rr\"]').textContent.includes('next beat')", timeout=15000)
            check("Heart: three forecast overlays report", "ahead" in page.inner_text('[data-fc-status="acc"]')
                  and "next beat" in page.inner_text('[data-fc-status="rr"]'), page.inner_text('[data-fc-status="rr"]'))
            page.wait_for_timeout(4500)                          # two of the page's settings refreshes
            check("Heart: the Forecast buttons stay after switching on", page.eval_on_selector_all(
                "[data-fc]", "bs => bs.every(b => !b.hidden && b.getAttribute('aria-pressed') === 'true')"))
            check("Heart: the R-R forecast's error, from the service's fields", "error 14 ms (hold 17)" in page.inner_text('[data-fc-status="rr"]'),
                  page.inner_text('[data-fc-status="rr"]'))

            go("oxygen", '[data-v="spo2"]')
            page.wait_for_function(f"({number})(document.querySelector('[data-v=\"spo2\"]'))", timeout=15000)
            check("Oxygen: live SpO2", True)

            go("brain", "[data-eeg] canvas")
            page.wait_for_function(f"({number})(document.querySelector('[data-v=\"alpha\"]'))", timeout=20000)
            drawn = page.evaluate("""() => { const c = document.querySelector('[data-eeg] canvas'), d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
                let n = 0; for (let i = 3; i < d.length; i += 4) if (d[i] > 60) n++; return n; }""")
            check("Brain: alpha share, contact and EEG traces", "all four in contact" in page.inner_text('[data-n="contact"]') and drawn > 1500, drawn)

            go("air", ".page-head")
            go("water", ".page-head")
            go("analysis", ".page-head")
            page.wait_for_selector("[data-cards]", timeout=30000)
            check("Air, Water, Analysis load", True)

            # a second person, from Settings
            go("settings", "[data-add-person]")
            check("Settings: every section", page.eval_on_selector_all("section.card h2", "els => els.map(e => e.textContent)")
                  == ["Devices", "People", "Look", "Forecasts", "Bluetooth radios", "Updates", "This computer", "About"])
            page.click("[data-add-person]")
            page.fill("[data-ask]", "Sam")
            page.click(".dialog [data-ok]")
            page.wait_for_selector(".person >> text=Sam", timeout=10000)
            check("Settings: the new person has a strap and a ring", page.query_selector('[data-dev="heart-p2"]') and page.query_selector('[data-dev="oxygen-p2"]'))
            page.wait_for_selector('#nav [data-page="compare"]', timeout=5000)

            go("compare", "table.compare")
            page.wait_for_function("[...document.querySelectorAll('table.compare [data-v$=\".hr\"]')].every(e => /^\\d/.test(e.textContent))", timeout=25000)
            a, b = [int(x) for x in page.eval_on_selector_all('table.compare [data-v$=".hr"]', "els => els.map(e => e.textContent)")]
            check("Compare: two people, two heart rates", page.inner_text("table.compare tbody tr:nth-child(2) th").strip() == "Sam" and b - a > 3, (a, b))

            go("heart/p2", ".people-seg")
            check("Heart: the second person's strap", "DEMO0002" in page.inner_text("[data-devname]")
                  and page.inner_text('.people-seg [aria-current="true"]') == "Sam" and page.query_selector('[data-fc="ecg"]').is_hidden())
            page.click("[data-device-sheet]")
            page.wait_for_selector(".sheet [data-disconnect]", timeout=5000)
            check("Device sheet: whose it is, and Disconnect", "Sam" in page.inner_text(".sheet-head h2"))
            page.keyboard.press("Escape")

            check("no script errors", not errors, errors)

            phone = browser.new_page(viewport={"width": 390, "height": 800})
            for h in ("", "compare", "heart", "brain", "settings"):
                phone.goto(f"{base}/#/{h}")
                phone.wait_for_timeout(1500)
                over = phone.evaluate("document.documentElement.scrollWidth - window.innerWidth")
                check(f"phone: /{h} doesn't scroll sideways", over <= 1, over)
            browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=25)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"{ok} checks passed")


if __name__ == "__main__":
    main()
