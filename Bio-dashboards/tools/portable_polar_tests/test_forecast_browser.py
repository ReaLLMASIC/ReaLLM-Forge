#!/usr/bin/env python3
"""The forecast in the browser (forecast.js + onnxruntime-web) against the same graph in Python's onnxruntime:
identical samples for the same random numbers, and the same greedy rollout. Then the page with a stand-in
strap: the Forecast button runs and draws; and with no model next to the page, the runner loads the two files
once and they're kept for the next visit.

Needs: polar_web/models/ecg (export_onnx.py), polar_web/vendor/ort (fetch_ort.sh), onnxruntime and numpy for
Python, Playwright with Chromium. Skips what it can't run. A GPU isn't needed: Chromium's software WebGPU
(SwiftShader) runs the WebGPU path when it's there, slowly; WebAssembly otherwise."""
import json
import re
import math
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from locate import page_dir  # noqa: E402
WEB = page_dir()
MODEL = os.path.join(WEB, "models", "ecg")
ok = 0


def check(name, cond, detail=""):
    global ok
    print(("  ok  " if cond else "  FAIL ") + name + ("" if cond else f"  {detail}"))
    if not cond:
        raise SystemExit(1)
    ok += 1


missing = [w for w, p in (("models/ecg (run tools/export_onnx.py)", os.path.join(MODEL, "model.onnx")),
                          ("vendor/ort (run tools/fetch_ort.sh)", os.path.join(WEB, "vendor", "ort", "ort.webgpu.min.mjs"))) if not os.path.exists(p)]
try:
    import numpy as np
    import onnxruntime as ort
    from playwright.sync_api import sync_playwright
except ImportError as e:
    missing.append(str(e))
if missing:
    print("skipped: " + "; ".join(missing))
    sys.exit(0)

# ---- the reference: the graph in Python, the page's random numbers (Park-Miller from a seed)
meta = json.load(open(os.path.join(MODEL, "model.json")))
sess = ort.InferenceSession(os.path.join(MODEL, "model.onnx"), providers=["CPUExecutionProvider"])
L, H, D, TK, lo = meta["layers"], meta["kv_heads"], meta["head_dim"], meta["top_k"], meta["lanes"][0]["int_min"]
ctx = [int(round(900 * math.exp(-((((i % 100) - 40) / 2.0) ** 2)) + 200 * math.exp(-((((i % 100) - 70) / 7.0) ** 2)) - 50))
       for i in range(132)]
SEED, N, STEPS = 12345, 2, 30


def reference(greedy):
    """The page's loop: the context once (batch 1), then S rows per call, each call fed the last row."""
    x, S = [SEED], meta.get("steps_per_call", 1)

    def noise():
        a = np.zeros(N * S * TK, np.float32)
        if not greedy:
            for i in range(a.size):
                x[0] = (x[0] * 48271) % 2147483647
                a[i] = -math.log(-math.log(max(x[0] / 2147483647, 1e-12)))
        return a.reshape(N, 1, S, TK)
    e = np.zeros((L, 1, H, 0, D), np.float32)
    feeds = {"tokens": np.array([[[c - lo for c in ctx]]], np.int64), "past_k": e, "past_v": e}
    rows = [[] for _ in range(N)]
    while len(rows[0]) < STEPS:
        nx, pk, pv, _ = sess.run(None, {**feeds, "noise": noise()})
        for n in range(N):
            rows[n] += [int(v) + lo for v in nx[n, 0]]
        feeds = {"tokens": nx[:, :, -1:].copy(), "past_k": pk, "past_v": pv}
    return [r[:STEPS] for r in rows]


ref = {"ctx": ctx, "seed": SEED, "sampled": reference(False), "greedy": reference(True)[0]}
tmp = tempfile.mkdtemp()
json.dump(ref, open(os.path.join(tmp, "ref.json"), "w"))
HARNESS = """<!doctype html><meta charset="utf-8"><script type="module">
import { Forecaster } from './forecast.js';
const ref = await (await fetch('__ref.json')).json();
const f = new Forecaster();
try {
  await f.load();
  let x = ref.seed; const rng = () => { x = (x * 48271) % 2147483647; return x / 2147483647; };
  const t0 = performance.now();
  const sampled = await f.forecast(ref.ctx, 30, 2, rng);
  const ms = performance.now() - t0;
  const greedy = await f.forecast(ref.ctx, 30, 1, Math.random, true);
  window.result = { backend: f.backend, sampled, greedy: greedy[0], ms };
} catch (e) { window.result = { error: String(e && e.stack || e) }; }
</script>"""


def serve(root):
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    p = subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"], cwd=root,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1)
    return p, f"http://127.0.0.1:{port}"


# a copy of the page with the harness in it (and, later, one without the model)
site = os.path.join(tmp, "site")
shutil.copytree(WEB, site, ignore=shutil.ignore_patterns("tests", "__pycache__"))
open(os.path.join(site, "__harness.html"), "w").write(HARNESS)
shutil.copy(os.path.join(tmp, "ref.json"), os.path.join(site, "__ref.json"))
bare = os.path.join(tmp, "bare")
shutil.copytree(site, bare, ignore=shutil.ignore_patterns("models"))
exe = "/opt/pw-browsers/chromium" if os.path.exists("/opt/pw-browsers/chromium") else None
GPU_ARGS = ["--enable-unsafe-webgpu", "--use-webgpu-adapter=swiftshader", "--enable-features=Vulkan"]

srv, url = serve(site)
srv2, url2 = serve(bare)
try:
    with sync_playwright() as p:
        for label, args in (("WebAssembly", []), ("WebGPU (software)", GPU_ARGS)):
            b = p.chromium.launch(executable_path=exe, args=args)
            pg = b.new_page()
            pg.goto(url + "/__harness.html" + ("?webgpu" if args else ""))
            pg.wait_for_function("window.result", timeout=900000)
            r = pg.evaluate("window.result")
            b.close()
            if r.get("error"):
                check(f"{label}: runs", False, r["error"])
            if label.startswith("WebGPU") and r["backend"] != "WebGPU":
                print(f"  (no software WebGPU here: {label} ran as {r['backend']})")
            check(f"{label} [{r['backend']}]: 2 samples x 30 rows identical to Python's onnxruntime ({r['ms']:.0f} ms)",
                  r["sampled"] == ref["sampled"], f"{r['sampled'][0][:6]} vs {ref['sampled'][0][:6]}")
            check(f"{label}: greedy rollout identical", r["greedy"] == ref["greedy"])

        b = p.chromium.launch(executable_path=exe)
        c = b.new_context(viewport={"width": 1300, "height": 1100})
        c.add_init_script(path=os.path.join(HERE, "fake_h10.js"))
        pg = c.new_page()
        errors = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto(url + "/")
        time.sleep(1.5)
        check("model next to the page -> 🔮 Forecast", pg.inner_text("#fc-btn") == "🔮 Forecast")
        pg.click("#connect"); time.sleep(3); pg.click("#fc-btn")
        pg.wait_for_function("document.querySelector('#fc-status').textContent.includes(' ms on ')", timeout=120000)
        st = pg.inner_text("#fc-status")
        check(f"forecasting with the strap: '{st[:60]}…'", "samples" in st and pg.is_visible("#ecg-inference"))
        # too slow here (a forecast takes over 0.9 s on this machine's CPU): it forecasts less until it keeps up
        slow = float(re.search(r"(\d+) ms on", st).group(1)) > 900
        if slow:
            try:
                pg.wait_for_function("/0\\.4 s ahead/.test(document.querySelector('#fc-status').textContent)", timeout=60000)
            except Exception:
                pass
        st2 = pg.inner_text("#fc-status")
        check(f"too slow -> scales down to keep up: '{st2[:40]}…'", not slow or "0.4 s ahead" in st2, st2)
        check("no script errors", not errors, errors)

        pg = b.new_context(viewport={"width": 1300, "height": 1100}).new_page()
        pg.goto(url2 + "/")
        time.sleep(1.5)
        check("no model next to the page -> 🔮 Load forecast model…", pg.inner_text("#fc-btn").startswith("🔮 Load forecast model"))
        with pg.expect_file_chooser() as fc:
            pg.click("#fc-btn")
        fc.value.set_files([os.path.join(MODEL, "model.onnx"), os.path.join(MODEL, "model.json")])
        pg.wait_for_function("document.querySelector('#fc-btn').textContent.includes('Forecast on')", timeout=120000)
        check("the picked files load and the forecast turns on", True)
        pg.reload(); time.sleep(1.5)
        check("…and are kept for the next visit", pg.inner_text("#fc-btn") == "🔮 Forecast")
        b.close()
finally:
    srv.terminate(); srv2.terminate()
    shutil.rmtree(tmp, ignore_errors=True)
print(f"{ok} checks passed")
