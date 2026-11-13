# Tests for Portable Polar

Portable Polar (the Polar H10 in the browser) is its own lean repo; its tests live here, next to the Bio-dash
code it was ported from. They find it through `$PORTABLE_POLAR`, or a `portable_polar` folder next to this
clone (or in `~/Forks`, `~`).

```bash
node tools/portable_polar_tests/test_h10.mjs              # parsing vs polar-python, timestamps vs stream_clock.py
node tools/portable_polar_tests/test_forecast_js.mjs      # 8 ms rows vs waves.py, line drawn vs forecast_service.py
python3 tools/portable_polar_tests/test_page.py           # the page with a stand-in strap (Playwright + Chromium)
python3 tools/portable_polar_tests/test_forecast_browser.py   # the model in the browser vs onnxruntime
xvfb-run -a node tools/portable_polar_tests/test_desktop.mjs  # the desktop app (Electron), needs playwright-core
```

The last one needs `portable_polar/models/ecg` (its `tools/export_onnx.py`) and `vendor/ort` (`tools/fetch_ort.sh`).
The fixtures (`*.json`) were made with polar-python 1.1.1, `stream_clock.py`, `waves.py` and `forecast_service.py`.
