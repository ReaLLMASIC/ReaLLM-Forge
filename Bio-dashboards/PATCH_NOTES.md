# The Forecast buttons stay put
- Heart: switching on the beat-to-beat (R-R) forecast no longer makes all three Forecast buttons disappear (the page read the R-R error under the wrong names)
- Heart: forecasts that arrive slower than every 3 s (on a CPU, or with all three on) stay on screen instead of flashing "Starting…"
- A forecast the page can't read can no longer take the buttons away
- The demo's made-up forecasts now use the service's own field names; test_web.py checks the buttons stay
