#!/usr/bin/env python3
"""Checks for stream_clock.py with a simulated H10: ECG at 129.97 Hz in 73-sample frames, the
accelerometer at 205.6 Hz in 36-sample frames, the strap's clock 300 ppm fast against the wall
clock, delivery delays of 20-200 ms with a 1.5 s stall now and then. Compares with the old clock.
Run: python3 polar_dashboard/test_stream_clock.py"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stream_clock import ClockAnchor, StreamClock  # noqa: E402

ok = 0


def check(name, got, want):
    global ok
    if got != want:
        raise AssertionError(f"{name}: got {got!r}, want {want!r}")
    ok += 1


class OldClock:
    """The clock before: nominal spacing, an offset per stream, re-anchored past 1 s."""

    def __init__(self, hz):
        self.period_ms, self.offset_ms = 1000.0 / hz, None

    def stamps(self, sensor_ts_ns, n, now_ms):
        last_ms = sensor_ts_ns / 1e6
        if self.offset_ms is None or abs(last_ms + self.offset_ms - now_ms) > 1000.0:
            self.offset_ms = now_ms - last_ms
        end_ms = last_ms + self.offset_ms
        return [int(end_ms - (n - 1 - i) * self.period_ms) for i in range(n)]


def session(minutes, seed=1, drift_ppm=300.0, stalls=True):
    """Frames in arrival order: (stream, sensor_ts_ns of the last sample, n, wall ms on arrival,
    true sensor ms of every sample)."""
    rnd = random.Random(seed)
    sensor0 = 812_345_678.0                        # the strap's clock, ms (not real time)
    wall0 = 1_760_000_000_000.0
    frames = []
    for stream, hz, n in (("ecg", 129.97, 73), ("acc", 205.6, 36)):
        per = 1000.0 / hz
        k = 0
        while k * per < minutes * 60_000:
            true = [sensor0 + (k + i) * per for i in range(n)]
            frames.append([stream, int(true[-1] * 1e6), n, true])
            k += n
    frames.sort(key=lambda f: f[1])
    stall_until = 0.0
    last_arrival = {}                                  # Bluetooth keeps each stream in order
    out = []
    for stream, ts_ns, n, true in frames:
        s_ms = ts_ns / 1e6
        wall = wall0 + (s_ms - sensor0) / (1 + drift_ppm * 1e-6)   # the strap's clock runs fast
        delay = rnd.uniform(20, 200)
        if stalls and rnd.random() < 0.002:
            stall_until = wall + 1500                  # the radio holds frames back for 1.5 s
        arrive = max(wall + delay, stall_until + rnd.uniform(0, 30), last_arrival.get(stream, 0.0))
        last_arrival[stream] = arrive
        out.append((stream, ts_ns, n, arrive, true))
    out.sort(key=lambda f: f[3])
    return out


def run(frames, make):
    clocks = make()
    rows = {"ecg": [], "acc": []}
    for stream, ts_ns, n, arrive, true in frames:
        st = clocks[stream].stamps(ts_ns, n, now_ms=arrive)
        rows[stream] += list(zip(st, true))
    return rows, clocks


def new_clocks():
    a = ClockAnchor()
    return {"ecg": StreamClock(130, a), "acc": StreamClock(200, a), "anchor": a}


def old_clocks():
    return {"ecg": OldClock(130), "acc": OldClock(200)}


def steps_back(rows):
    return sum(1 for (a, _), (b, _) in zip(rows, rows[1:]) if b < a)


def big_steps_back(rows):
    return sum(1 for (a, _), (b, _) in zip(rows, rows[1:]) if b < a - 500)


frames = session(60)

old, _ = run(frames, old_clocks)
check("old clock: steps back of ~1 s in an hour (the 'clock re-syncs')", big_steps_back(old["ecg"]) > 0, True)

new, clocks = run(frames, new_clocks)
for s in ("ecg", "acc"):
    check(f"{s}: never backwards", steps_back(new[s]), 0)
check("no re-anchors in an hour of drift and stalls", clocks["anchor"].reanchors, 0)
check("accelerometer rate measured", round(clocks["acc"].measured_hz, 1), 205.6)
check("ECG rate measured", round(clocks["ecg"].measured_hz, 2), 129.97)


def spacing_error(rows):
    """Within each frame-sized run: how far stamps stray from the true sensor spacing (ms, max)."""
    worst = 0.0
    for (a, ta), (b, tb) in zip(rows, rows[1:]):
        worst = max(worst, abs((b - a) - (tb - ta)))
    return worst


def spacing_share(rows, ms):
    errs = [abs((b - a) - (tb - ta)) for (a, ta), (b, tb) in zip(rows, rows[1:])]
    return sum(e <= ms for e in errs) / len(errs)


check("accelerometer spacing: whole-ms rounding only (99.9% within 1 ms)", spacing_share(new["acc"][100:], 1.0) > 0.999, True)
check("accelerometer spacing: never more than 3.5 ms off (rounding + two 1 ms corrections)", spacing_error(new["acc"][100:]) < 3.5, True)
check("old accelerometer spacing was off more", spacing_error(old["acc"]) > spacing_error(new["acc"]), True)


def cross_offset(rows):
    """wall - sensor for each stream, at the start (ms): equal = the streams line up."""
    return {s: rows[s][300][0] - rows[s][300][1] for s in rows}


c = cross_offset(new)
check("ECG and accelerometer line up (same sensor instant -> same ms, +-1)", abs(c["ecg"] - c["acc"]) <= 1, True)

# a stall longer than 5 s, or the strap's clock jumping back: starts over, once
frames2 = session(3, seed=4, stalls=False)
frames2 = [(s, ts, n, arr + (8000 if ts / 1e6 > 812_345_678 + 90_000 else 0), t) for s, ts, n, arr, t in frames2]
_, cl2 = run(frames2, new_clocks)
check("an 8 s stall: one re-anchor", cl2["anchor"].reanchors, 1)

clk = StreamClock(130)
a = clk.stamps(10_000 * 10**6, 73, now_ms=1e12)
b = clk.stamps(9_000 * 10**6, 73, now_ms=1e12 + 600)          # the strap's clock went back 1 s
check("strap clock reset: starts over", clk.anchor.reanchors, 1)
check("no timestamp from the library: arrival time", StreamClock(130).stamps(None, 3, now_ms=5000.0), [4984, 4992, 5000])
print(f"{ok} checks passed")
