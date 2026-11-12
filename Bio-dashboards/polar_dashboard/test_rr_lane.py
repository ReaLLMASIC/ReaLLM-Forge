#!/usr/bin/env python3
"""The live R-R lane (rr_lane.py) and reading beats out of an R-R forecast (forecast_service.next_beats).
Standard library only:  python3 test_rr_lane.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rr_lane  # noqa: E402

ok = 0


def check(name, cond, detail=""):
    global ok
    print(("  ok  " if cond else "  FAIL ") + name + ("" if cond else f"  {detail}"))
    if not cond:
        raise SystemExit(1)
    ok += 1


T0 = 1_760_000_000_000

# ---- beats: the first message is placed back from its arrival, the next ones chain on
b = rr_lane.LiveBeats()
for arr, rr in [(T0, 800), (T0, 820), (T0 + 1300, 790), (T0 + 2100, 810)]:
    b.push(arr, rr)
check("first message: its last beat at the arrival, the one before one R-R earlier",
      b.times[:2] == [T0 - 820, T0] and b.rrs[:2] == [800, 820], (b.times, b.rrs))
check("next messages chain from the previous beat", b.times[2:] == [T0 + 790, T0 + 1600], b.times)
check("one anchor so far", b.anchors == 1 and b.messages == 3, (b.anchors, b.messages))

# ---- a chain that ran ahead of its message (or fell > 2 s behind) is re-anchored
b.push(T0 + 2000, 900)                     # chained beat would be T0 + 2500, after its arrival
check("re-anchored at the arrival when the chain runs ahead", b.times[-1] == T0 + 2000 and b.anchors == 2, b.times)
b.push(T0 + 9000, 800)                     # a dropout: chain T0 + 2800, 6.2 s behind
check("re-anchored after a dropout", b.times[-1] == T0 + 9000 and b.anchors == 3, b.times)

# ---- a message read in two parts gives the same beats as read at once
whole, split = rr_lane.LiveBeats(), rr_lane.LiveBeats()
rows = [(T0, 700), (T0 + 900, 760), (T0 + 900, 740), (T0 + 1700, 780), (T0 + 1700, 30), (T0 + 1700, 790)]
for r in rows:
    whole.push(*r)
for r in rows[:2]:
    split.push(*r)
part = (list(split.times), list(split.rrs))
for r in rows[2:]:
    split.push(*r)
check("a growing message is placed again", (split.times, split.rrs) == (whole.times, whole.rrs)
      and part != (split.times, split.rrs), (split.times, whole.times))
check("a beat that doesn't come after the one before is dropped",
      whole.dropped == 1 and len(whole) == 5, (whole.dropped, whole.times))

# ---- the lane: a straight line between beats, the newest beat's R-R held after it, nothing before the first
L = rr_lane.LiveBeats()
for arr, rr in [(T0, 800), (T0 + 1000, 1000)]:
    L.push(arr, rr)
check("no value before the first beat", L.value_at(T0 - 8) is None)
check("at a beat: its R-R", L.value_at(T0) == 800)
check("between beats: the straight line", L.value_at(T0 + 500) == 900 and L.value_at(T0 + 250) == 850)
check("after the newest beat: held", L.value_at(T0 + 1000) == 1000 and L.value_at(T0 + 1700) == 1000)
check("whole ms (rounded)", L.value_at(T0 + 2) == 800 and L.value_at(T0 + 6) == 801,
      (L.value_at(T0 + 2), L.value_at(T0 + 6)))
check("lane() is value_at for each instant", L.lane([T0 - 8, T0, T0 + 500]) == [None, 800, 900])

# ---- reading the next beats out of a lane (rr_infer_demo.py's rule)
lane = [760] * 95 + [780] * 100          # one row per 8 ms from the beat's row on
got = rr_lane.read_beats([None] + lane, 0, 3)     # row 0 = the beat
check("the next beat: the first row whose distance reaches the lane's R-R",
      got[:2] == [(95, 760.0), (193, 780.0)], got)
check("read from before the first row (start = -1)", rr_lane.read_beats(lane, -1, 1) == [(94, 760.0)])

# ---- forecast_service.next_beats: the samples' median per beat while half of them have it
import forecast_service as fs  # noqa: E402

dt = 8
times = [T0 + i * dt for i in range(100)]
beat_t = times[80] + 3                                  # the newest beat known: just after row 80's instant
rr_ctx = [800] * 100                                    # held after it
s1 = [800] * 125                                        # beat at 100 rows after row 80: row 180 (20 rows in)
s2 = [760] * 125                                        # 95 rows after: row 175
s3 = [None] * 0 + [2000] * 125                          # no beat in this sample's horizon
out = fs.next_beats(times, rr_ctx, [s1, s2, s3], beat_t, dt, dt)
check("a beat 2 of 3 samples have: kept, median time and R-R, the band",
      len(out) == 1 and out[0]["of"] == 2 and out[0]["lo"] == 760 and out[0]["hi"] == 800
      and out[0]["rr"] == 780.0 and out[0]["t"] == times[80] + 97.5 * dt, out)
check("no beat when fewer than half have it", fs.next_beats(times, rr_ctx, [s3, s3, s1], beat_t, dt, dt) == [])
check("no beats before the context", fs.next_beats(times, rr_ctx, [s1], times[0] - 5, dt, dt) == [])

print(f"{ok} checks passed")
