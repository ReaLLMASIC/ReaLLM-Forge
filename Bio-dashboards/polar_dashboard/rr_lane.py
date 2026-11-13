"""The R-R lane of the ECG + R-R forecast model, built live from the strap's R-R messages.

The model (forecast/models/ecg_rr) was trained on rows whose rr_ms lane was made by ReaLLM-Forge's
rr_poc/export_ecg_rr.py; this module makes the same lane from the session's <time>_rr.csv as it grows:

  * Beat times: the rows that arrived together (one Timestamp_Epoch_ms) came in one strap message. A message's
    beats continue the chain from the previous beat (each beat = the one before + its R-R) when that fits: the
    last one no later than the message's arrival and not more than 2 s before it. Otherwise (the start, a dropout,
    drift) they're placed back from the arrival time, spaced by their R-R. A beat that doesn't come after the one
    before is dropped.
  * rr_ms at a grid instant: a straight line between the beats either side; at or after the newest beat known so
    far, that beat's R-R held (the "hold" live mode: in training the line already ran on to the next beat, which
    a live forecast can't know yet). Before the first beat there is no value.
  * Whole ms (Python's round), clipped to 0..2500.

Standard library only.
"""
import bisect

RR_MIN, RR_MAX = 0, 2500          # the model's rr_ms range (config.json lanes)
CHAIN_LAG_MS = 2000               # a chain more than this behind a message's arrival is re-anchored
GRID_MS = 8


def _place(prev, arr, rrs):
    """Beat times for one message (rebuild_beats in export_ecg_rr.py): chained from `prev` when that fits, else
    back from the arrival. -> (times, anchored)."""
    if prev is not None:
        t, times = prev, []
        for r in rrs:
            t += r
            times.append(t)
        if not (times[-1] > arr or arr - times[-1] > CHAIN_LAG_MS):
            return times, False
    t, times = arr, [arr]
    for r in reversed(rrs[1:]):
        t -= r
        times.append(t)
    times.reverse()
    return times, True


class LiveBeats:
    """Beats from R-R rows as they arrive: push(arrival_ms, rr_ms) for each row of the _rr.csv, in file order.

    Rows of one message share their arrival time. The newest message may still be growing (a read can end in the
    middle of it), so its beats are placed again each time a row joins it; older messages are final."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.times, self.rrs = [], []             # beats, strictly increasing in time
        self._msg = None                          # the newest message: [arrival, [rr...]]
        self._base = 0                            # beats before the newest message
        self.messages = self.anchors = self.dropped = 0
        self._anchored_last, self._dropped_last = False, 0

    def push(self, arrival_ms, rr_ms):
        rr = int(round(float(rr_ms)))
        if rr <= 0:
            return
        arr = int(arrival_ms)
        if self._msg is not None and self._msg[0] == arr:
            self._msg[1].append(rr)
            del self.times[self._base:]
            del self.rrs[self._base:]
            self.anchors -= self._anchored_last
            self.dropped -= self._dropped_last
        else:
            self._msg = [arr, [rr]]
            self._base = len(self.times)
            self.messages += 1
        prev = self.times[-1] if self.times else None
        times, anchored = _place(prev, arr, self._msg[1])
        self._anchored_last = anchored
        self.anchors += anchored
        self._dropped_last = 0
        for t, r in zip(times, self._msg[1]):
            if self.times and t <= self.times[-1]:
                self._dropped_last += 1
                continue
            self.times.append(t)
            self.rrs.append(r)
        self.dropped += self._dropped_last

    def __len__(self):
        return len(self.times)

    def value_at(self, t):
        """rr_ms at grid instant t (ms), or None before the first beat."""
        if not self.times or t < self.times[0]:
            return None
        j = bisect.bisect_right(self.times, t) - 1
        if j + 1 < len(self.times):
            ta, ra, tb, rb = self.times[j], self.rrs[j], self.times[j + 1], self.rrs[j + 1]
            v = ra + (rb - ra) * (t - ta) / (tb - ta)
        else:
            v = self.rrs[j]
        return min(max(round(v), RR_MIN), RR_MAX)

    def lane(self, times):
        """rr_ms for each grid instant in `times` (None before the first beat)."""
        return [self.value_at(t) for t in times]


def read_beats(rr, start, max_beats, grid_ms=GRID_MS):
    """The next beats in an rr_ms lane: rr = lane values (ms) one row apart, `start` = the row of the last known
    beat (may be negative: before the first value). From a beat at row p the next beat is the first row r with
    (r - p) * grid >= the lane's value at r, and that value is its R-R (rr_infer_demo.py's rule).
    -> [(row, R-R ms)], chained."""
    out, pos, n = [], start, len(rr)
    for _ in range(max_beats):
        nxt = next((r for r in range(max(pos + 1, 0), n) if (r - pos) * grid_ms >= rr[r]), None)
        if nxt is None:
            break
        out.append((nxt, float(rr[nxt])))
        pos = nxt
    return out
