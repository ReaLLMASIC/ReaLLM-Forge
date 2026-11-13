"""Sensor time -> wall-clock epoch ms for the Polar H10's PMD frames (shared file: same copy in
polar_dashboard and Omni-dashboard).

The H10 stamps every PMD frame (ECG, accelerometer) with the time of its LAST sample, on its own
clock. Its clock usually isn't set to real time, so one offset maps it onto the wall clock.

  anchor = ClockAnchor()                       # one per strap: ECG and accelerometer share it
  ecg_clock = StreamClock(130, anchor)
  acc_clock = StreamClock(200, anchor)
  stamps = acc_clock.stamps(frame.timestamp_ns, len(frame.samples))

* Sample spacing comes from the sensor, at the rate it really runs. The accelerometer set to
  "200 Hz" delivers about 205.6 samples a second; spacing samples 5 ms apart stamped each
  frame's first samples up to ~5 ms early, overlapping the previous frame. The rate is now
  measured from the frame times (whole frames with nothing lost) and the nominal rate is only
  the starting point.
* ECG and accelerometer share one offset, so the same sensor instant gets the same wall-clock
  time in both files. With one offset each, they differed by the gap between their first frames'
  delivery delays, often tens of ms.
* No steps back in time. The offset used to be re-anchored whenever sensor and wall clock were
  more than 1 s apart, a jump of 1-1.3 s, often backwards: the "clock re-syncs" seen in the
  recordings. Now drift is corrected gently, at most 1 ms per frame and only past 250 ms, and the
  stamps never go backwards. Only a gap of more than 5 s (a long stall, the strap's clock reset,
  a reconnect) starts over, which leaves a visible gap or step in the recording.
"""
import time

REANCHOR_MS = 5000.0      # sensor and wall clock further apart than this: start over
SLEW_FROM_MS = 250.0      # drift that starts a gentle correction
SLEW_MS = 1.0             # the most one frame's stamps move while correcting


class ClockAnchor:
    """The sensor-to-wall-clock offset of one strap, shared by its streams."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.offset_ms = None
        self.reanchors = 0        # times it started over (each leaves a gap or step in the files)

    def offset(self, last_ms, now_ms, fresh=False):
        target = now_ms - last_ms
        if self.offset_ms is None or fresh or abs(target - self.offset_ms) > REANCHOR_MS:
            if self.offset_ms is not None:
                self.reanchors += 1
            self.offset_ms = target
        else:
            err = target - self.offset_ms   # > 0: frames arrive later than the clock says
            if abs(err) > SLEW_FROM_MS:
                self.offset_ms += max(-SLEW_MS, min(SLEW_MS, err))
        return self.offset_ms


class StreamClock:
    """Wall-clock stamps for one stream's frames (see the module docstring)."""

    MIN_SAMPLES_FOR_RATE = 30     # measured once a whole frame or two has been seen

    def __init__(self, hz, anchor=None):
        self.nominal_ms = 1000.0 / hz
        self.anchor = anchor or ClockAnchor()
        self.reset()

    def reset(self):
        """A new connection: forget the measured rate and the anchor (shared: reset once is enough)."""
        self.anchor.reset()
        self.period_ms = self.nominal_ms
        self.prev_last_ms = None
        self.prev_stamp = None
        self.span_ms = 0.0        # sensor time covered by whole, back-to-back frames
        self.span_n = 0           # and the samples in them
        self.dropped = 0          # samples missing between frames (from sensor timestamps)
        self.last_frame_len = 0

    @property
    def measured_hz(self):
        return 1000.0 / self.period_ms

    def stamps(self, sensor_ts_ns, n, now_ms=None):
        """Epoch-ms stamps for a frame of n samples whose last sample is at sensor_ts_ns."""
        now_ms = time.time() * 1000.0 if now_ms is None else now_ms
        self.last_frame_len = n
        if not sensor_ts_ns:      # the library gave no timestamp: arrival time, nominal spacing
            out = [int(now_ms - (n - 1 - i) * self.period_ms) for i in range(n)]
            return self._forward(out)
        last_ms = sensor_ts_ns / 1e6
        fresh = False
        if self.prev_last_ms is not None and n:
            d = last_ms - self.prev_last_ms
            if d <= 0 or d > REANCHOR_MS:
                fresh = True      # the strap's clock went back or jumped: start over
            else:
                per = d / n
                if 0.9 * self.nominal_ms <= per <= 1.1 * self.nominal_ms:
                    self.span_ms += d             # back to back, nothing lost: measures the rate
                    self.span_n += n
                    if self.span_n >= self.MIN_SAMPLES_FOR_RATE:
                        self.period_ms = self.span_ms / self.span_n
                gap = d - n * self.period_ms
                if gap > 1.5 * self.period_ms:
                    self.dropped += round(gap / self.period_ms)
        self.prev_last_ms = last_ms
        end_ms = last_ms + self.anchor.offset(last_ms, now_ms, fresh)
        out = [int(end_ms - (n - 1 - i) * self.period_ms) for i in range(n)]
        if fresh:
            self.prev_stamp = out[-1] if out else self.prev_stamp
            return out
        return self._forward(out)

    def _forward(self, out):
        """Never earlier than the previous frame's last stamp (equal is allowed: whole ms)."""
        if out and self.prev_stamp is not None and out[0] < self.prev_stamp:
            shift = self.prev_stamp - out[0]
            out = [t + shift for t in out]
        if out:
            self.prev_stamp = out[-1]
        return out
