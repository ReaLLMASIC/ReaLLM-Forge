"""What the dashboard works out from the Muse's raw signals, once a second (numpy).

  EegBands   per channel: power in delta 1-4, theta 4-8, alpha 8-13, beta 13-30, gamma 30-44 Hz over the
             last 2 s (Hann window), as log10 uV^2 ("Bels", like Muse's absolute band powers) and as a share
             of the total; and contact: good / fair / bad / off from how much the signal swings or sits railed.
  PpgHeart   beats from the forehead optics: the IR / NIR channel with the clearest pulse (autocorrelation
             over 8 s), peaks at least 0.33 s apart (40-180 bpm) -> beat time, interval, heart rate.
  Fnirs      relative oxy / deoxy-haemoglobin change per side (modified Beer-Lambert, 730 + 850 nm), from the
             outer optodes when the preset has them (else the inner ones), against the first 10 s
             ("Zero" sets a new baseline). The path length isn't measured, so it assumes 3 cm x DPF 6:
             the shape and direction of the change are what to read, not the absolute uM.
"""
import numpy as np

EEG_HZ, OPTICS_HZ = 256, 64
BANDS = (("delta", 1.0, 4.0), ("theta", 4.0, 8.0), ("alpha", 8.0, 13.0), ("beta", 13.0, 30.0), ("gamma", 30.0, 44.0))
QUALITY = {1: "good", 2: "fair", 4: "bad", 0: "off"}          # Muse's own scale is 1 / 2 / 4 ("horseshoe")


class EegBands:
    WINDOW = 2 * EEG_HZ

    def __init__(self, channels):
        self.channels = list(channels)
        self.buf = np.zeros((self.WINDOW, len(self.channels)))
        self.n = 0
        w = np.hanning(self.WINDOW)
        self.win, self.scale = w, 2.0 / (EEG_HZ * (w ** 2).sum())
        f = np.fft.rfftfreq(self.WINDOW, 1 / EEG_HZ)
        self.df = f[1]
        self.masks = [(f >= lo) & (f < hi) for _, lo, hi in BANDS]

    def push(self, rows):
        a = np.asarray(rows, dtype=float)[:, :len(self.channels)]
        k = len(a)
        if k >= self.WINDOW:
            self.buf[:] = a[-self.WINDOW:]
        else:
            self.buf = np.roll(self.buf, -k, axis=0)
            self.buf[-k:] = a
        self.n += k

    @staticmethod
    def contact(x):
        """1 good, 2 fair, 4 bad, 0 off: railed (stuck at the ends of the range) or flat = off the skin."""
        railed = np.mean((x < 5) | (x > 1445))
        d = x - np.convolve(x, np.ones(64) / 64, mode="same")        # drop the slow drift (~4 Hz high-pass)
        sd = float(np.std(d[32:-32]))
        if railed > 0.05 or sd < 0.5:
            return 0
        return 1 if sd < 40 else 2 if sd < 90 else 4

    def compute(self):
        """-> {channel: {"abs": {band: log10 uV^2}, "rel": {band: share}, "q": contact}}, or None until 2 s are in."""
        if self.n < self.WINDOW:
            return None
        out = {}
        for c, name in enumerate(self.channels):
            x = self.buf[:, c]
            psd = np.abs(np.fft.rfft((x - x.mean()) * self.win)) ** 2 * self.scale
            p = np.array([psd[m].sum() * self.df for m in self.masks])
            tot = p.sum()
            out[name] = {"abs": {b[0]: round(float(np.log10(max(v, 1e-6))), 3) for b, v in zip(BANDS, p)},
                         "rel": {b[0]: round(float(v / tot), 4) if tot > 0 else 0.0 for b, v in zip(BANDS, p)},
                         "q": self.contact(x)}
        return out


def _smooth(x, n):
    return np.convolve(x, np.ones(n) / n, mode="same")


class PpgHeart:
    WINDOW = 8 * OPTICS_HZ
    MIN_GAP = int(0.33 * OPTICS_HZ)          # 180 bpm
    MAX_LAG = int(1.5 * OPTICS_HZ)           # 40 bpm

    def __init__(self, channels):
        self.channels = list(channels)
        self.cand = [i for i, c in enumerate(self.channels) if c.endswith("_IR") or c.endswith("_NIR")] or list(range(len(self.channels)))
        self.t = np.zeros(self.WINDOW)
        self.buf = np.zeros((self.WINDOW, len(self.channels)))
        self.n = 0
        self.last_beat = 0.0
        self.ibis = []
        self.channel = None
        self.quality = 0.0
        self.hr = None

    def push(self, times, rows):
        a = np.asarray(rows, dtype=float)
        k = min(len(a), self.WINDOW)
        self.buf = np.roll(self.buf, -k, axis=0)
        self.buf[-k:] = a[-k:]
        self.t = np.roll(self.t, -k)
        self.t[-k:] = np.asarray(times[-k:], dtype=float)
        self.n += len(a)

    def _pulse(self, x):
        y = -(x - _smooth(x, OPTICS_HZ))                 # high-pass ~1 Hz; more blood = less light back, so flip
        return _smooth(y, 5)[OPTICS_HZ // 2:-OPTICS_HZ // 2]

    def _periodicity(self, y):
        y = y - y.mean()
        if y.std() == 0:
            return 0.0, None
        ac = np.correlate(y, y, "full")[len(y) - 1:]
        ac = ac / ac[0]
        lo = self.MIN_GAP
        seg = ac[lo:self.MAX_LAG]
        i = int(np.argmax(seg))
        return float(seg[i]), lo + i

    def compute(self):
        """-> new beats [(time ms, interval ms, heart rate bpm)], after choosing the clearest channel."""
        if self.n < self.WINDOW:
            return []
        best = (-1.0, None, None, None)
        for c in self.cand:
            y = self._pulse(self.buf[:, c])
            r, lag = self._periodicity(y)
            if r > best[0]:
                best = (r, c, y, lag)
        r, c, y, lag = best
        self.quality, self.channel = round(max(r, 0.0), 2), self.channels[c] if c is not None else None
        if r < 0.4 or y is None:
            self.hr = None
            return []
        t = self.t[OPTICS_HZ // 2:-OPTICS_HZ // 2]
        thr = 0.3 * np.std(y)
        peaks, last = [], -10 ** 9
        for i in range(1, len(y) - 1):
            if y[i] > thr and y[i] >= y[i - 1] and y[i] > y[i + 1]:
                if i - last >= self.MIN_GAP:
                    peaks.append(i); last = i
                elif y[i] > y[peaks[-1]]:
                    peaks[-1] = i; last = i
        beats = []
        for i in peaks:
            den = y[i - 1] - 2 * y[i] + y[i + 1]          # the peak between samples (parabola through 3 points):
            frac = 0.5 * (y[i - 1] - y[i + 1]) / den if den < 0 else 0.0   # 64 Hz alone is +-8 ms per beat
            ts = float(t[i]) + max(-0.5, min(0.5, frac)) * 1000 / OPTICS_HZ
            if ts <= self.last_beat + 1000 * self.MIN_GAP / OPTICS_HZ or i > len(y) - self.MIN_GAP:
                continue                                # seen already, or too close to the window's end to be sure
            if self.last_beat:
                ibi = ts - self.last_beat
                if 333 <= ibi <= 1500:
                    self.ibis = (self.ibis + [ibi])[-5:]
                    self.hr = int(round(60000 / float(np.median(self.ibis))))
                    beats.append((int(ts), int(round(ibi)), self.hr))
                elif ibi > 1500:
                    self.ibis = []                      # a gap (movement, lost contact): start again
            self.last_beat = ts
        return beats


# Molar extinction coefficients, cm^-1 M^-1 (Prahl's tabulation): rows 730 nm, 850 nm; columns HbO2, HHb
_E = np.array([[390.0, 1102.2], [1058.0, 691.32]])
_E_INV = np.linalg.inv(_E)
PATH_CM, DPF = 3.0, 6.0


class Fnirs:
    BASELINE_S = 10

    def __init__(self, channels):
        self.channels = list(channels)
        ix = {c: i for i, c in enumerate(self.channels)}
        self.sides = {}
        for side in ("L", "R"):
            for pos in ("O", "I"):                       # outer optodes first: the longer path reaches deeper
                nir, ir = f"{side}{pos}_NIR", f"{side}{pos}_IR"
                if nir in ix and ir in ix:
                    self.sides[side] = (ix[nir], ix[ir], pos)
                    break
        self.acc, self.base, self.base_n = [], None, 0

    @property
    def optodes(self):
        return {s: ("outer" if p == "O" else "inner") for s, (_, _, p) in self.sides.items()}

    def rebaseline(self):
        self.base, self.base_n = None, 0

    def push(self, rows):
        self.acc.extend(rows)

    def compute(self):
        """-> {"L": (dHbO uM, dHbR uM), "R": ...} for the last second, or None while the baseline builds."""
        if not self.acc or not self.sides:
            self.acc = []
            return None
        m = np.asarray(self.acc, dtype=float).mean(axis=0)
        self.acc = []
        if self.base is None or self.base_n < self.BASELINE_S:
            self.base = m if self.base is None else (self.base * self.base_n + m) / (self.base_n + 1)
            self.base_n += 1
            return None
        out = {}
        for side, (inir, iir, _) in self.sides.items():
            if min(m[inir], m[iir], self.base[inir], self.base[iir]) <= 0:
                continue
            dod = np.array([np.log10(self.base[inir] / m[inir]), np.log10(self.base[iir] / m[iir])])
            hbo, hbr = _E_INV @ dod / (PATH_CM * DPF) * 1e6
            out[side] = (round(float(hbo), 3), round(float(hbr), 3))
        return out
