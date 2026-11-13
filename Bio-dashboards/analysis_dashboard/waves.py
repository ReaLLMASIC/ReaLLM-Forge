"""The Polar H10's ECG and accelerometer on one even time grid (default: every 8 ms, 125 Hz).

Used to build training data for ReaLLM-Forge (tools/export_waves.py) and, live, by the forecast
service, so a model sees exactly the same kind of rows in both places.

The grid method (Settings(grid_ms=8), the default; "bio-dash waves v2"):
  * one row every grid_ms, at whole multiples of grid_ms in epoch time (so offline and live land on
    exactly the same instants)
  * every channel is interpolated in a straight line at each grid instant: the ECG from the two ECG
    samples either side of it, each accelerometer axis from the two accelerometer samples either side
  * a piece of recording ends at an ECG step of more than 50 ms (or a step backwards), and at a grid
    instant whose two neighbouring accelerometer samples are more than 50 ms apart (or that falls
    outside the accelerometer recording)
  * offline, pieces shorter than 30 s are left out; live, a piece is used once it is long enough
  * whole numbers: the ECG in microvolts clipped to +-2500, the accelerometer in milli-g clipped to
    +-2000

The ECG-clock method (Settings(grid_ms=None); "bio-dash waves v1") is the one the first ECG models
were trained with (ReaLLM-Forge's data/csv_mc_int/ecg_data_conditioning.py; the output matches that
script row for row): one row per ECG sample as recorded (~130 Hz, the ECG not resampled), the
accelerometer interpolated at each ECG sample's time, the same gap rules and whole numbers.

Recordings come from Polar-dash (<time>_ecg.csv: Timestamp_Epoch_ms,ECG_mV,HR_BPM) or Omni
(Timestamp_Epoch_ms,ECG_mV), each with its <time>_acc.csv (Timestamp_Epoch_ms,X_mg,Y_mg,Z_mg).
Standard library only.
"""
import array
import bisect
import collections
import csv
import io
import math

ECG_COLUMN = "ECG_mV"
ACC_COLUMNS = ("X_mg", "Y_mg", "Z_mg")
TIME_COLUMN = "Timestamp_Epoch_ms"


class Settings:
    """The method's parameters. grid_ms=None is the ECG-clock method of the first ECG models."""

    KEYS = ("with_acc", "max_gap_ms", "acc_max_gap_ms", "min_piece_s", "sample_rate_hz", "grid_ms",
            "ecg_step_uv", "ecg_clip_uv", "acc_step_mg", "acc_clip_mg")

    def __init__(self, with_acc=True, max_gap_ms=50.0, acc_max_gap_ms=50.0, min_piece_s=30.0,
                 sample_rate_hz=130.0, ecg_step_uv=1, ecg_clip_uv=2500, acc_step_mg=1, acc_clip_mg=2000,
                 grid_ms=8):
        if ecg_step_uv < 1 or ecg_clip_uv % ecg_step_uv:
            raise ValueError("the ECG step must be at least 1 uV and divide the ECG clip")
        if acc_step_mg < 1 or acc_clip_mg % acc_step_mg:
            raise ValueError("the accelerometer step must be at least 1 mg and divide its clip")
        if grid_ms is not None and not grid_ms > 0:
            raise ValueError("the grid step must be positive (or None for the ECG-clock method)")
        self.with_acc = bool(with_acc)
        self.max_gap_ms = float(max_gap_ms)
        self.acc_max_gap_ms = float(acc_max_gap_ms)
        self.min_piece_s = float(min_piece_s)
        # whole-ms grids stay ints, so grid instants are exact integers
        self.grid_ms = None if grid_ms is None else (int(grid_ms) if float(grid_ms).is_integer() else float(grid_ms))
        self.sample_rate_hz = 1000.0 / self.grid_ms if self.grid_ms else float(sample_rate_hz)
        self.ecg_step_uv = int(ecg_step_uv)
        self.ecg_clip_uv = int(ecg_clip_uv)
        self.acc_step_mg = int(acc_step_mg)
        self.acc_clip_mg = int(acc_clip_mg)

    @property
    def method(self):
        return "grid" if self.grid_ms else "ecg-clock"

    @property
    def min_piece_rows(self):
        return int(round(self.min_piece_s * self.sample_rate_hz))

    @property
    def ecg_limit(self):
        return self.ecg_clip_uv // self.ecg_step_uv

    @property
    def acc_limit(self):
        return self.acc_clip_mg // self.acc_step_mg

    def as_dict(self):
        return {k: getattr(self, k) for k in self.KEYS}

    @classmethod
    def from_dict(cls, d):
        """Settings saved by as_dict() (e.g. in an export's columns.json); unknown keys are ignored.
        Settings saved before the grid method existed (no grid_ms) are the ECG-clock method."""
        kw = {k: d[k] for k in cls.KEYS if k in d}
        kw.setdefault("grid_ms", None)
        return cls(**kw)

    def __eq__(self, other):
        return isinstance(other, Settings) and self.as_dict() == other.as_dict()

    def __repr__(self):
        return "Settings(" + ", ".join(f"{k}={v!r}" for k, v in self.as_dict().items()) + ")"


def columns(s):
    """[{column, int_min, int_max, vocab_size, unit, scale, signal}] in CSV order (for columns.json)."""
    ecg_name = "ecg_uv" if s.ecg_step_uv == 1 else f"ecg_{s.ecg_step_uv}uv"
    out = [{"column": ecg_name, "signal": "polar.ecg", "unit": "mV", "scale": 1000 // s.ecg_step_uv
            if 1000 % s.ecg_step_uv == 0 else 1000 / s.ecg_step_uv, "step": s.ecg_step_uv,
            "int_min": -s.ecg_limit, "int_max": s.ecg_limit, "vocab_size": 2 * s.ecg_limit + 1,
            "decode": f"mV = value * {s.ecg_step_uv} / 1000"}]
    if s.with_acc:
        sfx = "mg" if s.acc_step_mg == 1 else f"{s.acc_step_mg}mg"
        for axis in "xyz":
            out.append({"column": f"acc_{axis}_{sfx}", "signal": f"polar.acc.{axis}", "unit": "mg",
                        "scale": 1, "step": s.acc_step_mg, "int_min": -s.acc_limit, "int_max": s.acc_limit,
                        "vocab_size": 2 * s.acc_limit + 1, "decode": f"mg = value * {s.acc_step_mg}"})
    return out


# ---------------------------------------------------------------------------------- reading
def _rows(fh):
    """csv rows of a recording (binary or text handle), NUL bytes from a power cut removed."""
    text = fh if isinstance(fh, io.TextIOBase) else \
        io.TextIOWrapper(fh, encoding="utf-8-sig", errors="replace", newline="")
    return csv.reader(line.replace("\0", "") if "\0" in line else line for line in text)


def _open(src):
    return src() if callable(src) else open(src, "rb")


def _header(rd, wanted, what):
    """Column indexes of `wanted` in the header row (None for an empty file)."""
    head = next(rd, None)
    if not head:
        return None
    cols = [h.strip() for h in head]
    try:
        return [cols.index(c) for c in wanted]
    except ValueError:
        raise ValueError(f"not an H10 {what} recording (header {cols})")


def _samples(src, wanted, what, counts):
    """Good rows of one recording as tuples (int ms, float, ...) in file order. A row with a blank,
    unreadable or non-finite cell is skipped and counted in counts["bad"] (blank lines aren't)."""
    with _open(src) as fh:
        rd = _rows(fh)
        idx = _header(rd, wanted, what)
        if idx is None:
            return
        for row in rd:
            try:
                cells = [row[i].strip() for i in idx]
                if not all(cells):
                    raise ValueError
                vals = (int(cells[0]),) + tuple(float(c) for c in cells[1:])
                if not all(math.isfinite(v) for v in vals[1:]):
                    raise ValueError
            except (ValueError, IndexError):
                if counts is not None and any(cell.strip() for cell in row):
                    counts["bad"] = counts.get("bad", 0) + 1
                continue
            yield vals


def iter_ecg(src, counts=None):
    """(t_ms, mV) for every good row of one *_ecg.csv (a path, or a function that opens it)."""
    return _samples(src, (TIME_COLUMN, ECG_COLUMN), "ECG", counts)


def iter_acc(src, counts=None):
    """(t_ms, x, y, z) for every good row of one *_acc.csv, in file order."""
    return _samples(src, (TIME_COLUMN,) + ACC_COLUMNS, "accelerometer", counts)


def read_ecg(src):
    """(timestamps_ms, mV, bad_rows) from one *_ecg.csv, in file order (arrays: 16 bytes a sample)."""
    counts, ts, mv = {"bad": 0}, array.array("q"), array.array("d")
    for t, v in iter_ecg(src, counts):
        ts.append(t)
        mv.append(v)
    return ts, mv, counts["bad"]


def read_acc(src):
    """(timestamps_ms, [x, y, z], bad_rows) from one *_acc.csv, in time order (equal times keep file order)."""
    counts = {"bad": 0}
    ts, cols = array.array("q"), [array.array("d") for _ in range(3)]
    for t, x, y, z in iter_acc(src, counts):
        ts.append(t)
        cols[0].append(x)
        cols[1].append(y)
        cols[2].append(z)
    if any(ts[i] < ts[i - 1] for i in range(1, len(ts))):
        order = sorted(range(len(ts)), key=ts.__getitem__)          # stable
        ts = array.array("q", (ts[i] for i in order))
        cols = [array.array("d", (c[i] for i in order)) for c in cols]
    return ts, cols, counts["bad"]


# ---------------------------------------------------------------------------------- the method
def bracket(t, a_ts, max_gap_ms):
    """(lo, hi): the accelerometer samples time t is interpolated from (lo == hi when one is exactly
    at t), or None when t is outside the recording or they are more than max_gap_ms apart."""
    n = len(a_ts)
    if n < 2:
        return None
    k = bisect.bisect_left(a_ts, t)
    if k < n and a_ts[k] == t:
        lo = hi = k
    elif k == 0 or k == n:
        return None
    else:
        lo, hi = k - 1, k
    if a_ts[hi] - a_ts[lo] > max_gap_ms:
        return None
    return lo, hi


def acc_at(t, a_ts, a_cols, max_gap_ms):
    """(x, y, z) interpolated at time t, or None when t has no accelerometer sample within reach."""
    br = bracket(t, a_ts, max_gap_ms)
    if br is None:
        return None
    lo, hi = br
    t0, t1 = a_ts[lo], a_ts[hi]
    w = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
    return tuple(col[lo] + w * (col[hi] - col[lo]) for col in a_cols)


def split_at_gaps(ts, max_gap_ms):
    """Index ranges [start, end) with no ECG step over max_gap_ms and none backwards."""
    out, start = [], 0
    for i in range(1, len(ts)):
        d = ts[i] - ts[i - 1]
        if d > max_gap_ms or d < 0:
            out.append((start, i))
            start = i
    if len(ts):
        out.append((start, len(ts)))
    return out


def split_where(a, b, ok):
    """Sub-ranges of [a, b) where ok[i] is true."""
    out, start = [], None
    for i in range(a, b):
        if ok[i] and start is None:
            start = i
        elif not ok[i] and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, b))
    return out


def ecg_int(mv, s):
    """mV -> whole steps of s.ecg_step_uv, clipped. (Returns (value, clipped).)"""
    q = round(mv * 1000.0 / s.ecg_step_uv)
    if q > s.ecg_limit or q < -s.ecg_limit:
        return (s.ecg_limit if q > 0 else -s.ecg_limit), True
    return q, False


def acc_int(mg, s):
    """mg -> whole steps of s.acc_step_mg (half rounds up), clipped. (Returns (value, clipped).)"""
    q = math.floor(mg / s.acc_step_mg + 0.5)
    if q > s.acc_limit or q < -s.acc_limit:
        return (s.acc_limit if q > 0 else -s.acc_limit), True
    return q, False


def decode(row, s):
    """An integer row back to real units: [ECG mV] or [ECG mV, x mg, y mg, z mg]."""
    out = [row[0] * s.ecg_step_uv / 1000.0]
    if s.with_acc:
        out += [v * s.acc_step_mg for v in row[1:4]]
    return out


def pieces(ecg_ts, acc=None, s=None, report=None):
    """The kept pieces of one recording, as ECG index ranges [a, b).

    acc: (a_ts, [x, y, z]) from read_acc (None: no accelerometer file), used when s.with_acc.
    report (a dict) gets the counts: rows, gap_pieces, ecg_rows_without_acc, kept, kept_pieces.
    """
    s = s or Settings()
    rep = report if report is not None else {}
    out = split_at_gaps(ecg_ts, s.max_gap_ms)
    rep["gap_pieces"] = len(out)
    if s.with_acc:
        a_ts = acc[0] if acc else ()
        ok = [bracket(t, a_ts, s.acc_max_gap_ms) is not None for t in ecg_ts]
        rep["ecg_rows_without_acc"] = len(ok) - sum(ok)
        out = [p for a, b in out for p in split_where(a, b, ok)]
    kept = [(a, b) for a, b in out if b - a >= s.min_piece_rows]
    rep["rows"] = len(ecg_ts)
    rep["kept"] = sum(b - a for a, b in kept)
    rep["kept_pieces"] = len(kept)
    return kept


def piece_rows(a, b, ecg_ts, ecg_mv, acc=None, s=None, report=None):
    """The integer rows [ecg] or [ecg, x, y, z] of ECG samples [a, b) (a piece from pieces()).
    report["ecg_clipped"] / report["acc_clipped"] count the values clipped."""
    s = s or Settings()
    rep = report if report is not None else {}
    rep.setdefault("ecg_clipped", 0)
    clipped = rep.setdefault("acc_clipped", [0, 0, 0])
    for i in range(a, b):
        q, c = ecg_int(ecg_mv[i], s)
        if c:
            rep["ecg_clipped"] += 1
        if not s.with_acc:
            yield [q]
            continue
        row = [q]
        for axis, v in enumerate(acc_at(ecg_ts[i], acc[0], acc[1], s.acc_max_gap_ms)):
            qa, ca = acc_int(v, s)
            if ca:
                clipped[axis] += 1
            row.append(qa)
        yield row


def recording_rows(ecg_ts, ecg_mv, acc=None, s=None, report=None):
    """The rows of one recording, piece by piece: yields (piece_ts, piece_rows) for every kept piece
    (piece_ts: the ECG sample times, or the grid instants with the grid method).
    (pieces() + piece_rows() do the same for the ECG-clock method without holding a piece in memory.)"""
    s = s or Settings()
    rep = report if report is not None else {}
    if s.grid_ms:
        yield from grid_recording_rows(ecg_ts, ecg_mv, acc, s, rep)
        return
    for a, b in pieces(ecg_ts, acc, s, rep):
        yield ecg_ts[a:b], list(piece_rows(a, b, ecg_ts, ecg_mv, acc, s, rep))


def export_pieces(ecg_ts, ecg_mv, acc=None, s=None, report=None):
    """For the exporter, either method: yields (first_ms, last_ms, n_rows, rows) per kept piece, rows
    an iterable of integer rows (streamed for the ECG-clock method)."""
    s = s or Settings()
    rep = report if report is not None else {}
    if s.grid_ms:
        for times, rows in grid_recording_rows(ecg_ts, ecg_mv, acc, s, rep):
            yield times[0], times[-1], len(rows), rows
        return
    for a, b in pieces(ecg_ts, acc, s, rep):
        yield ecg_ts[a], ecg_ts[b - 1], b - a, piece_rows(a, b, ecg_ts, ecg_mv, acc, s, rep)


# ---------------------------------------------------------------------------------- the grid method
def grid_index_range(t0, t1, g, include_t0=True):
    """(k0, k1): the grid instants k * g within [t0, t1] (or (t0, t1] with include_t0=False)."""
    if isinstance(g, int) and isinstance(t0, int) and isinstance(t1, int):
        k0 = -((-t0) // g) if include_t0 else t0 // g + 1
        return k0, t1 // g
    k0 = math.ceil(t0 / g) if include_t0 else math.floor(t0 / g) + 1
    return k0, math.floor(t1 / g)


def ecg_between(gt, t0, v0, t1, v1):
    """The ECG at grid instant gt from its neighbours (t0, v0) and (t1, v1), t0 < gt <= t1."""
    if gt == t1:
        return v1
    return v0 + (gt - t0) / (t1 - t0) * (v1 - v0)


def grid_recording_rows(ecg_ts, ecg_mv, acc=None, s=None, report=None):
    """The grid method on one recording: yields (grid_instants, rows) for every kept piece.

    Inside each stretch of ECG without a step over max_gap_ms (or backwards), every multiple of
    grid_ms from its first to its last sample gets a row: the ECG interpolated between its two
    neighbouring samples (a sample exactly on the instant is used as is; of several at the same time,
    the first), and with the accelerometer, each axis interpolated the same way. An instant with no
    accelerometer within reach ends the piece. report gets: rows (instants considered), gap_pieces,
    ecg_rows_without_acc (instants without accelerometer), kept, kept_pieces, ecg_clipped, acc_clipped.
    """
    s = s or Settings()
    rep = report if report is not None else {}
    rep.setdefault("ecg_clipped", 0)
    clipped = rep.setdefault("acc_clipped", [0, 0, 0])
    g = s.grid_ms
    a_ts, a_cols = (acc if acc else ((), None)) if s.with_acc else ((), None)
    gaps = split_at_gaps(ecg_ts, s.max_gap_ms)
    rep["gap_pieces"] = len(gaps)
    total = no_acc = kept_rows = kept_pieces = 0

    def finish(times, rows, cl):
        nonlocal kept_rows, kept_pieces
        if len(rows) < s.min_piece_rows:
            return None
        kept_rows += len(rows)
        kept_pieces += 1
        rep["ecg_clipped"] += cl[0]
        for axis in range(3):
            clipped[axis] += cl[1 + axis]
        return times, rows

    for a, b in gaps:
        k0, k1 = grid_index_range(ecg_ts[a], ecg_ts[b - 1], g)
        j = a
        times, rows, cl = [], [], [0, 0, 0, 0]
        for k in range(k0, k1 + 1):
            gt = k * g
            while ecg_ts[j] < gt:              # j: the first sample at or after gt (exists: gt <= last)
                j += 1
            mv = ecg_mv[j] if ecg_ts[j] == gt else ecg_between(gt, ecg_ts[j - 1], ecg_mv[j - 1], ecg_ts[j], ecg_mv[j])
            total += 1
            q, c = ecg_int(mv, s)
            row = [q]
            if s.with_acc:
                av = acc_at(gt, a_ts, a_cols, s.acc_max_gap_ms) if len(a_ts) else None
                if av is None:
                    no_acc += 1
                    done = finish(times, rows, cl)
                    if done:
                        yield done
                    times, rows, cl = [], [], [0, 0, 0, 0]
                    continue
                for axis, v in enumerate(av):
                    qa, ca = acc_int(v, s)
                    cl[1 + axis] += ca
                    row.append(qa)
            cl[0] += c
            times.append(gt)
            rows.append(row)
        done = finish(times, rows, cl)
        if done:
            yield done
    rep["rows"] = total
    if s.with_acc:
        rep["ecg_rows_without_acc"] = no_acc
    rep["kept"] = kept_rows
    rep["kept_pieces"] = kept_pieces


# ---------------------------------------------------------------------------------- live
class LiveAligner:
    """The same method on a live stream: feed ECG and accelerometer samples as they arrive, get rows.

    push_ecg(ts, mv) / push_acc(ts, x, y, z) take one sample each, in the order they arrive (each
    file's order); rows() returns the rows that are now final, as (t_ms, [ecg, x, y, z], new_piece).
    new_piece is True on the first row after a break, so a model can reset its context.

    A row is final once the accelerometer has run settle_ms past the sample just after the row's
    time: its two neighbours can't change any more. The H10's timestamps overlap by a few ms at
    some frame boundaries (a sample can arrive 1-4 ms "early"); settle_ms (10) absorbs that, so the
    rows match offline exactly. A row waits at most wait_ms of ECG time (the newest ECG sample more
    than wait_ms after it); then it has no accelerometer and ends the piece, as offline. While the
    accelerometer flows neither adds noticeable delay; wait_ms only needs to exceed how far the
    accelerometer can lag the ECG (frame length plus delivery). Feeding the accelerometer before the
    ECG of the same moment avoids waits.

    Fed a recording file by file order and then finish(), it gives the rows recording_rows() gives,
    except that live there is no minimum piece length (that filter needs the piece's end). Live can
    differ only around the strap's clock re-syncs, where one stream's timestamps step back ~1 s:
    accelerometer samples arriving more than settle_ms behind its newest one are placed where
    offline sorting puts them, but rows already given out aren't revised (offline mixes the two
    overlapping stretches; live keeps the first); and accelerometer data more than wait_ms late is
    treated as missing. Accelerometer history is kept for keep_ms (4 s), so an ECG step back finds
    the accelerometer offline would use.

    With the grid method, each ECG sample that arrives fixes the ECG value of the grid instants since
    the previous sample (a grid instant needs the ECG sample after it, so rows trail the newest ECG
    sample by up to one ECG step); those instants then wait for the accelerometer as above. Rows are
    (grid_instant_ms, [ecg, x, y, z], new_piece).
    """

    def __init__(self, s=None, wait_ms=2000.0, settle_ms=10.0, keep_ms=4000.0):
        self.s = s or Settings()
        self.wait_ms = float(wait_ms)
        self.settle_ms = float(settle_ms)
        self.keep_ms = float(keep_ms)
        self.a_ts, self.a_cols = [], [[], [], []]   # sorted by time; equal times in arrival order
        self.pending = collections.deque()   # rows to finish: (time, ECG mV, break_before)
        self.out = []
        self.last_ecg_t = None
        self.last_ecg_mv = None
        self.brk_next = False                # grid method: a break waiting for the next grid instant
        self.in_piece = False
        self.clipped = 0
        self.late_acc = 0                    # accelerometer samples that arrived out of time order

    def push_ecg(self, t, mv):
        t, mv = int(t), float(mv)
        if not math.isfinite(mv):
            return                                    # skipped, as the file readers skip such a row
        brk = False
        prev_t, prev_mv = self.last_ecg_t, self.last_ecg_mv
        if prev_t is not None:
            d = t - prev_t
            brk = d > self.s.max_gap_ms or d < 0
        self.last_ecg_t, self.last_ecg_mv = t, mv
        if not self.s.grid_ms:
            self.pending.append((t, mv, brk))
        else:
            # the grid instants this sample settles: from it (a piece's first sample) or after the last
            g = self.s.grid_ms
            new_piece = prev_t is None or brk
            self.brk_next = self.brk_next or brk
            k0, k1 = grid_index_range(t if new_piece else prev_t, t, g, include_t0=new_piece)
            for k in range(k0, k1 + 1):
                gt = k * g
                v = mv if gt == t else ecg_between(gt, prev_t, prev_mv, t, mv)
                self.pending.append((gt, v, self.brk_next))
                self.brk_next = False
        self._flush()

    def push_acc(self, t, x, y, z):
        t, xyz = int(t), (float(x), float(y), float(z))
        if not all(math.isfinite(v) for v in xyz):
            return
        if self.a_ts and t < self.a_ts[-1]:
            k = bisect.bisect_right(self.a_ts, t)     # where a stable sort by time puts it
            self.a_ts.insert(k, t)
            for axis, v in enumerate(xyz):
                self.a_cols[axis].insert(k, v)
            self.late_acc += 1
        else:
            self.a_ts.append(t)
            for axis, v in enumerate(xyz):
                self.a_cols[axis].append(v)
        self._flush()

    def finish(self):
        """No more samples: settle everything still waiting (end of a replay or of a session)."""
        self._flush(final=True)

    def reset(self):
        """Start over (a new session): forget all samples and pending rows."""
        self.__init__(self.s, self.wait_ms, self.settle_ms, self.keep_ms)

    def rows(self):
        out, self.out = self.out, []
        return out

    def _emit(self, t, mv, brk, acc):
        if brk:
            self.in_piece = False
        q, c = ecg_int(mv, self.s)
        self.clipped += c
        if self.s.with_acc and acc is None:
            self.in_piece = False                     # no accelerometer here: the piece ends
            return
        row = [q] + ([acc_int(v, self.s)[0] for v in acc] if self.s.with_acc else [])
        self.out.append((t, row, not self.in_piece))
        self.in_piece = True

    def _settled(self, t):
        """True once the accelerometer is settle_ms past the first sample at or after t."""
        k = bisect.bisect_left(self.a_ts, t)
        return k < len(self.a_ts) and self.a_ts[-1] >= self.a_ts[k] + self.settle_ms

    def _flush(self, final=False):
        s = self.s
        while self.pending:
            t, mv, brk = self.pending[0]
            acc = None
            if s.with_acc:
                if not final and not self._settled(t) and self.last_ecg_t - t <= self.wait_ms:
                    break                             # wait for the accelerometer to pass t
                acc = acc_at(t, self.a_ts, self.a_cols, s.acc_max_gap_ms)
            self._emit(t, mv, brk, acc)
            self.pending.popleft()
        self._trim()

    def _trim(self):
        """Forget accelerometer samples more than keep_ms older than any ECG still to come (and never
        keep more than 2 x keep_ms behind the newest, should the ECG stall)."""
        if len(self.a_ts) < 512:
            return
        ref = self.last_ecg_t if self.last_ecg_t is not None else self.a_ts[-1]
        if self.pending:
            ref = min(ref, self.pending[0][0])
        ref = max(ref, self.a_ts[-1] - self.keep_ms)
        k = bisect.bisect_left(self.a_ts, ref - self.keep_ms)
        if k > 128:
            del self.a_ts[:k]
            for col in self.a_cols:
                del col[:k]
