"""HidrateSpark bottle protocol: identifiers, handshake, frame parsing, and the small
state machines (weight stability, refills, fill level, sip de-duplication).

Written for Bio-dash from publicly documented protocol facts -- see PROTOCOL.md for the
sources and what's verified on which bottle. No third-party code is included.
"""
import time
from collections import deque

# ---------------------------------------------------------------------------
# GATT identifiers
# ---------------------------------------------------------------------------
SERVICE_REF = "45855422-6565-4cd7-a2a9-fe8af41b85e8"
CHAR_SET_POINT = "b44b03f0-b850-4090-86eb-72863fb3618d"     # handshake writes
CHAR_DEBUG = "e3578b0d-caa7-46d6-b7c2-7331c08de044"         # handshake writes + cap-state notify
CHAR_DATA_POINT = "016e11b1-6c8a-4074-9e5a-076053f93784"    # sip records ("legacy" path)
SERVICE_USER = "bf2d1ba0-c473-49f2-9571-0ce69036c642"
CHAR_USER_DATA = "bf2d1ba1-c473-49f2-9571-0ce69036c642"     # sip records ("modern" path, newer firmware)
CHAR_WEIGHT = "1807a063-4e2d-4636-981a-35e93d1c7b94"        # 2-byte weight, ~every 2 s
CHAR_BATTERY = "00002a19-0000-1000-8000-00805f9b34fb"
CHAR_SERIAL = "00002a25-0000-1000-8000-00805f9b34fb"
CHAR_FIRMWARE = "00002a26-0000-1000-8000-00805f9b34fb"

# Bottles advertise as "h2o..." (some firmware: "HidrateSpark ..."); also match by service.
NAME_PREFIXES = ("h2o", "hidrate")
SERVICE_HINTS = (SERVICE_REF, SERVICE_USER)

# 13 writes, 50 ms apart; after the last one the bottle starts sending sip records.
HANDSHAKE = [
    (CHAR_DEBUG, "2100d1"),
    (CHAR_SET_POINT, "92"),
    (CHAR_DEBUG, "2200f7"),
    (CHAR_SET_POINT, "7700000032d70000"),
    (CHAR_SET_POINT, "00341b00e0790000"),
    (CHAR_SET_POINT, "02345200c0a80000"),
    (CHAR_SET_POINT, "03346e0030c00000"),
    (CHAR_SET_POINT, "04348900a0d70000"),
    (CHAR_SET_POINT, "0534a50010ef0000"),
    (CHAR_SET_POINT, "0634c00080060100"),
    (CHAR_SET_POINT, "0734dc00f01d0100"),
    (CHAR_SET_POINT, "0834000000000000"),
    (CHAR_SET_POINT, "0934000000000000"),
]
HANDSHAKE_INTERVAL_S = 0.05
DRAIN = bytes([0x57])               # "send / ack the next buffered sip record"
MAX_IDENTICAL_FRAMES = 5            # stop re-draining a record the bottle never advances past

KNOWN_CHARS = {
    CHAR_SET_POINT: "set point (handshake)", CHAR_DEBUG: "debug / cap state",
    CHAR_DATA_POINT: "sip records (legacy)", CHAR_USER_DATA: "sip records (modern)",
    CHAR_WEIGHT: "weight", CHAR_BATTERY: "battery", CHAR_SERIAL: "serial number",
    "316c4914-1f59-462e-af06-185418674c0c": "bottle settings (capacity mL, LE)",
    "2007a063-4e2d-4636-981a-35e93d1c7b94": "PRO 2 sensor (next to weight)",
    "4f860001-943b-49ef-bed4-2f730304427a": "Apple Find My", "4f860002-943b-49ef-bed4-2f730304427a": "Apple Find My",
    "4f860003-943b-49ef-bed4-2f730304427a": "Apple Find My", "4f860004-943b-49ef-bed4-2f730304427a": "Apple Find My",
    CHAR_FIRMWARE: "firmware revision",
}


class Drainer:
    """Asks the bottle for buffered sips one record at a time. Each 0x57 acknowledges a
    record, so two requests in flight skip one (seen on a PRO 2 as gaps in the pending count).
    `write` is an async callable that sends hp.DRAIN."""

    def __init__(self, write, timeout=2.0):
        self.write, self.timeout = write, timeout
        self.busy_since = None
        self.writes = 0
        self.last_frame_at = 0.0

    def idle(self, now=None):
        now = time.time() if now is None else now
        return self.busy_since is None or now - self.busy_since > self.timeout

    async def request(self, now=None):
        now = time.time() if now is None else now
        if not self.idle(now):
            return False
        self.busy_since = now
        try:
            await self.write()
            self.writes += 1
            return True
        except Exception:
            self.busy_since = None
            return False

    def on_frame(self, now=None):
        self.busy_since = None
        self.last_frame_at = time.time() if now is None else now


def is_bottle(name, service_uuids=()):
    n = (name or "").strip().lower()
    if any(n.startswith(p) for p in NAME_PREFIXES):
        return True
    return any(u.lower() in SERVICE_HINTS for u in (service_uuids or ()))


# ---------------------------------------------------------------------------
# Sip records
# ---------------------------------------------------------------------------
EPOCH_MIN = 1420070400          # 2015-01-01: no bottle sip predates the company
TEN_YEARS_S = 10 * 365 * 24 * 3600


def _u16(b):
    return int.from_bytes(b, "big")


def _u32(b):
    return int.from_bytes(b, "big")


def parse_sip_frame(data, capacity_ml, now=None):
    """Decode one sip-record notification.

    Two layouts are documented for different firmwares; each frame is decoded with both
    and whichever gives plausible values wins (in practice they never both do):

      layout "epoch":   [0] records pending  [2:4] total mL so far  [4:8] Unix time (s)
                        [8:10] sip volume mL
      layout "percent": [0] records pending  [1] sip volume as % of capacity
                        [2:4] total mL so far  [5:9] seconds since the sip

    Returns a dict with "kind":
      "empty"    queue empty (records pending == 0)        -> nothing to do
      "pending"  records pending but frame too short to hold one -> drain again
      "sip"      a sip: ts (epoch s), volume_ml, layout, pct, total_reported, remaining
      "unknown"  records pending but no layout fits       -> drain again, keep the raw frame
    """
    now = time.time() if now is None else now
    data = bytes(data or b"")
    if not data:
        return {"kind": "unknown", "remaining": 0}
    remaining = data[0]
    if remaining == 0:
        return {"kind": "empty", "remaining": 0}
    if len(data) < 9 or not any(data[1:]):
        return {"kind": "pending", "remaining": remaining}     # e.g. 01 00 00 ...: a record is coming

    # PRO 2 (firmware 100.x), all little-endian -- verified on a real bottle:
    #   [0] pending [1] sip, % of capacity [2:4] today's total, % [4:8] seconds since the sip
    #   [8:12] (constant per session) [12:14] weight before [14:16] weight after
    if len(data) >= 16:
        pct = data[1]
        secs = int.from_bytes(data[4:8], "little")
        w_before = int.from_bytes(data[12:14], "little")
        w_after = int.from_bytes(data[14:16], "little")
        if 1 <= pct <= 100 and secs <= TEN_YEARS_S and w_before and w_after and abs(w_before - w_after) < 20000:
            return {"kind": "sip", "remaining": remaining, "layout": "pro2", "ambiguous": False,
                    "ts": float(now - secs), "volume_ml": round(capacity_ml * pct / 100), "pct": pct,
                    "total_reported": int.from_bytes(data[2:4], "little"),
                    "weight_before": w_before, "weight_after": w_after}

    total = _u16(data[2:4])
    candidates = []
    if len(data) >= 10:
        ts = _u32(data[4:8])
        vol = _u16(data[8:10])
        if EPOCH_MIN <= ts <= now + 86400 and 0 < vol <= max(2000, capacity_ml * 2):
            candidates.append({"layout": "epoch", "ts": float(ts), "volume_ml": vol, "pct": None})
    pct = data[1]
    secs_ago = _u32(data[5:9])
    if 1 <= pct <= 100 and 0 <= secs_ago <= TEN_YEARS_S:
        vol = round(capacity_ml * pct / 100)
        if vol > 0:
            candidates.append({"layout": "percent", "ts": float(now - secs_ago), "volume_ml": vol, "pct": pct})

    if not candidates:
        return {"kind": "unknown", "remaining": remaining}
    best = next((c for c in candidates if c["layout"] == "percent"), candidates[0])   # newer firmware first
    return {"kind": "sip", "remaining": remaining, "total_reported": total,
            "ambiguous": len(candidates) > 1, **best}


# ---------------------------------------------------------------------------
# Cap, weight, battery
# ---------------------------------------------------------------------------
def parse_cap(data):
    """True = cap open, False = closed (bit 0 of byte 0), None = empty frame."""
    data = bytes(data or b"")
    return bool(data[0] & 0x01) if data else None


def parse_weight(data):
    """Raw 16-bit big-endian weight value (rises as the bottle fills), or None."""
    data = bytes(data or b"")
    return _u16(data[:2]) if len(data) >= 2 else None


def parse_battery(data):
    data = bytes(data or b"")
    return data[0] if data and data[0] <= 100 else None


class StableWeight:
    """A reading counts as 'upright and settled' after `need` consecutive samples within
    `tolerance` raw units -- readings while the bottle is moved jump around and never
    form a streak."""

    def __init__(self, tolerance=2, need=3):
        self.tolerance, self.need = tolerance, need
        self.last = None
        self.streak = 0
        self.stable = None          # latest settled value
        self.stable_at = None

    def add(self, raw, now=None):
        now = time.time() if now is None else now
        if self.last is not None and abs(raw - self.last) <= self.tolerance:
            self.streak += 1
        else:
            self.streak = 1
        self.last = raw
        if self.streak >= self.need:
            self.stable, self.stable_at = raw, now
            return raw
        return None


class RefillDetector:
    """Cap opened -> remember the settled weight; cap closed -> wait (up to `window_s`)
    for a new settled weight; a rise of >= `min_rise` raw units is a refill."""

    def __init__(self, min_rise=25, window_s=30):
        self.min_rise, self.window_s = min_rise, window_s
        self.pre = None
        self.closed_at = None
        self.cap_open = False

    def on_cap(self, is_open, stable_weight, now=None):
        now = time.time() if now is None else now
        if is_open and not self.cap_open:
            self.pre, self.closed_at = stable_weight, None
        elif not is_open and self.cap_open:
            self.closed_at = now
        self.cap_open = is_open

    def on_stable(self, raw, now=None):
        """Feed settled weights; returns {"pre", "post", "rise"} once when a refill is seen."""
        now = time.time() if now is None else now
        if self.closed_at is None or self.pre is None:
            return None
        if now - self.closed_at > self.window_s:
            self.closed_at = self.pre = None
            return None
        rise = raw - self.pre
        if rise >= self.min_rise:
            event = {"pre": self.pre, "post": raw, "rise": rise}
            self.closed_at = self.pre = None
            return event
        return None


def fill_from_weight(raw, settings):
    """(fill_ml, fill_pct) from a settled weight and the calibration anchors, or (None, None).

    Both anchors (full + empty): linear between them.
    Only "full": capacity minus (full - raw) * scale  (scale ~1 mL per raw unit).
    """
    if raw is None:
        return None, None
    cap = float(settings.get("capacity_ml") or 0) or None
    full, empty = settings.get("weight_full_raw"), settings.get("weight_empty_raw")
    if not cap or full is None:
        return None, None
    if empty is not None and full != empty:
        frac = (raw - empty) / (full - empty)
    else:
        frac = 1.0 - (full - raw) * float(settings.get("weight_scale_ml", 1.0)) / cap
    frac = max(0.0, min(1.0, frac))
    return round(frac * cap), round(frac * 100)


class SipDeduper:
    """Drops a sip already seen: same volume within +/- `window_s` seconds (replays after a
    partial resync re-send a few records)."""

    def __init__(self, window_s=2.0, keep=200):
        self.window_s = window_s
        self.seen = deque(maxlen=keep)

    def seed(self, sips):
        for ts, vol in sips:
            self.seen.append((float(ts), int(vol)))

    def is_new(self, ts, volume_ml):
        for t, v in self.seen:
            if v == volume_ml and abs(t - ts) <= self.window_s:
                return False
        self.seen.append((float(ts), int(volume_ml)))
        return True
