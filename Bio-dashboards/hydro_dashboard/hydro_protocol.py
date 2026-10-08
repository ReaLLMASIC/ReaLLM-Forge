"""HidrateSpark bottle protocol: identifiers, handshake, frame parsing, and the small
state machines (weight stability, refills, fill level, sip de-duplication).

Written for Bio-dash from publicly documented protocol facts and from how the official app
talks to the bottle -- see PROTOCOL.md for the sources and what's verified on which bottle.
No third-party code is included.
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
CHAR_BOTTLE_SIZE = "316c4914-1f59-462e-af06-185418674c0c"   # capacity in mL, 16-bit little-endian
CHAR_LED = "a1d9a5bf-f5d8-49f3-a440-e6bf27440cb0"           # write one byte: play a glow now
CHAR_GLOW_COUNT = "3bbd83e1-09bd-4b2d-a4e2-03e37694252b"    # glow upload: number of packets to expect
CHAR_GLOW_DATA = "3bbd83e2-09bd-4b2d-a4e2-03e37694252b"     # glow upload: the pattern, 20 bytes at a time
CHAR_HARDWARE = "00002a27-0000-1000-8000-00805f9b34fb"      # hardware revision ("Sensor-SM-V1" = small puck)

# Bottles advertise as "h2o..." (some firmware: "HidrateSpark ..."); also match by service.
NAME_PREFIXES = ("h2o", "hidrate")
SERVICE_HINTS = (SERVICE_REF, SERVICE_USER)

# The sync sent after connecting: today's total, goal, the bottle's clock and its glow-reminder
# schedule. (Bio-dash V3.0 sent a fixed copy of these 13 writes -- someone's settings, with the
# clock always set to 15:18. build_sync() below sends the real values instead.)
LEGACY_HANDSHAKE = [
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
HANDSHAKE = LEGACY_HANDSHAKE
HANDSHAKE_INTERVAL_S = 0.05
# Writes to the sip-record characteristic:
READY_FAST = bytes([0x57])          # "send the next record" (fast mode; older, non-Telink firmware)
READY = bytes([0x55])               # "send the next record"
ACK = bytes([0x33])                 # "got that record" -- the bottle then drops it from its buffer
DRAIN = READY_FAST                  # kept for older callers
# Writes to the debug characteristic: redo the bottle's own calibration (it stores MIN / MAX itself)
CAL_FULL = bytes([0x41])            # "the bottle is full right now"
CAL_EMPTY = bytes([0x40])           # "the bottle is empty right now" (firmware 50+)
CAL_EMPTY_OLD = bytes([0x66])       # same, firmware below 50
# 0xF0 on the debug characteristic puts the bottle into firmware-update mode. Never sent.
MAX_IDENTICAL_FRAMES = 5            # stop re-draining a record the bottle never advances past

KNOWN_CHARS = {
    CHAR_SET_POINT: "set point (handshake)", CHAR_DEBUG: "debug / cap state",
    CHAR_DATA_POINT: "sip records (legacy)", CHAR_USER_DATA: "sip records (modern)",
    CHAR_WEIGHT: "weight", CHAR_BATTERY: "battery", CHAR_SERIAL: "serial number",
    CHAR_LED: "glow now (LED control)", CHAR_GLOW_COUNT: "glow upload (packet count)",
    CHAR_GLOW_DATA: "glow upload (pattern data)", CHAR_HARDWARE: "hardware revision",
    CHAR_BOTTLE_SIZE: "bottle size (capacity mL, LE)",
    "2007a063-4e2d-4636-981a-35e93d1c7b94": "PRO 2 sensor (next to weight)",
    "4f860001-943b-49ef-bed4-2f730304427a": "Apple Find My", "4f860002-943b-49ef-bed4-2f730304427a": "Apple Find My",
    "4f860003-943b-49ef-bed4-2f730304427a": "Apple Find My", "4f860004-943b-49ef-bed4-2f730304427a": "Apple Find My",
    CHAR_FIRMWARE: "firmware revision",
}


def firmware_major(firmware):
    """Leading number of a firmware string ("100.64.0" -> 100), or None."""
    try:
        return int(str(firmware or "").strip().split(".")[0])
    except ValueError:
        return None


def bottle_family(firmware):
    """Hardware family from the firmware's major version (the ranges the official app uses)."""
    m = firmware_major(firmware)
    if m is None:
        return "unknown"
    if m >= 100:
        return "telink"            # PRO 2 and other current bottles
    if m >= 80:
        return "nordic"
    if m >= 50:
        return "cypress"
    if m >= 30:
        return "v3"
    return "early"


def drain_mode_for(firmware):
    """"ack": acknowledge each record (0x33), then ask for the next (0x55) -- Telink and the
    earliest firmware. "fast": one 0x57 per record, no acknowledgement -- everything between.
    Unknown firmware gets "fast", which is what Bio-dash V3.0 used for every bottle."""
    fam = bottle_family(firmware)
    return "ack" if fam in ("telink", "early") else "fast"


class Drainer:
    """Asks the bottle for buffered sips one record at a time, never with two requests in
    flight (overlapping requests skip records).

    mode "fast": `write(READY_FAST)` per record.
    mode "ack":  after a record arrives, `write(ACK)` then `write(READY)`; the very first
                 request is a bare READY.
    If "ack" requests go unanswered `fallback_after` times in a row while records are known
    to be waiting, the drainer falls back to "fast" for the rest of the connection.
    `write` is an async callable taking the bytes to send to the sip characteristic."""

    def __init__(self, write, mode="fast", timeout=2.0, fallback_after=3):
        self.write, self.mode, self.timeout = write, mode, timeout
        self.fallback_after = fallback_after
        self.busy_since = None
        self.writes = 0
        self.acks = 0
        self.last_frame_at = 0.0
        self.unanswered = 0
        self.fell_back = False

    def idle(self, now=None):
        now = time.time() if now is None else now
        return self.busy_since is None or now - self.busy_since > self.timeout

    async def request(self, now=None, ack=False, pending=True):
        """Ask for the next record. ack=True: first acknowledge the one just received."""
        now = time.time() if now is None else now
        if not self.idle(now):
            return False
        if self.busy_since is not None:                  # the previous request timed out
            self.unanswered += 1
            if self.mode == "ack" and pending and self.unanswered >= self.fallback_after:
                self.mode, self.fell_back = "fast", True
        self.busy_since = now
        try:
            if self.mode == "ack":
                if ack:
                    await self.write(ACK)
                    self.acks += 1
                await self.write(READY)
            else:
                await self.write(READY_FAST)
            self.writes += 1
            return True
        except Exception:
            self.busy_since = None
            return False

    def on_frame(self, now=None):
        self.busy_since = None
        self.unanswered = 0
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
    #   [0] pending [1] sip, whole % of capacity [2:4] today's total, % [4:8] seconds since the sip
    #   [8:10] MIN = calibrated empty weight [10:12] MAX = calibrated full weight
    #   [12:14] weight before [14:16] weight after
    # The volume comes from the weights (as the official app computes it); byte 1 is the same
    # quantity rounded down to a whole percent and is only the fallback.
    if len(data) >= 16:
        pct = data[1]
        secs = int.from_bytes(data[4:8], "little")
        cal_min = int.from_bytes(data[8:10], "little")
        cal_max = int.from_bytes(data[10:12], "little")
        w_before = int.from_bytes(data[12:14], "little")
        w_after = int.from_bytes(data[14:16], "little")
        if secs <= TEN_YEARS_S and w_before and w_after and abs(w_before - w_after) < 20000:
            vol_w = sip_from_weights(w_before, w_after, cal_min, cal_max, capacity_ml)
            vol_pct = round(capacity_ml * pct / 100) if 1 <= pct <= 100 else 0
            # Use the weights when they agree with byte 1 (which truncates, so it may read up to
            # one percent low); otherwise the calibration looks off and byte 1 is the safer value.
            use_w = vol_w is not None and vol_w >= 1 and (not vol_pct or abs(vol_w - vol_pct) <= capacity_ml * 0.03 + 2)
            volume = vol_w if use_w else vol_pct
            if volume >= 1:
                return {"kind": "sip", "remaining": remaining, "layout": "pro2", "ambiguous": False,
                        "ts": float(now - secs), "volume_ml": int(volume), "pct": pct,
                        "volume_source": "weights" if use_w else "percent",
                        "total_reported": int.from_bytes(data[2:4], "little"),
                        "weight_before": w_before, "weight_after": w_after,
                        "cal_min": cal_min if valid_calibration(cal_min, cal_max) else None,
                        "cal_max": cal_max if valid_calibration(cal_min, cal_max) else None}

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
# The bottle's own calibration (MIN / MAX in every sip record)
# ---------------------------------------------------------------------------
MIN_CAL_RANGE = 200             # raw units between empty and full; real bottles are well over 1000


def valid_calibration(cal_min, cal_max):
    return bool(cal_min) and bool(cal_max) and cal_max - cal_min >= MIN_CAL_RANGE


def fill_fraction(raw, cal_min, cal_max):
    """0.0 (empty) .. 1.0 (full) for a raw weight, or None without a usable calibration.
    Linear, which is what the official app uses for every model after the V3."""
    if raw is None or not valid_calibration(cal_min, cal_max):
        return None
    return max(0.0, min(1.0, (raw - cal_min) / (cal_max - cal_min)))


def sip_from_weights(w_before, w_after, cal_min, cal_max, capacity_ml):
    """Sip volume in mL from the weights before / after, or None without a usable calibration."""
    a, b = fill_fraction(w_before, cal_min, cal_max), fill_fraction(w_after, cal_min, cal_max)
    if a is None or b is None:
        return None
    return round((a - b) * capacity_ml)


def fill_from_calibration(raw, cal_min, cal_max, capacity_ml):
    """(fill_ml, fill_pct) from a raw weight and the bottle's calibration, or (None, None)."""
    f = fill_fraction(raw, cal_min, cal_max)
    if f is None or not capacity_ml:
        return None, None
    return round(f * capacity_ml), round(f * 100)


def weight_in_band(raw, cal_min, cal_max, slack=0.25):
    """Is a live weight reading plausibly on the same scale as the calibration? (Guards the
    fill level against a reading that isn't a settled, upright weight.)"""
    if raw is None or not valid_calibration(cal_min, cal_max):
        return False
    pad = (cal_max - cal_min) * slack
    return cal_min - pad <= raw <= cal_max + pad


def parse_bottle_size(data):
    """Capacity in mL (16-bit little-endian) from the bottle-size characteristic, or None."""
    data = bytes(data or b"")
    if len(data) < 2:
        return None
    ml = int.from_bytes(data[:2], "little")
    return ml if 100 <= ml <= 3000 else None


# ---------------------------------------------------------------------------
# Glow, reminders and the sync sent after connecting
# ---------------------------------------------------------------------------
GLOW_TELINK = 71                # LED byte for current bottles: play the glow stored on the bottle
GLOW_OLDER = 48                 # LED byte for older bottles: five strobes, all LEDs
REMINDER_LIGHT = 0x34           # light pattern id carried in each reminder slot
SIP_GLOW_ON, SIP_GLOW_OFF = bytes([0xB1]), bytes([0xB0])     # debug characteristic
GOAL_GLOW_ON, GOAL_GLOW_OFF = bytes([0x93, 61]), bytes([0x92])   # set-point characteristic
DEFAULT_REMINDERS = {"enabled": True, "from": "08:00", "to": "22:00", "every_min": 60,
                     "always": False, "sound": False}


def glow_now_bytes(firmware):
    return bytes([GLOW_TELINK if bottle_family(firmware) == "telink" else GLOW_OLDER])


def reminder_slots(firmware):
    return 48 if bottle_family(firmware) == "telink" else 12


def _hhmm(text, default):
    try:
        h, m = str(text).split(":")
        v = int(h) * 3600 + int(m) * 60
        return v if 0 <= v < 86400 else default
    except (ValueError, AttributeError):
        return default


def reminder_times(rem):
    """Seconds-of-day of each reminder: every `every_min` minutes after "from", up to "to"
    (which may be past midnight)."""
    rem = {**DEFAULT_REMINDERS, **(rem or {})}
    every = int(rem.get("every_min") or 0) * 60
    if not rem.get("enabled") or every < 600:
        return []
    start, end = _hhmm(rem["from"], 8 * 3600), _hhmm(rem["to"], 22 * 3600)
    span = (end - start) % 86400 or 86400
    return [(start + k * every) % 86400 for k in range(1, span // every + 1)]


def build_sync(settings, firmware, total_today_ml, seconds_since_midnight):
    """The writes that bring the bottle in step with the dashboard, as (characteristic, bytes):
    today's total, goal glow, goal, clock, then every reminder slot (unused ones cleared).

    A reminder slot says "at this time of day, glow if less than this much has been drunk".
    The amount is the share of the goal that should be done by then, as % of the bottle's
    capacity; 0xFFFF means "glow regardless". Current (Telink) bottles take two extra bytes:
    glow on/off and sound on/off."""
    cap = max(1, int(settings.get("capacity_ml") or 621))
    goal = int(settings.get("goal_ml") or 2500)
    telink = bottle_family(firmware) == "telink"
    rem = {**DEFAULT_REMINDERS, **(settings.get("reminders") or {})}
    pct = lambda ml: max(0, min(0xFFFE, int(ml * 100 / cap)))
    out = [(CHAR_DEBUG, bytes([0x21]) + pct(total_today_ml).to_bytes(2, "big")),
           (CHAR_SET_POINT, GOAL_GLOW_ON if settings.get("goal_glow", True) else GOAL_GLOW_OFF),
           (CHAR_DEBUG, bytes([0x22]) + pct(goal).to_bytes(2, "big")),
           (CHAR_SET_POINT, bytes([0x77, 0, 0, 0]) + int(seconds_since_midnight % 86400).to_bytes(4, "little"))]
    sip_glow = settings.get("sip_glow")              # None = leave the bottle as it is
    if sip_glow is not None:
        out.append((CHAR_DEBUG, SIP_GLOW_ON if sip_glow else SIP_GLOW_OFF))
    slots = reminder_slots(firmware)
    times = [t for t in reminder_times(rem) if t != 0][:slots]     # time 0 means "slot unused"
    for i in range(slots):
        if i < len(times):
            amount = 0xFFFF if rem.get("always") else pct(goal * (i + 1) / (len(times) + 1))
            body = bytes([i, REMINDER_LIGHT]) + amount.to_bytes(2, "little") + times[i].to_bytes(4, "little")
            if telink:
                body += bytes([1, 1 if rem.get("sound") else 0])
        else:
            body = bytes([i]) + bytes(9 if telink else 7)
        out.append((CHAR_SET_POINT, body))
    return out


# ---------------------------------------------------------------------------
# Custom glow patterns (firmware 100+): built here, uploaded to the bottle, played by "glow now"
# and by reminders
# ---------------------------------------------------------------------------
# A pattern is a list of frame groups (animations) and a sequence saying which group to play,
# how often and with what pause. A frame sets every LED of the ring to a colour, stays for
# `shown` ticks and fades to the next frame over `transition` ticks. On the wire it is a
# protobuf message:
#   Pattern    { 1: numSequences  2: Sequence[]  3: numFrameGroups  4: FrameGroup[] }
#   Sequence   { 1: frameGroupId  2: scale  3: repeats  4: delay }
#   FrameGroup { 1: id  2: numFrames  3: Frame[] }
#   Frame      { 2: ticksShown  3: transitionTicks  4: LEDColor[] }
#   LEDColor   { 1: r  2: g  3: b }        (0 is sent as 1: a zero field would be left out)
GLOW_STYLES = ("pulse", "spin", "flash", "solid", "rainbow")
GLOW_CHUNK = 20                 # bytes per packet (a 23-byte MTU minus the 3-byte header)
GLOW_MAX_BYTES = 255 * GLOW_CHUNK
RAINBOW = ("#ff0000", "#ffa200", "#eeff00", "#2bff00", "#1499ff", "#ff1ff8")


def led_count(hardware):
    """LEDs in the ring: 10 on the small puck ("Sensor-SM-V1", e.g. PRO 2 21 oz), 13 on the large."""
    return 10 if "sensor-sm" in str(hardware or "").lower() else 13


def _rgb(hex_colour):
    h = str(hex_colour or "").lstrip("#")
    if len(h) != 6:
        return (0, 0, 0)
    try:
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return (0, 0, 0)


def _mix(a, b, t):
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _ring(colours, n):
    """n LED colours going once round the ring through `colours` and back to the first."""
    stops = [_rgb(c) for c in colours] or [(255, 255, 255)]
    out = []
    for i in range(n):
        pos = i * len(stops) / n
        k = int(pos)
        out.append(_mix(stops[k % len(stops)], stops[(k + 1) % len(stops)], pos - k))
    return out


def glow_pattern(style, colours, leds=10):
    """A pattern as plain data: {"sequence": [(group, scale, repeats, delay)], "groups": {id: [frame]}}
    with frame = (ticks_shown, transition_ticks, [(r, g, b)] * leds)."""
    style = style if style in GLOW_STYLES else "pulse"
    ring = _ring(RAINBOW if style == "rainbow" else colours, leds)
    dark = [(0, 0, 0)] * leds
    spin = [(5, 5, ring[-k:] + ring[:-k] if k else ring) for k in range(leds)]
    if style == "pulse":          # fade up, hold, fade out; three times
        return {"sequence": [(1, 100, 0, 5), (1, 100, 0, 5), (1, 100, 0, 0)],
                "groups": {1: [(10, 90, dark), (30, 90, ring), (10, 90, dark)]}}
    if style == "flash":          # quick on / off
        return {"sequence": [(1, 100, 4, 0)], "groups": {1: [(5, 20, dark), (5, 20, ring)]}}
    if style == "solid":          # on for a while, then fade out
        return {"sequence": [(1, 100, 0, 0)], "groups": {1: [(10, 40, dark), (200, 60, ring), (10, 10, dark)]}}
    return {"sequence": [(1, 50, 5, 5), (1, 50, 5, 5), (1, 50, 5, 0)], "groups": {1: spin}}     # spin / rainbow


def _varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _pb_int(field, value):
    return (_varint(field << 3) + _varint(int(value))) if value else b""      # proto3: zeros are omitted


def _pb_msg(field, body):
    return _varint(field << 3 | 2) + _varint(len(body)) + body


def encode_glow_pattern(pattern):
    """The protobuf bytes for a pattern from glow_pattern()."""
    out = _pb_int(1, len(pattern["sequence"]))
    for group, scale, repeats, delay in pattern["sequence"]:
        out += _pb_msg(2, _pb_int(1, group) + _pb_int(2, scale) + _pb_int(3, repeats) + _pb_int(4, delay))
    out += _pb_int(3, len(pattern["groups"]))
    for gid, frames in sorted(pattern["groups"].items()):
        body = _pb_int(1, gid) + _pb_int(2, len(frames))
        for shown, transition, colours in frames:
            leds = b"".join(_pb_msg(4, _pb_int(1, max(1, r)) + _pb_int(2, max(1, g)) + _pb_int(3, max(1, b)))
                            for r, g, b in colours)
            body += _pb_msg(3, _pb_int(2, shown) + _pb_int(3, transition) + leds)
        out += _pb_msg(4, body)
    return out


def glow_packets(data):
    """(count byte, [packets]) for an encoded pattern: the count is written to CHAR_GLOW_COUNT,
    then each packet to CHAR_GLOW_DATA, in order."""
    if not data or len(data) > GLOW_MAX_BYTES:
        raise ValueError(f"pattern is {len(data)} bytes; the bottle takes 1..{GLOW_MAX_BYTES}")
    chunks = [data[i:i + GLOW_CHUNK] for i in range(0, len(data), GLOW_CHUNK)]
    return bytes([len(chunks)]), chunks


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
    """Drops a sip already seen: the same time (+/- `window_s` seconds) and about the same
    volume (+/- `volume_tol` mL). The tolerance covers a record first logged from the
    whole-percent byte and replayed later with the more precise weight-based volume."""

    def __init__(self, window_s=2.0, keep=200, volume_tol=0):
        self.window_s = window_s
        self.volume_tol = volume_tol
        self.seen = deque(maxlen=keep)

    def seed(self, sips):
        for ts, vol in sips:
            self.seen.append((float(ts), int(vol)))

    def is_new(self, ts, volume_ml):
        for t, v in self.seen:
            if abs(v - volume_ml) <= self.volume_tol and abs(t - ts) <= self.window_s:
                return False
        self.seen.append((float(ts), int(volume_ml)))
        return True
