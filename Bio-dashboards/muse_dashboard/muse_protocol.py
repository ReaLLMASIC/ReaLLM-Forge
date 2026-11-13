"""The Muse S Athena's Bluetooth protocol: commands, packets, and the sensors' bit packing.

Interaxon doesn't publish it. This follows OpenMuse (github.com/DominiqueMakowski/OpenMuse, MIT), which
worked it out from recordings and validated it, crediting amused-py (github.com/Amused-EEG/amused-py, MIT) and
AbosaSzakal's MuseAthenaDataformatParser. See PROTOCOL.md.

Commands go to the control characteristic as length-prefixed ASCII lines ("p1041\\n" -> b"\\x06p1041\\n").
Data arrives as notifications on two characteristics; each notification holds one or more packets:

    packet:   [len][index][device clock, u32 LE, 256 kHz][3 bytes][id][3 bytes][1 byte]   14-byte header
              then the first block of sensor `id` (no tag), then tagged blocks:
              [tag][index][3 bytes][data]  ...  until the packet ends

    EEG       14-bit unsigned, packed LSB first; x 1450/16383 -> uV       4 ch x 4 or 8 ch x 2 per block, 256 Hz
    optics    20-bit unsigned, packed LSB first; / 32768                 4 ch x 3, 8 x 2 or 16 x 1, 64 Hz
    acc/gyro  int16 LE, 3 samples x [ax ay az gx gy gz]; g and deg/s     52 Hz
    battery   u16 LE / 256 -> %                                            every few seconds

Standard library only (the worker and the tests use it; numpy is only for the signal maths).
"""
import re
import struct

CONTROL_UUID = "273e0001-4c4d-454d-96be-f03bac821358"
DATA_UUIDS = ("273e0013-4c4d-454d-96be-f03bac821358", "273e0014-4c4d-454d-96be-f03bac821358")

CLOCK_HZ = 256_000.0
PACKET_HEADER = 14
BLOCK_HEADER = 5            # tag + index + 3 bytes

# tag -> (sensor, channels, samples per block, rate Hz, data bytes)
SENSORS = {
    0x11: ("eeg", 4, 4, 256.0, 28),
    0x12: ("eeg", 8, 2, 256.0, 28),
    0x34: ("optics", 4, 3, 64.0, 30),
    0x35: ("optics", 8, 2, 64.0, 40),
    0x36: ("optics", 16, 1, 64.0, 40),
    0x47: ("imu", 6, 3, 52.0, 36),
    0x53: ("unknown", 0, 0, 0.0, 24),
    0x88: ("battery", 1, 1, 0.0, 188),       # newer firmware: battery + status, 188-230 bytes; the first 2 count
    0x98: ("battery", 1, 1, 0.0, 20),
}
RATES = {"eeg": 256.0, "optics": 64.0, "imu": 52.0}

EEG_SCALE = 1450.0 / 16383.0                 # uV per count (0 .. 1450 uV, mid-scale ~725)
OPTICS_SCALE = 1.0 / 32768.0
ACC_SCALE = 0.0000610352                     # g per count (+-2 g)
GYRO_SCALE = -0.0074768                      # deg/s per count (+-250 dps)

EEG_CHANNELS = ("TP9", "AF7", "AF8", "TP10", "AUX1", "AUX2", "AUX3", "AUX4")
MAIN_EEG = EEG_CHANNELS[:4]
# Optics: L/R = left / right, O / I = outer / inner optode; NIR 730 nm, IR 850 nm, RED, AMB = ambient
OPTICS_CHANNELS = ("LO_NIR", "RO_NIR", "LO_IR", "RO_IR", "LI_NIR", "RI_NIR", "LI_IR", "RI_IR",
                   "LO_RED", "RO_RED", "LO_AMB", "RO_AMB", "LI_RED", "RI_RED", "LI_AMB", "RI_AMB")
OPTICS_BY_COUNT = {4: OPTICS_CHANNELS[4:8], 8: OPTICS_CHANNELS[:8], 16: OPTICS_CHANNELS}
IMU_CHANNELS = ("ACC_X_g", "ACC_Y_g", "ACC_Z_g", "GYRO_X_dps", "GYRO_Y_dps", "GYRO_Z_dps")

# Presets (OpenMuse's table, from the packets each one produces; more may exist)
PRESETS = {
    "p1041": "EEG 8 ch + optics 16 ch (all wavelengths, both sides) + motion + battery, red LED bright",
    "p1034": "EEG 8 ch + optics 8 ch + motion + battery, red LED bright",
    "p1044": "EEG 8 ch + optics 8 ch + motion + battery, red LED dim",
    "p1045": "EEG 8 ch + optics 4 ch (inner) + motion + battery, red LED dim",
    "p1035": "EEG 4 ch + optics 4 ch (inner) + motion + battery, red LED dim",
    "p21": "EEG 4 ch + motion + battery, no optics (no light on the forehead)",
}
DEFAULT_PRESET = "p1041"
START_SEQUENCE = ("v6", "s", "h", "{preset}", "s", "dc001", "dc001", "L1", "s")   # as OpenMuse.MuseS.initialize_device


def encode_command(token):
    """'p1041' -> b'\\x06p1041\\n' (length-prefixed ASCII line)."""
    payload = (token + "\n").encode("ascii")
    if not token or len(payload) > 255:
        raise ValueError("bad command")
    return bytes([len(payload)]) + payload


def unpack(data, width, count):
    """`count` unsigned `width`-bit values packed LSB first (bit i of byte k is bit 8k+i of the stream)."""
    v = int.from_bytes(data, "little")
    mask = (1 << width) - 1
    return [(v >> (i * width)) & mask for i in range(count)]


def pack(values, width, nbytes):
    """The inverse of unpack (the stand-in Muse and the tests)."""
    v = 0
    for i, x in enumerate(values):
        v |= (int(x) & ((1 << width) - 1)) << (i * width)
    return v.to_bytes(nbytes, "little")


def decode_block(tag, data):
    """One sensor block -> rows (one per sample), or None for a block this doesn't decode."""
    kind, ch, n, _, need = SENSORS[tag]
    if kind == "eeg":
        if len(data) < 28:
            return None
        vals = unpack(data[:28], 14, ch * n)
        return [[round(vals[s * ch + c] * EEG_SCALE, 3) for c in range(ch)] for s in range(n)]
    if kind == "optics":
        if len(data) < need:
            return None
        vals = unpack(data[:need], 20, ch * n)
        return [[round(vals[s * ch + c] * OPTICS_SCALE, 6) for c in range(ch)] for s in range(n)]
    if kind == "imu":
        if len(data) < 36:
            return None
        v = struct.unpack_from("<18h", data)
        return [[round(v[s * 6 + i] * ACC_SCALE, 5) for i in range(3)] + [round(v[s * 6 + i] * GYRO_SCALE, 3) for i in range(3, 6)]
                for s in range(3)]
    if kind == "battery":
        if len(data) < 2:
            return None
        return [[round(struct.unpack_from("<H", data)[0] / 256.0, 2)]]
    return None


def parse_notification(payload):
    """A notification -> [{"kind", "tag", "channels", "pkt_index", "tick", "rows"}, ...] in packet order.
    Unknown blocks end their packet (their length isn't known); a malformed packet ends the notification."""
    out = []
    off = 0
    while off + PACKET_HEADER <= len(payload):
        plen = payload[off]
        if plen < PACKET_HEADER or off + plen > len(payload):
            break
        pkt = payload[off:off + plen]
        off += plen
        pid = pkt[9]
        if pid not in SENSORS:
            continue
        index, tick = pkt[1], struct.unpack_from("<I", pkt, 2)[0]
        body = pkt[PACKET_HEADER:]
        blocks = []
        need = SENSORS[pid][4]
        if need <= len(body):
            blocks.append((pid, body[:need]))
            pos = need
            while pos + BLOCK_HEADER <= len(body):
                tag = body[pos]
                if tag not in SENSORS:
                    break
                n = SENSORS[tag][4]
                if pos + BLOCK_HEADER + n > len(body):
                    # newer firmware's battery block varies in length: take what's there
                    if SENSORS[tag][0] == "battery" and len(body) - pos - BLOCK_HEADER >= 2:
                        blocks.append((tag, body[pos + BLOCK_HEADER:]))
                    break
                blocks.append((tag, body[pos + BLOCK_HEADER:pos + BLOCK_HEADER + n]))
                pos += BLOCK_HEADER + n
        elif SENSORS[pid][0] == "battery" and len(body) >= 2:
            blocks.append((pid, body))
        for tag, data in blocks:
            rows = decode_block(tag, data)
            if rows:
                kind, ch = SENSORS[tag][0], SENSORS[tag][1]
                out.append({"kind": kind, "tag": tag, "channels": ch, "pkt_index": index, "tick": tick, "rows": rows})
    return out


def build_packet(pkt_id, index, tick, first, blocks=()):
    """A packet as the headband sends it (the stand-in Muse and the tests): first = the id's block data,
    blocks = [(tag, block_index, data), ...]."""
    body = bytes(first) + b"".join(bytes([tag, bi & 0xFF, 0, 0, 0]) + bytes(d) for tag, bi, d in blocks)
    n = PACKET_HEADER + len(body)
    if n > 255:
        raise ValueError("packet too long")
    return bytes([n, index & 0xFF]) + struct.pack("<I", int(tick) & 0xFFFFFFFF) + b"\0\0\0" + bytes([pkt_id]) + b"\0\0\0\0" + body


# ---- the control characteristic's answers: JSON-ish text in pieces ("fw", "bp", "hn", ...)
_FIELD = re.compile(r'"(\w+)"\s*:\s*("([^"]*)"|-?[\d.]+)')


def parse_control(text):
    """The fields seen in the control answers so far: {"fw": "...", "bp": 87.0, "hn": "MuseS-1A2B", ...}."""
    out = {}
    for m in _FIELD.finditer(text):
        k, raw, s = m.group(1), m.group(2), m.group(3)
        if s is not None:
            out[k] = s
        else:
            try:
                out[k] = float(raw)
            except ValueError:
                pass
    return out


class MuseClock:
    """Epoch-ms times for each sensor's samples, from the headband's 256 kHz clock.

    Each stream counts its samples at the nominal rate, re-synced to the device clock when the clock says
    samples went missing (a gap) and never going backwards (blocks of one packet share its time). One
    offset maps the device clock to the computer's clock for every stream, so EEG, optics and motion line
    up; as the H10's ClockAnchor it is set at the first packet, slewed 1 ms per packet when the two drift
    more than 250 ms apart, and re-set after a jump of over 5 s (the headband restarted its clock)."""

    REANCHOR_MS, SLEW_FROM_MS, SLEW_MS = 5000.0, 250.0, 1.0
    WRAP = 1 << 32

    def __init__(self):
        self.reset()

    def reset(self):
        self.offset_ms = None
        self.reanchors = 0
        self.streams = {}          # kind -> {"base": abs tick, "count": samples so far, "last": abs tick, "wrap": n}
        self.dropped = {}

    def _abs_tick(self, st, tick):
        prev = st["last"] % self.WRAP
        if tick < prev and prev - tick > self.WRAP // 2:
            st["wrap"] += self.WRAP                               # the 32-bit clock wrapped (every ~4.7 h)
        a = tick + st["wrap"]
        if a < st["last"]:
            a = st["last"]                                        # small inversion: keep order
        st["last"] = a
        return a

    def stamps(self, kind, tick, n, now_ms):
        rate = RATES[kind]
        st = self.streams.get(kind)
        if st is None:
            st = self.streams[kind] = {"base": tick, "count": 0, "last": tick, "wrap": 0}
            self.dropped[kind] = 0
        a = self._abs_tick(st, tick)
        idx = int(round((a - st["base"]) / CLOCK_HZ * rate))
        if idx > st["count"]:
            self.dropped[kind] += idx - st["count"]
        idx = max(idx, st["count"])
        st["count"] = idx + n
        dev_ms = [(st["base"] / CLOCK_HZ + (idx + i) / rate) * 1000.0 for i in range(n)]
        target = now_ms - dev_ms[-1]
        if self.offset_ms is None or abs(target - self.offset_ms) > self.REANCHOR_MS:
            if self.offset_ms is not None:
                self.reanchors += 1
            self.offset_ms = target
        else:
            err = target - self.offset_ms
            if abs(err) > self.SLEW_FROM_MS:
                self.offset_ms += max(-self.SLEW_MS, min(self.SLEW_MS, err))
        return [int(d + self.offset_ms) for d in dev_ms]
