"""A stand-in Muse S Athena: runs muse_worker.py with a fake bleak that "finds" one headband and streams
packets in the real format (built with muse_protocol.build_packet), so the whole dashboard runs without
the headband or Bluetooth:  python3 fake_muse.py  (test_muse_dashboard.py runs it with the API).

What it sends, at the real rates and in the real packing:
  EEG 8 ch, 256 Hz   TP9 / TP10: 10 Hz alpha; AF7 / AF8: 18 Hz beta and a blink every ~4 s; 60 Hz mains
                     on all; AUX: noise
  optics 16 ch, 64 Hz  a 66 bpm pulse on the IR / NIR channels, breathing, and a slow oxygenation swing
                     (left side, 60 s period) for the fNIRS chart
  motion, 52 Hz      gravity on Z and a slow nod; battery every 5 s; the control answers (fw, bp, hn)
"""
import asyncio
import math
import random
import sys
import time
import types

import muse_protocol as mp

NAME, ADDRESS = "MuseS-Demo-0000", "STAND-IN-MUSE"
random.seed(7)


class FakeDevice:
    name, address = NAME, ADDRESS


class FakeAdv:
    local_name, rssi = NAME, -58


class FakeScanner:
    def __init__(self, callback=None, **_kw):
        self.callback = callback

    async def __aenter__(self):
        if self.callback:
            self.callback(FakeDevice(), FakeAdv())
        return self

    async def __aexit__(self, *a):
        return False

    @staticmethod
    async def find_device_by_address(address, **_kw):
        return FakeDevice() if address == ADDRESS else None


class Headband:
    """The signals, sample by sample, as device counts."""

    def __init__(self):
        self.t0 = time.time()

    @staticmethod
    def eeg(t):
        mains = 10 * math.sin(2 * math.pi * 60 * t)
        blink = 150 * math.exp(-((t % 6.5) - 0.3) ** 2 / 0.01) + 120 * math.exp(-((t % 9.7) - 0.5) ** 2 / 0.01)   # irregular
        uv = [20 * math.sin(2 * math.pi * 10 * t) + 6 * random.gauss(0, 1) + mains,
              8 * math.sin(2 * math.pi * 18 * t) + 6 * random.gauss(0, 1) + mains + blink,
              8 * math.sin(2 * math.pi * 18 * t + 1) + 6 * random.gauss(0, 1) + mains + blink,
              18 * math.sin(2 * math.pi * 10 * t + 2) + 6 * random.gauss(0, 1) + mains] + [3 * random.gauss(0, 1) for _ in range(4)]
        return [max(0, min(16383, int(round((725 + u) / mp.EEG_SCALE)))) for u in uv]

    @staticmethod
    def optics(t):
        pulse = math.sin(math.pi * ((t * 1.1) % 1.0)) ** 8          # 66 bpm
        breath = math.sin(2 * math.pi * t / 4.5)
        swing = math.sin(2 * math.pi * t / 60)                        # slow oxygenation change, left side
        vals = []
        for name in mp.OPTICS_CHANNELS:
            base = {"NIR": 0.45, "IR": 0.55, "RED": 0.35, "AMB": 0.02}[name.split("_")[1]]
            v = base * (1 - (0.006 * pulse + 0.002 * breath) * (name.split("_")[1] != "AMB"))
            if name.startswith("L") and name.endswith("_NIR"):
                v *= 1 - 0.004 * swing
            if name.startswith("L") and name.endswith("_IR"):
                v *= 1 + 0.002 * swing
            vals.append(max(0, min((1 << 20) - 1, int(v * 32768 * (1 + 0.0004 * random.gauss(0, 1))))))
        return vals

    @staticmethod
    def imu(t):
        nod = 0.15 * math.sin(2 * math.pi * 0.1 * t)
        acc = [math.sin(nod), 0.02 * math.sin(2 * math.pi * 0.05 * t), math.cos(nod)]
        gyro = [0.15 * 2 * math.pi * 0.1 * math.cos(2 * math.pi * 0.1 * t) * 180 / math.pi + random.gauss(0, 0.3), random.gauss(0, 0.3), random.gauss(0, 0.3)]
        return [int(round(a / mp.ACC_SCALE)) for a in acc] + [int(round(g / mp.GYRO_SCALE)) for g in gyro]


class FakeClient:
    def __init__(self, device, **_kw):
        self.notify, self.task, self.band, self.index = {}, None, Headband(), 0
        self.layout = (8, 16)                          # EEG and optics channels: set by the preset, as the headband does

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        if self.task:
            self.task.cancel()
        return False

    async def start_notify(self, uuid, cb):
        self.notify[uuid] = cb

    async def write_gatt_char(self, uuid, data, response=False):
        cmd = bytes(data)[1:].decode().strip()
        if cmd.startswith("p") and cmd[1:].isdigit():
            from muse_worker import PRESET_LAYOUT
            self.layout = PRESET_LAYOUT.get(cmd, (8, 16))
        if cmd == "s" and mp.CONTROL_UUID in self.notify:
            text = f'{{"hn":"{NAME}","fw":"3.2.1-demo","bp":87.5,"ps":3}}'
            for i in range(0, len(text), 20):                     # the control answer arrives in pieces
                self.notify[mp.CONTROL_UUID](None, bytearray(text[i:i + 20].encode()))
        if cmd == "dc001" and self.task is None:
            self.task = asyncio.create_task(self._stream())
        if cmd == "h" and self.task:
            self.task.cancel()
            self.task = None

    def _packet(self, tag, t_first, samples, per_block, width, nbytes):
        """Packets for one sensor: blocks of per_block samples, as many as fit in a 255-byte packet."""
        out, blocks = [], [samples[i:i + per_block] for i in range(0, len(samples), per_block)]
        room = (255 - mp.PACKET_HEADER - nbytes) // (mp.BLOCK_HEADER + nbytes) + 1
        rate = mp.SENSORS[tag][3]
        k = 0
        while k < len(blocks):
            chunk = blocks[k:k + room]
            tick = (t_first + k * per_block / rate) * mp.CLOCK_HZ
            data = [self._block(tag, b, width, nbytes) for b in chunk]
            out.append(mp.build_packet(tag, self.index, tick, data[0], [(tag, (self.index + j) & 0xFF, d) for j, d in enumerate(data[1:], 1)]))
            self.index = (self.index + 1) & 0xFF
            k += len(chunk)
        return out

    @staticmethod
    def _block(tag, rows, width, nbytes):
        if width == 16:
            import struct
            return struct.pack("<18h", *[v for r in rows for v in r])
        return mp.pack([v for r in rows for v in r], width, nbytes)

    async def _stream(self):
        start = time.time()
        sent = {"eeg": 0, "optics": 0, "imu": 0}
        rates = {"eeg": 256, "optics": 64, "imu": 52}
        last_bat = 0.0
        while True:
            await asyncio.sleep(0.05)
            now = time.time() - start
            pkts = []
            eeg_n, opt_n = self.layout
            names = mp.OPTICS_BY_COUNT.get(opt_n, ())
            pick = [mp.OPTICS_CHANNELS.index(n) for n in names]
            streams = [("eeg", 0x12 if eeg_n == 8 else 0x11, 2 if eeg_n == 8 else 4, 14, 28, lambda t: Headband.eeg(t)[:eeg_n]),
                       ("imu", 0x47, 3, 16, 36, Headband.imu)]
            if opt_n:
                tag = {4: 0x34, 8: 0x35, 16: 0x36}[opt_n]
                streams.append(("optics", tag, mp.SENSORS[tag][2], 20, mp.SENSORS[tag][4], lambda t: [Headband.optics(t)[i] for i in pick]))
            for kind, tag, per, width, nbytes, gen in streams:
                due = int(now * rates[kind]) // per * per
                if due > sent[kind]:
                    first = sent[kind] / rates[kind]
                    rows = [gen((sent[kind] + i) / rates[kind]) for i in range(due - sent[kind])]
                    pkts += self._packet(tag, first, rows, per, width, nbytes)
                    sent[kind] = due
            if now - last_bat > 5:
                last_bat = now
                import struct
                pkts.append(mp.build_packet(0x98, self.index, now * mp.CLOCK_HZ, struct.pack("<H", int(87.5 * 256)) + bytes(18)))
            uuids = mp.DATA_UUIDS
            for i, p in enumerate(pkts):
                cb = self.notify.get(uuids[i % 2])
                if cb:
                    cb(None, bytearray(p))


fake_bleak = types.ModuleType("bleak")
fake_bleak.BleakScanner, fake_bleak.BleakClient = FakeScanner, FakeClient
sys.modules["bleak"] = fake_bleak

if __name__ == "__main__":
    import muse_worker
    import bt_debug

    async def no_adapters():
        return []
    bt_debug.list_adapters = no_adapters
    muse_worker.debug["versions"]["bleak"] = "stand-in (fake_muse.py)"
    try:
        asyncio.run(muse_worker.main())
    except KeyboardInterrupt:
        pass
    finally:
        muse_worker.session.close()
