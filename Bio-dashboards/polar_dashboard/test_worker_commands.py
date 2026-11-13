#!/usr/bin/env python3
"""The worker's UI commands with a stand-in strap and scanner (no Bluetooth): Disconnect ends the session and
holds auto-reconnect until a device is picked; picking another device while streaming switches to it.
Takes ~25 s (real scan sweeps). Run with the dashboards' Python (it needs bleak and polar_python to import)."""
import asyncio
import json
import os
import sys
import tempfile
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
TMP = tempfile.mkdtemp()
os.environ["BIODASH_DATA_DIR"] = os.path.join(TMP, "data")
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.dirname(HERE))
import advanced_worker as w  # noqa: E402
import bt_debug  # noqa: E402

for name in ("COMMAND_FILE", "STATUS_FILE", "DEVICES_FILE"):
    setattr(w, name, os.path.join(TMP, os.path.basename(getattr(w, name))))
w.last_device = bt_debug.LastDevice(os.path.join(TMP, "last_device.json"))

A, B = "TEST-STRAP-A1", "TEST-STRAP-B2"     # stand-in addresses (not MAC-shaped: scrub_repo.py)
events = []


class Dev:
    def __init__(self, a):
        self.address, self.name = a, "Polar H10 " + a[-2:]


DEVS = {a: Dev(a) for a in (A, B)}


class FakeScanner:
    def __init__(self, cb, **_):
        self.cb = cb

    async def __aenter__(self):
        for d in DEVS.values():
            self.cb(d, types.SimpleNamespace(local_name=d.name, rssi=-60))
        return self

    async def __aexit__(self, *_):
        pass

    @staticmethod
    async def find_device_by_address(mac, **_):
        return DEVS.get(mac)


class FakePolar:
    def __init__(self, d):
        self.d = d

    async def __aenter__(self):
        events.append(("connect", self.d.address))
        return self

    async def __aexit__(self, *_):
        events.append(("disconnect", self.d.address))

    async def start_hr_stream(self, cb):
        pass

    async def start_acc_stream(self, cb, *_):
        pass

    async def start_ecg_stream(self, cb, *_):
        pass


w.BleakScanner, w.PolarDevice = FakeScanner, FakePolar


def command(c):
    with open(w.COMMAND_FILE, "w") as f:
        json.dump(c, f)


async def scenario():
    async def beats():                       # the strap keeps "sending": the watchdog never fires
        while True:
            w.last_heartbeat = time.time()
            await asyncio.sleep(0.2)

    w.last_device.remember(A, "Polar H10 01")        # auto-reconnect target, in range
    worker, heart = asyncio.create_task(w.main()), asyncio.create_task(beats())
    await asyncio.sleep(5)
    events.append(("ui", "disconnect"))
    command({"action": "disconnect"})
    await asyncio.sleep(8)                   # two scan sweeps with A in range: must not reconnect by itself
    events.append(("ui", "pick B"))
    command({"action": "connect", "address": B})
    await asyncio.sleep(6)
    events.append(("ui", "pick A while on B"))
    command({"action": "connect", "address": A})
    await asyncio.sleep(6)
    worker.cancel()
    heart.cancel()


asyncio.run(scenario())
got = [e for e in events]
want = [("connect", A), ("ui", "disconnect"), ("disconnect", A), ("ui", "pick B"), ("connect", B),
        ("ui", "pick A while on B"), ("disconnect", B), ("connect", A)]
ok = 0
for i, (g, e) in enumerate(zip(got, want)):
    good = g == e
    print(("  ok  " if good else "  FAIL ") + f"{i + 1}. {' '.join(e)}" + ("" if good else f"  (got {' '.join(g)})"))
    if not good:
        sys.exit(1)
    ok += 1
if len(got) < len(want):
    print(f"  FAIL only {len(got)} events: {got}")
    sys.exit(1)
print(f"{ok} checks passed")
