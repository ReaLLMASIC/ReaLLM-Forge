#!/usr/bin/env python3
"""Checks for Hydro's "any radio" mode with stand-in radios (no Bluetooth needed).
Run: python3 test_any_radio.py"""
import asyncio
import os
import sys
import tempfile
import time

tmp = tempfile.mkdtemp()
os.environ["XDG_CONFIG_HOME"] = tmp
os.environ["BIODASH_DATA_DIR"] = tmp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the shared bt_debug.py
import hydro_worker as w  # noqa: E402

ok = 0


def check(name, got, want):
    global ok
    if got != want:
        raise AssertionError(f"{name}: got {got!r}, want {want!r}")
    ok += 1


RADIOS = [{"name": "hci0", "address": "RADIO-A", "powered": True},
          {"name": "hci1", "address": "RADIO-B", "powered": True},
          {"name": "hci2", "address": "RADIO-C", "powered": False}]
w.debug["adapters"] = RADIOS
w._adapters_at = time.time() + 3600        # don't ask BlueZ


class Dev:
    def __init__(self, address, name):
        self.address, self.name = address, name


class Adv:
    def __init__(self, rssi, name):
        self.rssi, self.local_name = rssi, name
        self.service_uuids, self.manufacturer_data = [], {}


HEARD = {"hci0": -78, "hci1": -55}          # what each stand-in radio hears the bottle at
started, stopped = [], []


class FakeScanner:
    def __init__(self, callback, bluez=None, **kw):
        self.cb, self.hci = callback, (bluez or {}).get("adapter")

    async def start(self):
        started.append(self.hci)
        rssi = HEARD.get(self.hci)
        if rssi is not None:
            asyncio.get_running_loop().call_later(0.05 if self.hci == "hci0" else 0.3,
                                                  self.cb, Dev("11:22", "h2o00001234"), Adv(rssi, "h2o00001234"))

    async def stop(self):
        stopped.append(self.hci)

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *a):
        await self.stop()


w.BleakScanner = FakeScanner

# the default is "any", from a new file (a pin saved by an older version doesn't carry over)
check("default is any", w.radio_pin.config(), "any")
check("new settings file", os.path.basename(w.radio_pin.path), "radio_choice.json")
w.radio_pin.path = os.path.join(tmp, "radio_choice.json")     # never touch the real ones
w.RADIO_AVOID_FILE = os.path.join(tmp, "radio_avoid.json")
check("powered radios only", [a["name"] for a in w.any_radio()], ["hci0", "hci1"])

# pick: strongest wins, a recent failure is passed over, the last good radio gets a head start
check("strongest", w.pick_radio({"hci0": (None, -78), "hci1": (None, -55)})[0], "hci1")
w.radio_failed["RADIO-B"] = time.time() + 600
hci, why = w.pick_radio({"hci0": (None, -78), "hci1": (None, -55)})
check("failed recently: passed over", hci, "hci0")
check("and says so", "failed recently" in why, True)
check("only one heard it: used anyway", w.pick_radio({"hci1": (None, -55)})[0], "hci1")
w.radio_failed.clear()
w.radio_good = "RADIO-A"
check("last good gets +6 dB", w.pick_radio({"hci0": (None, -60), "hci1": (None, -56)})[0], "hci0")
check("but not more", w.pick_radio({"hci0": (None, -70), "hci1": (None, -56)})[0], "hci1")
w.radio_good = None

# the connect loop: listens on both, waits for the slower radio, connects through the better one
used = {}


async def fake_session(device, on_connected=None):
    used["hci"] = w.radio_hci
    used["device"] = device.address
    if on_connected:
        await on_connected()
    raise asyncio.CancelledError          # end the test here


w.run_session = fake_session
w.selected_target = {"address": "11:22", "name": "h2o00001234"}
w.ANY_SETTLE_S = 0.5
w.set_status = lambda *a, **k: None
started.clear(); stopped.clear()
try:
    asyncio.run(w.main())
except asyncio.CancelledError:
    pass
check("listened on both powered radios", sorted(started), ["hci0", "hci1"])
check("connected through the stronger one, though it heard the bottle second", used["hci"], "hci1")
check("weaker radio stopped first, chosen one when the link came up", stopped, ["hci0", "hci1"])
check("reason shown in the debug panel", w.debug["radio"]["reason"].startswith("any radio: hci1 heard it best (-55 dBm)"), True)

# failures: an empty service list (Bluetooth 4.0) is avoided for a week and remembered; others 10 min
w.debug["adapters"] = RADIOS
w.event = lambda *a, **k: None
w.radio_failed.clear()
w.avoid_radio("hci0", RuntimeError("connected, but the bottle's service list came back empty"))
w.avoid_radio("hci1", RuntimeError("TimeoutError"))
left = {k: round((v - time.time()) / 60) for k, v in w.radio_failed.items()}
check("week vs 10 min", (left["RADIO-A"] > 60 * 24 * 6, left["RADIO-B"]), (True, 10))
w.radio_failed.clear()
w.load_radio_avoid()
check("only the long one is remembered across a restart", list(w.radio_failed), ["RADIO-A"])
w.radio_failed.clear()

# a pin still pins
w.radio_pin.save("RADIO-A")
check("pinned: no any-radio", w.any_radio(), None)
check("pinned resolves", w.radio_pin.resolve(RADIOS)[0], "hci0")
w.radio_pin.save("any")
# one radio only: plain single-radio scanning
w.debug["adapters"] = RADIOS[:1]
check("one radio: no any-radio", w.any_radio(), None)

print(f"{ok} checks passed")
