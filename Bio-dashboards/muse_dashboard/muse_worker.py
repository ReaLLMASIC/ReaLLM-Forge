"""Muse S Athena worker: finds the headband, starts its streams and records them (see muse_protocol.py).

Same shape as the other dashboards' workers: it scans and lists headbands for the page (logs_muse/devices.json),
takes commands from it (command.json: connect / disconnect / preset / zero), reports what it's doing
(status.json, with the debug panel's numbers) and writes the session's CSVs, which api.py follows:

    <Documents>/Bio-dash/Muse/<headband>/<date>/<time>_<stream>.csv
      eeg      Timestamp_Epoch_ms, TP9_uV, AF7_uV, AF8_uV, TP10_uV [, AUX1_uV .. AUX4_uV]     256 Hz
      optics   Timestamp_Epoch_ms, <LO_NIR ...> (4, 8 or 16 channels, by preset)              64 Hz
      imu      Timestamp_Epoch_ms, ACC_X_g .. ACC_Z_g, GYRO_X_dps .. GYRO_Z_dps                52 Hz
      bands    Timestamp_Epoch_ms, <channel>_<band> (log10 uV^2) x 4 x 5, <channel>_contact   1 / s
      heart    Timestamp_Epoch_ms, IBI_ms, HR_BPM                                             per beat
      fnirs    Timestamp_Epoch_ms, L_HbO_uM, L_HbR_uM, R_HbO_uM, R_HbR_uM                     1 / s
      battery  Timestamp_Epoch_ms, Battery_pct                                                 as reported
"""
import asyncio
import json
import logging
import os
import sys
import time
from importlib.metadata import version as _pkg_version

from bleak import BleakClient, BleakScanner

import os as _os, sys as _sys
_sys.path.insert(1, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # the shared bt_debug.py, in the release folder
import bt_debug
import muse_protocol as mp
import muse_signal as ms

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S",
                    handlers=[logging.StreamHandler(sys.stdout)])

BASE_DIR = bt_debug.state_dir(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(BASE_DIR, "logs_muse")
os.makedirs(LOGS_DIR, exist_ok=True)
DEVICES_FILE = os.path.join(LOGS_DIR, "devices.json")
COMMAND_FILE = os.path.join(LOGS_DIR, "command.json")
STATUS_FILE = os.path.join(LOGS_DIR, "status.json")
SETTINGS_FILE = os.path.join(BASE_DIR, "muse_settings.json")

# The Polar H10 can take over the radio it streams on, so "auto" leaves an external dongle to it
radio_pin = bt_debug.RadioPin(os.path.join(BASE_DIR, "radio_config.json"), prefer="internal")
last_device = bt_debug.LastDevice(os.path.join(BASE_DIR, "last_device.json"))
session = bt_debug.SessionFiles("Muse")

CONNECT_ATTEMPTS = 3
WATCHDOG_S = 10.0
NAME_HINTS = ("muse",)            # "MuseS-1A2B", "Muse-1A2B"

# EEG and optics channel counts each preset gives (the files' columns); unknown presets: whatever arrives
PRESET_LAYOUT = {"p1041": (8, 16), "p1042": (8, 16), "p1034": (8, 8), "p1043": (8, 8), "p1044": (8, 8),
                 "p1045": (8, 4), "p1046": (8, 4), "p4129": (8, 4), "p1035": (4, 4),
                 "p20": (4, 0), "p21": (4, 0), "p50": (4, 0), "p51": (4, 0), "p60": (4, 0), "p61": (4, 0)}


def load_settings():
    try:
        with open(SETTINGS_FILE) as f:
            s = json.load(f)
        preset = str(s.get("preset") or mp.DEFAULT_PRESET)
    except (OSError, ValueError, AttributeError):
        preset = mp.DEFAULT_PRESET
    return {"preset": preset if preset in PRESET_LAYOUT else mp.DEFAULT_PRESET}


def _write_json(path, obj):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f)
    os.replace(path + ".tmp", path)


meters = {k: bt_debug.RateMeter() for k in ("eeg", "optics", "imu", "packets", "notifications", "beats")}
debug = {
    "adapters": [], "adapter": None, "device": None, "connected_at": None, "recording": None,
    "nominal": {"eeg_hz": 256, "optics_hz": 64, "imu_hz": 52},
    "rates": {}, "info": {}, "preset": None, "layout": None, "signals": {},
    "versions": {}, "radio": {"config": "auto", "hci": None, "reason": None},
}
for _pkg in ("bleak", "dbus-fast", "numpy"):
    try:
        debug["versions"][_pkg] = _pkg_version(_pkg)
    except Exception:
        debug["versions"][_pkg] = None
debug["versions"]["python"] = sys.version.split()[0]

auto_held = False                 # after Disconnect: wait for a pick instead of reconnecting by itself
selected = None
next_target = None
ble_cache, last_rssi = {}, {}


def set_status(state, address=None, attempt=None):
    _write_json(STATUS_FILE, {"state": state, "address": address, "attempt": attempt, "last_device": last_device.load(),
                              "auto_held": auto_held, "max_attempts": CONNECT_ATTEMPTS, "debug": debug,
                              "preset": load_settings()["preset"], "presets": mp.PRESETS})


def read_command():
    if not os.path.exists(COMMAND_FILE):
        return None
    try:
        with open(COMMAND_FILE) as f:
            cmd = json.load(f)
        os.remove(COMMAND_FILE)
        if cmd.get("action") in ("connect", "disconnect", "preset", "zero"):
            return cmd
    except Exception:
        pass
    return None


class Session:
    """One connection: the decoded streams -> files, and the once-a-second maths."""

    def __init__(self, preset):
        self.preset = preset
        self.eeg_n, self.opt_n = PRESET_LAYOUT.get(preset, (8, 16))
        self.clock = mp.MuseClock()
        self.last_data = time.time()
        self.control = ""
        self.unknown_layout = set()
        self.bands = ms.EegBands(mp.MAIN_EEG)
        names = mp.OPTICS_BY_COUNT.get(self.opt_n, ())
        self.heart = ms.PpgHeart(names) if names else None
        self.fnirs = ms.Fnirs(names) if names else None
        self.latest = {}

    def streams(self):
        s = {"eeg": ["Timestamp_Epoch_ms"] + [f"{c}_uV" for c in mp.EEG_CHANNELS[:self.eeg_n]],
             "imu": ["Timestamp_Epoch_ms", *mp.IMU_CHANNELS],
             "bands": ["Timestamp_Epoch_ms"] + [f"{c}_{b[0]}" for c in mp.MAIN_EEG for b in ms.BANDS] + [f"{c}_contact" for c in mp.MAIN_EEG],
             "heart": ["Timestamp_Epoch_ms", "IBI_ms", "HR_BPM"],
             "battery": ["Timestamp_Epoch_ms", "Battery_pct"]}
        if self.opt_n:
            s["optics"] = ["Timestamp_Epoch_ms", *mp.OPTICS_BY_COUNT[self.opt_n]]
            s["fnirs"] = ["Timestamp_Epoch_ms", "L_HbO_uM", "L_HbR_uM", "R_HbO_uM", "R_HbR_uM"]
        return s

    def on_control(self, _char, data):
        self.control = (self.control + bytes(data).decode("utf-8", "ignore"))[-8192:]
        info = mp.parse_control(self.control)
        debug["info"] = {k: info[k] for k in ("fw", "hw", "bp", "hn", "sn", "ps", "rc", "tp", "bn") if k in info}
        if "bp" in info and "battery" not in self.latest:
            session.writerows("battery", [[int(time.time() * 1000), round(info["bp"], 1)]])
            self.latest["battery"] = round(info["bp"], 1)

    def on_data(self, _char, data):
        now = time.time() * 1000
        meters["notifications"].add()
        for blk in mp.parse_notification(bytes(data)):
            meters["packets"].add()
            kind, rows = blk["kind"], blk["rows"]
            if kind == "battery":
                pct = rows[0][0]
                if 0 <= pct <= 100:
                    session.writerows("battery", [[int(now), pct]])
                    self.latest["battery"] = pct
                continue
            want = {"eeg": self.eeg_n, "optics": self.opt_n, "imu": 6}[kind]
            if blk["channels"] != want:
                if (kind, blk["channels"]) not in self.unknown_layout:
                    self.unknown_layout.add((kind, blk["channels"]))
                    logging.warning(f"{kind} with {blk['channels']} channels (this preset's files have {want}): not recorded")
                continue
            ts = self.clock.stamps(kind, blk["tick"], len(rows), now)
            session.writerows(kind, [[t, *r] for t, r in zip(ts, rows)])
            meters[kind].add(len(rows))
            self.last_data = time.time()
            if kind == "eeg":
                self.bands.push([r[:4] for r in rows])
            elif kind == "optics":
                self.heart.push(ts, rows)
                self.fnirs.push(rows)

    def every_second(self):
        now = int(time.time() * 1000)
        b = self.bands.compute()
        if b:
            session.writerows("bands", [[now] + [b[c]["abs"][band[0]] for c in mp.MAIN_EEG for band in ms.BANDS]
                                        + [b[c]["q"] for c in mp.MAIN_EEG]])
            debug["signals"]["contact"] = {c: ms.QUALITY[b[c]["q"]] for c in mp.MAIN_EEG}
        if self.heart:
            beats = self.heart.compute()
            if beats:
                session.writerows("heart", [list(x) for x in beats])
                meters["beats"].add(len(beats))
            debug["signals"]["ppg"] = {"channel": self.heart.channel, "quality": self.heart.quality, "hr": self.heart.hr}
        if self.fnirs:
            f = self.fnirs.compute()
            if f:
                row = [now]
                for side in ("L", "R"):
                    row += list(f.get(side, ("", "")))
                session.writerows("fnirs", [row])
            debug["signals"]["fnirs"] = {"optodes": self.fnirs.optodes,
                                         "baseline_s": min(self.fnirs.base_n, ms.Fnirs.BASELINE_S), "of_s": ms.Fnirs.BASELINE_S}
        session.flush()
        debug["rates"] = {
            "eeg_hz": round(meters["eeg"].rate(), 1), "optics_hz": round(meters["optics"].rate(), 1),
            "imu_hz": round(meters["imu"].rate(), 1), "packets_per_s": round(meters["packets"].rate(), 1),
            "notifications_per_s": round(meters["notifications"].rate(), 1), "beats_per_min": round(meters["beats"].rate() * 60, 1),
            "dropped": dict(self.clock.dropped), "clock_reanchors": self.clock.reanchors,
            "battery": self.latest.get("battery", debug["info"].get("bp")),
        }


async def send(client, token):
    await client.write_gatt_char(mp.CONTROL_UUID, mp.encode_command(token), response=False)


async def start_streams(client, preset):
    """OpenMuse's start: version, status, halt, the preset, status, start (twice), low latency, status."""
    for tok in mp.START_SEQUENCE:
        tok = tok.format(preset=preset)
        try:
            await send(client, tok)
        except Exception as e:
            if tok.startswith("dc"):
                raise
            logging.warning(f"command {tok!r}: {e}")
        await asyncio.sleep(0.05 if tok == "dc001" else 0.2)


async def scan():
    global selected, auto_held
    debug["adapters"] = await bt_debug.list_adapters()
    hci, reason = radio_pin.resolve(debug["adapters"])
    debug["radio"] = {"config": radio_pin.config(), "hci": hci, "reason": reason}
    set_status("scanning")
    found, cut = {}, False

    def seen(device, adv):
        name = device.name or adv.local_name or ""
        if any(h in name.lower() for h in NAME_HINTS):
            ble_cache[device.address] = device
            last_rssi[device.address] = adv.rssi
            found[device.address] = {"name": name, "address": device.address, "rssi": adv.rssi if adv.rssi else -100}

    try:
        async with bt_debug.scanning(seen, hci, scanner=BleakScanner):
            for _ in range(15):
                await asyncio.sleep(0.2)
                if os.path.exists(COMMAND_FILE):
                    cut = True
                    break
    except Exception as e:
        logging.error(f"Scan error: {e}")
    if not cut:
        _write_json(DEVICES_FILE, sorted(found.values(), key=lambda d: d["rssi"], reverse=True))
    cmd = read_command()
    if cmd and cmd["action"] == "connect" and cmd.get("address"):
        selected, auto_held = cmd["address"], False
    elif not auto_held:
        last = last_device.target()
        if last and last["address"] in found:
            selected = last["address"]
            logging.info(f"Reconnecting to the last headband {last.get('name') or ''}")
    return hci


async def stream(device, hci):
    """One connected session. Returns when it should end (watchdog, Disconnect, another pick, a new preset)."""
    global auto_held, next_target
    preset = load_settings()["preset"]
    s = Session(preset)
    async with BleakClient(device, timeout=20.0, **bt_debug.bluez_args(hci)) as client:
        for m in meters.values():
            m.reset()
        debug.update(connected_at=time.time(), preset=preset, layout={"eeg": s.eeg_n, "optics": s.opt_n}, signals={})
        session.start(getattr(device, "name", None), device.address, s.streams())
        debug["recording"] = session.dir
        logging.info(f"Connected. Preset {preset} ({mp.PRESETS.get(preset, '?')}); recording to {session.dir}")
        try:
            await client.start_notify(mp.CONTROL_UUID, s.on_control)
        except Exception as e:
            logging.warning(f"control notifications: {e}")
        for u in mp.DATA_UUIDS:
            await client.start_notify(u, s.on_data)
        await start_streams(client, preset)
        set_status("streaming", device.address)
        last_device.remember(device.address, getattr(device, "name", None))
        try:
            while time.time() - s.last_data < WATCHDOG_S:
                await asyncio.sleep(1)
                s.every_second()
                set_status("streaming", device.address)
                cmd = read_command()
                if not cmd:
                    continue
                if cmd["action"] == "disconnect":
                    auto_held = True
                    logging.info("Disconnect from the page: pick a headband to connect again.")
                    return
                if cmd["action"] == "connect" and cmd.get("address") and cmd["address"] != device.address:
                    next_target = cmd["address"]
                    return
                if cmd["action"] == "preset" and load_settings()["preset"] != preset:
                    next_target = device.address              # same headband, new preset: a new session
                    logging.info(f"Preset changed to {load_settings()['preset']}: restarting the streams")
                    return
                if cmd["action"] == "zero" and s.fnirs:
                    s.fnirs.rebaseline()
                    logging.info("fNIRS: new baseline (the next 10 s)")
            logging.warning("No data for 10 s: back to scanning.")
        finally:
            try:
                await send(client, "h")
            except Exception:
                pass


async def main():
    global selected, next_target
    logging.info("Muse worker online.")
    _write_json(DEVICES_FILE, [])
    set_status("scanning")
    if os.path.exists(COMMAND_FILE):
        os.remove(COMMAND_FILE)
    hci = None
    while True:
        if not selected:
            hci = await scan()
            continue
        mac, device = selected, ble_cache.get(selected)
        if not device:
            try:
                device = await BleakScanner.find_device_by_address(mac, timeout=6.0, **bt_debug.bluez_args(hci))
            except Exception as e:
                logging.error(f"Lookup error: {e}")
        if device:
            debug["adapter"] = bt_debug.adapter_of(device)
            debug["device"] = {"name": getattr(device, "name", None), "address": device.address, "rssi": last_rssi.get(device.address)}
            for attempt in range(1, CONNECT_ATTEMPTS + 1):
                set_status("connecting", mac, attempt)
                logging.info(f"Connecting to {mac} (attempt {attempt}/{CONNECT_ATTEMPTS})...")
                try:
                    try:
                        await stream(device, hci)
                    finally:
                        session.close()
                        debug["recording"] = None
                    break
                except Exception as e:
                    logging.warning(f"Link error on attempt {attempt}: {e!r}")
                    await asyncio.sleep(2)
        else:
            logging.warning("Headband not found; back to scanning.")
        selected, next_target = next_target, None
        debug["connected_at"] = None


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    finally:
        session.close()
