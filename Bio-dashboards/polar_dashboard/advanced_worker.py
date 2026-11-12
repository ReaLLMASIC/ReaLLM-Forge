import asyncio
import csv
import datetime
import json
import logging
import os
import sys
import time

from importlib.metadata import version as _pkg_version

from bleak import BleakScanner
from polar_python import PolarDevice

# The strap answers each "start a stream" request on its control point. A refusal is a short,
# 4-byte reply, which polar-python 1.x can't parse (IndexError: bytearray index out of range),
# so the real reason was lost. Read the error code ourselves and give the parser a padded copy.
PMD_ERRORS = {1: "invalid op code", 2: "invalid measurement type", 3: "not supported", 4: "invalid length",
              5: "invalid parameter", 6: "already streaming", 7: "invalid resolution", 8: "invalid sample rate",
              9: "invalid range", 10: "invalid MTU", 11: "invalid number of channels",
              12: "busy (another connection is streaming)", 13: "in its charger"}
last_pmd_error = None
try:
    from polar_python.models import measurement_settings as _ms
    _orig_from_bytes = _ms.MeasurementSettings.from_bytes.__func__

    def _from_bytes(cls, data):
        global last_pmd_error
        last_pmd_error = data[3] if len(data) > 3 else None
        if len(data) < 5:
            data = bytearray(data) + bytearray(5 - len(data))
        return _orig_from_bytes(cls, data)

    _ms.MeasurementSettings.from_bytes = classmethod(_from_bytes)
except Exception as _e:          # a different polar-python: keep its own behaviour
    print(f"note: couldn't hook polar-python's control-point parser ({_e!r})")


class StreamRefused(Exception):
    def __init__(self, name, code):
        super().__init__(f"{name}: {PMD_ERRORS.get(code, code)}")
        self.name, self.code = name, code


async def start_pmd(name, start_fn, tries=3):
    """Run one polar.start_*_stream() and say plainly if the strap refused it. "Invalid MTU" is
    retried: the link's packet size is agreed just after connecting and can still be settling."""
    global last_pmd_error
    for attempt in range(1, tries + 1):
        last_pmd_error = None
        await start_fn()
        code = last_pmd_error
        if code in (None, 0, 6):        # 6: already streaming on this connection -> fine
            return
        if code == 10 and attempt < tries:
            logging.info(f"The strap says the link's packet size (MTU) is too small for {name}; trying again...")
            await asyncio.sleep(1.5)
            continue
        raise StreamRefused(name, code)


def bluez_mtu_cap(path="/etc/bluetooth/main.conf"):
    """The ExchangeMTU set under [GATT] in BlueZ's config, if any (None when it's the default)."""
    import re
    try:
        section = None
        for line in open(path):
            line = line.strip()
            if line.startswith("["):
                section = line
            elif section == "[GATT]" and re.match(r"ExchangeMTU\s*=", line):
                return int(line.split("=", 1)[1].strip())
    except (OSError, ValueError):
        pass
    return None


def refusal_advice(refused, mtu):
    """What to tell you, by the strap's reason."""
    names = " and ".join(n for n, _ in refused)
    codes = {c for _, c in refused}
    head = f"The strap refused the {names} stream{'s' if len(refused) > 1 else ''}: " + \
           ", ".join(sorted({PMD_ERRORS.get(c, f'error {c}') for c in codes})) + ". Recording heart rate and R-R"
    head += " only." if len(refused) > 1 else " and the rest."
    if 10 in codes:
        cap = bluez_mtu_cap()
        if cap:
            return (head + f" /etc/bluetooth/main.conf caps the Bluetooth packet size (ExchangeMTU = {cap}, the old-dongle "
                    "workaround from the bottle's TROUBLESHOOTING section 12), and the H10 needs more for ECG and motion. "
                    "Comment that line out, run sudo systemctl restart bluetooth, then Device → Restart.")
        return (head + f" The Bluetooth link's packet size (MTU) is {mtu or 'unknown'} bytes; the H10 needs a larger one "
                "for ECG and motion. BlueZ normally agrees a bigger size when it connects: check that "
                "/etc/bluetooth/main.conf doesn't set ExchangeMTU under [GATT], try the other radio "
                "(Device → Bluetooth radio), or restart Bluetooth (sudo systemctl restart bluetooth).")
    if codes & {12}:
        return (head + " Usually something else is already streaming from the strap (it takes two connections): an "
                "older Bio-dash still running, the Polar Beat app, or another computer. Stop it, then Device → Restart.")
    if codes & {13}:
        return head + " The pod is in its charger."
    return head + " Unclip the pod for 10 seconds to reset the strap, then Device → Restart."

import os as _os, sys as _sys
_sys.path.insert(1, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # the shared bt_debug.py, in the release folder
import bt_debug
from stream_clock import ClockAnchor, StreamClock

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

BASE_DIR = bt_debug.state_dir(os.path.dirname(os.path.abspath(__file__)))   # one more per extra person: see bt_debug.state_dir
LOGS_DIR = os.path.join(BASE_DIR, "logs_advanced")
os.makedirs(LOGS_DIR, exist_ok=True)

DEVICES_FILE = os.path.join(LOGS_DIR, "devices.json")
COMMAND_FILE = os.path.join(LOGS_DIR, "command.json")
STATUS_FILE = os.path.join(LOGS_DIR, "status.json")
# Which Bluetooth radio to use (set from the debug panel). Stored by adapter address.
# auto prefers an external radio: the H10 can monopolise the radio it streams on.
radio_pin = bt_debug.RadioPin(os.path.join(BASE_DIR, "radio_config.json"), prefer="external")
last_device = bt_debug.LastDevice(os.path.join(BASE_DIR, "last_device.json"))
radio_hci = None
CONNECT_ATTEMPTS = 3  # BlueZ often aborts the first connect to an H10; retry before rescanning

ECG_HZ = 130
ACC_HZ = 200
WATCHDOG_S = 10.0

# Recordings: <Documents>/Bio-dash/Polar H10/<device>/<date>/<time>_{ecg,acc,rr}.csv
# (see bt_debug.SessionFiles). Files open when a strap connects, one set per session.
SESSION_STREAMS = {
    "ecg": ["Timestamp_Epoch_ms", "ECG_mV", "HR_BPM"],
    "acc": ["Timestamp_Epoch_ms", "X_mg", "Y_mg", "Z_mg"],
    "rr":  ["Timestamp_Epoch_ms", "RR_ms", "HR_BPM"],
}
session = bt_debug.SessionFiles("Polar H10")

current_hr = 0
ecg_counter = 0
selected_target_mac = None
last_heartbeat = 0.0

# BLEDevice objects from the last scan, so connecting doesn't need a second DBus lookup
ble_device_cache = {}


h10_clock = ClockAnchor()   # ECG and accelerometer share the strap's clock
ecg_clock = StreamClock(ECG_HZ, h10_clock)
acc_clock = StreamClock(ACC_HZ, h10_clock)

# --- Debug telemetry (shown in the dashboard's debug panel) ---
meters = {k: bt_debug.RateMeter() for k in ("ecg", "acc", "ecg_frames", "acc_frames", "hr", "rr")}
last_rssi = {}
debug = {
    "adapters": [],
    "adapter": None,          # radio the Polar was found on / connected through
    "device": None,           # {"name", "address", "rssi"}
    "connected_at": None,
    "nominal": {"ecg_hz": ECG_HZ, "acc_hz": ACC_HZ},
    "rates": {},
    "versions": {},
    "radio": {"config": "auto", "hci": None, "reason": None},
}
for _pkg in ("bleak", "polar-python", "dbus-fast"):
    try:
        debug["versions"][_pkg] = _pkg_version(_pkg)
    except Exception:
        debug["versions"][_pkg] = None
debug["versions"]["python"] = sys.version.split()[0]


def refresh_rates():
    debug["rates"] = {
        "ecg_hz": round(meters["ecg"].rate(), 1),
        "acc_hz": round(meters["acc"].rate(), 1),
        "ecg_frames_per_s": round(meters["ecg_frames"].rate(), 2),
        "acc_frames_per_s": round(meters["acc_frames"].rate(), 2),
        "ecg_samples_per_frame": ecg_clock.last_frame_len,
        "acc_samples_per_frame": acc_clock.last_frame_len,
        "hr_notifications_per_s": round(meters["hr"].rate(), 2),
        "rr_per_min": round(meters["rr"].rate() * 60, 1),
        "ecg_dropped": ecg_clock.dropped,
        "acc_dropped": acc_clock.dropped,
        "acc_hz_measured": round(acc_clock.measured_hz, 2),
        "clock_reanchors": h10_clock.reanchors,
    }


def reset_meters():
    for m in meters.values():
        m.reset()
    debug["rates"] = {}


def _first_attr(obj, names, default=None):
    for n in names:
        v = getattr(obj, n, None)
        if v:
            return v
    return default


def hr_callback(data):
    # polar_python HRData: heartrate (int BPM), rr_intervals (list[float] ms)
    global current_hr, last_heartbeat
    last_heartbeat = time.time()
    meters["hr"].add()

    bpm = _first_attr(data, ("heartrate", "bpm", "heart_rate"))
    if bpm is not None:
        current_hr = int(bpm)

    rr_list = _first_attr(data, ("rr_intervals", "rrs", "rrs_ms"), [])
    if rr_list:
        now_ms = int(time.time() * 1000)
        rows = []
        for rr in rr_list:
            try:
                rr_val = int(round(float(rr)))
            except (ValueError, TypeError):
                continue
            if 200 < rr_val < 2000:
                rows.append([now_ms, rr_val, current_hr])
        if rows:
            meters["rr"].add(len(rows))
            session.writerows("rr", rows)
            session.flush("rr")


def _acc_xyz(val):
    if isinstance(val, (list, tuple)):
        return val[0], val[1], val[2]
    if isinstance(val, dict):
        return val.get("x", val.get("X", 0)), val.get("y", val.get("Y", 0)), val.get("z", val.get("Z", 0))
    return getattr(val, "x", 0), getattr(val, "y", 0), getattr(val, "z", 0)


def acc_callback(data):
    # polar_python ACCData: timestamp (ns, last sample), data (list[(x, y, z)] mG)
    samples = _first_attr(data, ("data", "samples", "acc"), [])
    if not samples:
        return
    stamps = acc_clock.stamps(getattr(data, "timestamp", None), len(samples))
    rows = []
    for ts, val in zip(stamps, samples):
        try:
            x, y, z = _acc_xyz(val)
            rows.append([ts, int(x), int(y), int(z)])
        except (ValueError, TypeError, IndexError):
            continue
    session.writerows("acc", rows)
    session.flush("acc")
    meters["acc"].add(len(rows))
    meters["acc_frames"].add()


def ecg_callback(data):
    # polar_python ECGData: timestamp (ns, last sample), data (list[int] µV)
    global ecg_counter
    samples = _first_attr(data, ("data", "samples", "ecg", "voltages"), [])
    if not samples:
        return
    stamps = ecg_clock.stamps(getattr(data, "timestamp", None), len(samples))
    hr = current_hr
    rows = []
    for ts, val in zip(stamps, samples):
        try:
            uv = int(getattr(val, "voltage", getattr(val, "ecg_uv", val)))
        except (ValueError, TypeError):
            continue
        rows.append([ts, round(uv / 1000.0, 3), hr])
    session.writerows("ecg", rows)
    session.flush("ecg")
    ecg_counter += len(rows)
    meters["ecg"].add(len(rows))
    meters["ecg_frames"].add()

    if rows:
        sys.stdout.write(f"\r[ ❤️ HR: {hr:3d} BPM ] [ ECG: {ecg_counter:6d} ] || ⚡ ECG: {rows[-1][1]:>6.3f} mV    ")
        sys.stdout.flush()


def _write_json(path, obj):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f)
    os.replace(path + ".tmp", path)


def _write_devices(targets):
    _write_json(DEVICES_FILE, targets)


def set_status(state, address=None, attempt=None):
    """Tells the UI what the worker is doing: scanning | connecting | streaming."""
    _write_json(STATUS_FILE, {"state": state, "address": address, "attempt": attempt,
                              "last_device": last_device.load(), "auto_held": auto_held,
                              "max_attempts": CONNECT_ATTEMPTS, "debug": debug})


def _read_command():
    """The UI's command (consumed): {"action": "connect", "address": ...} or {"action": "disconnect"}, else None."""
    if not os.path.exists(COMMAND_FILE):
        return None
    try:
        with open(COMMAND_FILE, "r") as f:
            cmd = json.load(f)
        os.remove(COMMAND_FILE)
        if cmd.get("action") in ("connect", "disconnect"):
            return cmd
    except Exception:
        pass
    return None


# After "Disconnect" in the UI, auto-reconnect waits until a device is picked (in this run of the worker)
auto_held = False
next_target_mac = None        # a device picked while streaming another: connect to it next


async def scan_and_list_devices():
    global selected_target_mac, radio_hci

    logging.info("Radar active. Sweeping for Polar units...")
    debug["adapters"] = await bt_debug.list_adapters()
    radio_hci, reason = radio_pin.resolve(debug["adapters"])
    debug["radio"] = {"config": radio_pin.config(), "hci": radio_hci, "reason": reason}
    set_status("scanning")
    found_devices = {}
    cut_short = False

    def detection_callback(device, adv_data):
        name = device.name or adv_data.local_name or ""
        if "Polar" in name:
            ble_device_cache[device.address] = device
            last_rssi[device.address] = adv_data.rssi
            found_devices[device.address] = {
                "name": name,
                "address": device.address,
                "rssi": adv_data.rssi if adv_data.rssi else -100,
            }

    try:
        async with BleakScanner(detection_callback, **bt_debug.bluez_args(radio_hci)):
            # Sweep for up to 3 s, but bail early the moment the UI picks a device
            for _ in range(15):
                await asyncio.sleep(0.2)
                if os.path.exists(COMMAND_FILE):
                    cut_short = True
                    break
    except Exception as e:
        logging.error(f"Radar error: {e}")

    # A sweep cut short by a Connect click is partial -- keep the last full list
    # instead of blanking the device cards the user is looking at.
    if not cut_short:
        _write_devices(sorted(found_devices.values(), key=lambda x: x["rssi"], reverse=True))

    global auto_held
    cmd = _read_command()
    mac = cmd.get("address") if cmd and cmd["action"] == "connect" else None
    if mac:
        selected_target_mac = mac
        auto_held = False
        logging.info(f"UI Command received! Target locked: {selected_target_mac}")
    elif not auto_held:
        # Auto-reconnect: the last device we streamed from is back in range
        last = last_device.target()
        if last and last["address"] in found_devices:
            selected_target_mac = last["address"]
            logging.info(f"Auto-reconnecting to last device {last.get('name') or ''} [{selected_target_mac}]")


async def stream_session(device):
    """One connected session. Returns when the watchdog fires; raises on link errors."""
    global last_heartbeat
    async with PolarDevice(device) as polar:
        ecg_clock.reset()
        acc_clock.reset()
        reset_meters()
        debug["connected_at"] = time.time()
        session.start(getattr(device, "name", None), device.address, SESSION_STREAMS)
        debug["recording"] = session.dir
        debug["files"] = dict(session.paths)          # the API follows these, not "the newest recording"
        logging.info(f"Recording to {session.dir}")

        logging.info("Connected! Starting standard Heart Rate + HRV engine...")
        await polar.start_hr_stream(hr_callback)
        await asyncio.sleep(0.5)

        debug["warning"] = None
        # the link's packet size (MTU): the H10 needs a large one for ECG and motion
        mtu = None
        try:
            backend = getattr(polar._client, "_backend", None)
            if backend is not None and hasattr(backend, "_acquire_mtu"):
                await backend._acquire_mtu()
            mtu = polar._client.mtu_size
            debug.setdefault("gatt", {})["mtu"] = mtu
            logging.info(f"Link packet size (MTU): {mtu} bytes")
        except Exception as e:
            logging.info(f"Couldn't read the link's packet size ({e!r})")

        # each stream on its own: one refused doesn't stop the other (or heart rate)
        refused = []
        logging.info(f"Initializing {ACC_HZ}Hz Accelerometer stream...")
        try:
            await start_pmd("motion", lambda: polar.start_acc_stream(acc_callback, ACC_HZ, 16, 8))
        except StreamRefused as e:
            refused.append((e.name, e.code))
        await asyncio.sleep(0.5)
        logging.info(f"Initializing {ECG_HZ}Hz clean ECG stream...")
        try:
            await start_pmd("ECG", lambda: polar.start_ecg_stream(ecg_callback, ECG_HZ, 14))
        except StreamRefused as e:
            refused.append((e.name, e.code))
        if refused:
            debug["warning"] = refusal_advice(refused, mtu)
            logging.warning(debug["warning"])

        set_status("streaming", device.address)
        last_device.remember(device.address, getattr(device, "name", None))
        logging.info(f"!!! TELEMETRY ACTIVE: recording to {session.dir} !!!\n")

        last_heartbeat = time.time()
        while time.time() - last_heartbeat < WATCHDOG_S:
            await asyncio.sleep(1)
            refresh_rates()
            set_status("streaming", device.address)  # 1 Hz debug refresh
            cmd = _read_command()
            if cmd and cmd["action"] == "disconnect":
                global auto_held
                auto_held = True
                logging.info("Disconnect from the UI: closing the session; pick a device to connect again.")
                return
            if cmd and cmd["action"] == "connect" and cmd.get("address") and cmd["address"] != device.address:
                global next_target_mac
                next_target_mac = cmd["address"]
                logging.info(f"Switching to {next_target_mac} (picked in the UI)")
                return

        logging.warning("\nConnection Dropped (Watchdog Timeout). Returning to Scanner...")


async def main():
    global selected_target_mac
    logging.info("Hardware Engine Online.")

    # Don't let a previous session's device list or stale command leak into this one
    _write_devices([])
    set_status("scanning")
    if os.path.exists(COMMAND_FILE):
        try:
            os.remove(COMMAND_FILE)
        except Exception:
            pass

    while True:
        if not selected_target_mac:
            await scan_and_list_devices()
            continue

        mac = selected_target_mac
        debug["connected_at"] = None
        device = ble_device_cache.get(mac)
        if not device:
            try:
                device = await BleakScanner.find_device_by_address(mac, timeout=5.0, **bt_debug.bluez_args(radio_hci))
            except Exception as e:
                logging.error(f"Lookup error: {e}")

        if device:
            debug["adapter"] = bt_debug.adapter_of(device)
            debug["device"] = {"name": getattr(device, "name", None), "address": device.address,
                               "rssi": last_rssi.get(device.address)}
            if debug["adapter"]:
                logging.info(f"Using Bluetooth radio {debug['adapter']}")
        set_status("connecting", mac, 1)

        if not device:
            logging.warning("\n⚠️ Device not found. Returning to scanner...")
        else:
            for attempt in range(1, CONNECT_ATTEMPTS + 1):
                set_status("connecting", mac, attempt)
                logging.info(f"Connecting to {mac} (attempt {attempt}/{CONNECT_ATTEMPTS})...")
                try:
                    try:
                        await stream_session(device)
                    finally:
                        session.close()          # a session's files end with its connection
                        debug["recording"] = None
                        debug["files"] = None
                    break  # session ran and then went quiet -> rescan
                except Exception as e:
                    logging.warning(f"\n⚠️ Link error on attempt {attempt}/{CONNECT_ATTEMPTS}: {e!r}")
                    await asyncio.sleep(2)

        global next_target_mac
        selected_target_mac, next_target_mac = next_target_mac, None      # a device picked meanwhile, else rescan
        debug["connected_at"] = None
        reset_meters()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("\nHalting streaming contexts. Closing open file descriptors...")
    finally:
        session.close()
        logging.info("Logs closed cleanly. Safe to exit.")
