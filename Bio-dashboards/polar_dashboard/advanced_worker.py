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

import bt_debug

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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


class StreamClock:
    """Maps sensor-side PMD timestamps onto wall-clock epoch ms.

    The H10 stamps every PMD frame with the time of its LAST sample, but its
    clock usually isn't set to real time. We lock a fixed offset between the
    sensor clock and wall clock on the first frame, so sample spacing comes from
    the sensor (no BLE delivery jitter) while absolute time stays wall-clock.
    Re-anchors if the two drift apart by more than a second.
    """

    def __init__(self, hz):
        self.period_ms = 1000.0 / hz
        self.offset_ms = None
        self.prev_last_ms = None
        self.dropped = 0          # samples missing between frames (from sensor timestamps)
        self.last_frame_len = 0

    def reset(self):
        self.offset_ms = None
        self.prev_last_ms = None
        self.dropped = 0

    def stamps(self, sensor_ts_ns, n):
        now_ms = time.time() * 1000.0
        self.last_frame_len = n
        if sensor_ts_ns:
            last_ms = sensor_ts_ns / 1e6
            if self.prev_last_ms is not None:
                # frames should abut exactly; anything beyond ~1.5 periods is lost samples
                gap = (last_ms - self.prev_last_ms) - n * self.period_ms
                if gap > 1.5 * self.period_ms:
                    self.dropped += round(gap / self.period_ms)
            self.prev_last_ms = last_ms
            if self.offset_ms is None or abs(last_ms + self.offset_ms - now_ms) > 1000.0:
                self.offset_ms = now_ms - last_ms
            end_ms = last_ms + self.offset_ms
        else:
            end_ms = now_ms  # library didn't give a timestamp; fall back to arrival time
        p = self.period_ms
        return [int(end_ms - (n - 1 - i) * p) for i in range(n)]


ecg_clock = StreamClock(ECG_HZ)
acc_clock = StreamClock(ACC_HZ)

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
                              "last_device": last_device.load(),
                              "max_attempts": CONNECT_ATTEMPTS, "debug": debug})


def _read_command():
    """Returns the MAC the UI asked for (and consumes the command), else None."""
    if not os.path.exists(COMMAND_FILE):
        return None
    try:
        with open(COMMAND_FILE, "r") as f:
            cmd = json.load(f)
        os.remove(COMMAND_FILE)
        if cmd.get("action") == "connect":
            return cmd.get("address")
    except Exception:
        pass
    return None


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

    mac = _read_command()
    if mac:
        selected_target_mac = mac
        logging.info(f"UI Command received! Target locked: {selected_target_mac}")
    else:
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
        logging.info(f"Recording to {session.dir}")

        logging.info("Connected! Starting standard Heart Rate + HRV engine...")
        await polar.start_hr_stream(hr_callback)
        await asyncio.sleep(0.5)

        logging.info(f"Initializing {ACC_HZ}Hz Accelerometer stream...")
        await polar.start_acc_stream(acc_callback, ACC_HZ, 16, 8)
        await asyncio.sleep(0.5)

        logging.info(f"Initializing {ECG_HZ}Hz clean ECG stream...")
        await polar.start_ecg_stream(ecg_callback, ECG_HZ, 14)

        set_status("streaming", device.address)
        last_device.remember(device.address, getattr(device, "name", None))
        logging.info(f"!!! TELEMETRY ACTIVE: recording to {session.dir} !!!\n")

        last_heartbeat = time.time()
        while time.time() - last_heartbeat < WATCHDOG_S:
            await asyncio.sleep(1)
            refresh_rates()
            set_status("streaming", device.address)  # 1 Hz debug refresh

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
                    break  # session ran and then went quiet -> rescan
                except Exception as e:
                    logging.warning(f"\n⚠️ Link error on attempt {attempt}/{CONNECT_ATTEMPTS}: {e!r}")
                    await asyncio.sleep(2)

        selected_target_mac = None
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
