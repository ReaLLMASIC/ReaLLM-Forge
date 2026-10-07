import asyncio
import csv
import datetime
import os
import sys
import json
import time
import logging
from importlib.metadata import version as _pkg_version

from bleak import BleakScanner, BleakClient

import bt_debug
from atmos_parser import LineParser, PROFILES, is_canonical

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs_atmos")
os.makedirs(LOGS_DIR, exist_ok=True)

DEVICES_FILE = os.path.join(LOGS_DIR, "devices.json")
COMMAND_FILE = os.path.join(LOGS_DIR, "command.json")
STATUS_FILE = os.path.join(LOGS_DIR, "status.json")
PROFILE_FILE = os.path.join(BASE_DIR, "atmos_profile.json")   # set from the debug panel; kept across sessions
# Which Bluetooth radio to use (debug panel), stored by adapter address. auto prefers an
# internal radio, leaving any external dongle to the Polar H10.
radio_pin = bt_debug.RadioPin(os.path.join(BASE_DIR, "radio_config.json"), prefer="internal")
last_device = bt_debug.LastDevice(os.path.join(BASE_DIR, "last_device.json"))
TCP_RECONNECT_EVERY_S = 15      # how often to retry a remembered Wi-Fi device that's unreachable
_last_tcp_try = 0.0
radio_hci = None
TCP_CONNECT_TIMEOUT_S = 5.0
CONNECT_ATTEMPTS = 3   # BlueZ often aborts a first connect; retry before rescanning
WATCHDOG_S = 10.0

# Nordic UART Service UUIDs
UART_SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
UART_TX_CHAR_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"



class SchemaLog:
    """CSV whose header names every series (e.g. ts,co2@scd30,temp@sen55,...).

    Devices differ, so the columns come from the data: the file is opened on the first
    reading, and if a later reading brings a field the header doesn't have, a new part
    file starts with the wider header. Missing values are left blank.

    Files go to <Documents>/Bio-dash/Atmos/<device>/<date>/<time>_readings.csv (then
    <time>_part2_readings.csv, ...). start() is called when a device connects.
    """

    def __init__(self):
        self.file = self.writer = None
        self.columns = []
        self.part = 0
        self.path = None
        self.dir = None
        self.stamp = None

    def start(self, device_name, address):
        self.close()
        now = datetime.datetime.now()
        self.dir = os.path.join(bt_debug.dashboard_dir("Atmos"), bt_debug.device_folder(device_name, address),
                                now.strftime("%Y-%m-%d"))
        os.makedirs(self.dir, exist_ok=True)
        self.stamp = now.strftime("%H-%M-%S")
        debug["recording"] = self.dir
        logging.info(f"Recording to {self.dir}")

    def _open(self, columns):
        if self.file:
            self.file.close()
        self.part += 1
        suffix = "" if self.part == 1 else f"_part{self.part}"
        self.path = os.path.join(self.dir, f"{self.stamp}{suffix}_readings.csv")
        self.file = open(self.path, mode="a", newline="")
        self.writer = csv.writer(self.file)
        self.columns = columns
        self.writer.writerow(["ts"] + columns)
        self.file.flush()
        logging.info(f"Logging {len(columns)} series to {os.path.basename(self.path)}")

    def write_rows(self, rows):
        """rows: [(ts_ms, {series_id: value}), ...]"""
        if self.dir is None:            # no device session (shouldn't happen): nowhere to file it
            return
        for ts, vals in rows:
            new = [k for k in vals if k not in self.columns]
            if self.file is None or new:
                self._open(self.columns + sorted(new))
            self.writer.writerow([ts] + [("" if vals.get(c) is None else round(vals[c], 3)) for c in self.columns])
        if self.file:
            self.file.flush()

    def close(self):
        if self.file:
            self.file.close()
        self.file = self.writer = None
        self.columns, self.part, self.dir = [], 0, None
        if "debug" in globals():
            debug["recording"] = None


log = SchemaLog()
parser = LineParser()
_profile_mtime = None


def load_profile_config(force=False):
    """Applies atmos_profile.json (profile + optional custom column map) when it changes."""
    global _profile_mtime
    try:
        mtime = os.path.getmtime(PROFILE_FILE)
    except OSError:
        mtime = None
    if mtime == _profile_mtime and not force:
        return False
    _profile_mtime = mtime
    cfg = {}
    if mtime is not None:
        try:
            with open(PROFILE_FILE) as f:
                cfg = json.load(f)
        except Exception:
            cfg = {}
    parser.configure(cfg.get("profile", "auto"), cfg.get("columns"))
    logging.info(f"Atmos profile: {cfg.get('profile', 'auto')}"
                 + (f", custom columns: {cfg['columns']}" if cfg.get("columns") else ""))
    # Re-evaluate the mapping now, so the debug panel reflects the change immediately
    # instead of after the next line arrives.
    if parser.last_format == "csv" and parser.last_field_count:
        _, parser.column_source = parser._columns_for(parser.last_field_count)
    return True

data_buffer = ""
samples_written = 0
selected_target_mac = None
last_message_time = 0

MAX_BUFFER = 8192  # a line is ~60-200 bytes; anything this long means framing is lost

# --- Debug telemetry (shown in the dashboard's debug panel) ---
meters = {k: bt_debug.RateMeter(window=15) for k in ("lines", "notifications", "bytes")}
counters = {"rejected_lines": 0, "header_lines": 0, "buffer_overflows": 0, "link_failures": 0}
fields = {}              # series_id -> latest value (this session)
_adapters_at = 0.0
line_times = []          # arrival times of recent good lines, for interval/jitter
last_rssi = {}
ble_device_cache = {}
debug = {
    "adapters": [], "adapter": None, "device": None,
    "connected_at": None, "last_line_at": None,
    "gatt": {"service": UART_SERVICE_UUID, "tx_char": UART_TX_CHAR_UUID, "mtu": None},
    "rates": {}, "counters": counters, "samples_written": 0, "versions": {},
    "radio": {"config": "auto", "hci": None, "reason": None}, "transport": None,
    "parser": {}, "fields": {},
    "profiles": {k: {"label": v["label"], "sensors": v["sensors"], "columns": v["columns"],
                     "order_confirmed": v["order_confirmed"]} for k, v in PROFILES.items()},
}
for _pkg in ("bleak", "dbus-fast", "fastapi"):
    try:
        debug["versions"][_pkg] = _pkg_version(_pkg)
    except Exception:
        debug["versions"][_pkg] = None
debug["versions"]["python"] = sys.version.split()[0]


def _write_json(path, obj):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f)
    os.replace(path + ".tmp", path)


def refresh_rates():
    now = time.time()
    recent = [t for t in line_times if now - t < 30]
    gaps = [b - a for a, b in zip(recent, recent[1:])]
    mean = sum(gaps) / len(gaps) if gaps else None
    jitter = (sum((g - mean) ** 2 for g in gaps) / len(gaps)) ** 0.5 if gaps else None
    notif_rate = meters["notifications"].rate()
    byte_rate = meters["bytes"].rate()
    debug["rates"] = {
        "lines_per_s": round(meters["lines"].rate(), 2),
        "notifications_per_s": round(notif_rate, 2),
        "bytes_per_s": round(byte_rate, 1),
        "bytes_per_notification": round(byte_rate / notif_rate, 1) if notif_rate else None,
        "line_interval_s": round(mean, 2) if mean else None,
        "line_jitter_s": round(jitter, 3) if jitter is not None else None,
    }
    debug["samples_written"] = samples_written
    load_profile_config()
    debug["parser"] = {
        "profile": parser.profile,
        "effective_profile": parser.effective_profile,
        "format": parser.last_format,
        "field_count": parser.last_field_count,
        "column_source": parser.column_source,
        "device_header": parser.device_header,
        "custom_columns": parser.custom_columns,
        "log_file": os.path.basename(log.path) if log.path else None,
    }
    debug["fields"] = {k: {"value": v, "mapped": is_canonical(k)} for k, v in sorted(fields.items())}


def reset_session_stats():
    for m in meters.values():
        m.reset()
    counters["rejected_lines"] = counters["buffer_overflows"] = counters["header_lines"] = 0
    line_times.clear()
    fields.clear()
    parser.reset_session()
    debug["rates"] = {}
    debug["last_line_at"] = None


def set_status(state, address=None, attempt=None):
    """Tells the UI what the worker is doing: scanning | connecting | streaming."""
    _write_json(STATUS_FILE, {"state": state, "address": address, "attempt": attempt,
                              "last_device": last_device.load(),
                              "max_attempts": CONNECT_ATTEMPTS, "debug": debug})


def handle_rx(sender, data):
    """Buffers UART chunks and writes EVERY complete line to the CSV.

    A single BLE notification can carry the end of one line and all of the next,
    so we must drain the whole buffer, not just the first line.
    """
    global data_buffer, samples_written, last_message_time
    last_message_time = time.time()
    meters["notifications"].add()
    meters["bytes"].add(len(data))

    data_buffer += data.decode("utf-8", errors="replace")
    if "\n" not in data_buffer:
        if len(data_buffer) > MAX_BUFFER:
            counters["buffer_overflows"] += 1
            data_buffer = ""
        return

    *lines, data_buffer = data_buffer.split("\n")
    epoch_ms = int(time.time() * 1000)
    rows = []
    for line in lines:
        if not line.strip():
            continue
        vals, kind = parser.parse(line)
        if kind == "data":
            rows.append((epoch_ms, vals))
            fields.update(vals)
        elif kind == "header":
            counters["header_lines"] += 1
            logging.info(f"Device header: {parser.device_header}")
        else:
            counters["rejected_lines"] += 1

    if rows:
        log.write_rows(rows)
        samples_written += len(rows)
        meters["lines"].add(len(rows))
        now = time.time()
        line_times.extend([now] * len(rows))
        del line_times[:-120]
        debug["last_line_at"] = now
        last = rows[-1][1]
        co2 = next((v for k, v in last.items() if k.split("@")[0] == "co2"), None)
        sys.stdout.write(f"\r[ 🌍 Atmos ] rows: {samples_written:5d} | fields: {len(last):2d}"
                         + (f" | CO2: {co2:4.0f} ppm   " if co2 is not None else "   "))
        sys.stdout.flush()

NAME_HINTS = ("SuperMini", "SEN69", "Air", "Atmos", "Sphere", "XIAO")

def _read_command():
    if not os.path.exists(COMMAND_FILE):
        return None
    try:
        with open(COMMAND_FILE, "r") as f:
            cmd = json.load(f)
        os.remove(COMMAND_FILE)
        if cmd.get("action") == "connect":
            return cmd.get("address")
        if cmd.get("action") == "connect_tcp" and cmd.get("host"):
            return f"tcp://{cmd['host']}:{int(cmd.get('port') or 8080)}"
    except Exception:
        pass
    return None

async def scan_and_list_devices():
    """Scans for active environmental monitors and updates the UI state list."""
    global selected_target_mac

    logging.info("Radar active. Sweeping for air monitors...")
    global _adapters_at, radio_hci
    if time.time() - _adapters_at > 10:      # radio list changes rarely; refresh every 10 s
        debug["adapters"] = await bt_debug.list_adapters()
        _adapters_at = time.time()
    radio_hci, reason = radio_pin.resolve(debug["adapters"])
    debug["radio"] = {"config": radio_pin.config(), "hci": radio_hci, "reason": reason}
    set_status("scanning")
    found_devices = {}
    cut_short = False

    def detection_callback(device, adv_data):
        name = device.name or adv_data.local_name or ""
        uuids = [u.lower() for u in (adv_data.service_uuids or [])]
        # Name hints, or anything advertising Nordic UART -- so a board swap/rename still shows up
        if any(h in name for h in NAME_HINTS) or UART_SERVICE_UUID in uuids:
            ble_device_cache[device.address] = device
            last_rssi[device.address] = adv_data.rssi
            found_devices[device.address] = {
                "name": name or f"UART device ({device.address[-5:]})",
                "address": device.address,
                "rssi": adv_data.rssi if adv_data.rssi else -100,
            }

    try:
        async with BleakScanner(detection_callback, **bt_debug.bluez_args(radio_hci)):
            for _ in range(15):  # up to 3 s, exit early once the UI picks something
                await asyncio.sleep(0.2)
                if os.path.exists(COMMAND_FILE):
                    cut_short = True
                    break
    except Exception as e:
        logging.error(f"Radar error: {e}")

    # A sweep cut short by a Connect click is partial -- keep the last full list
    if not cut_short:
        targets = sorted(found_devices.values(), key=lambda x: x["rssi"], reverse=True)
        _write_json(DEVICES_FILE, targets)

    global _last_tcp_try
    mac = _read_command()
    if mac:
        selected_target_mac = mac
        logging.info(f"UI Command received! Target locked: {selected_target_mac}")
    else:
        # Auto-reconnect to the last device: a BLE device once it's advertising again,
        # a Wi-Fi device by retrying its address (backing off while it's unreachable).
        last = last_device.target()
        if last and last.get("transport") == "tcp":
            if time.time() - _last_tcp_try >= TCP_RECONNECT_EVERY_S:
                _last_tcp_try = time.time()
                selected_target_mac = last["address"]
                logging.info(f"Auto-reconnecting over Wi-Fi to {selected_target_mac}")
        elif last and last["address"] in found_devices:
            selected_target_mac = last["address"]
            logging.info(f"Auto-reconnecting to last device {last.get('name') or ''} [{selected_target_mac}]")

async def stream_session(device):
    """One connected session. Returns when data stops (watchdog); raises on link errors."""
    global last_message_time, data_buffer
    async with BleakClient(device, **bt_debug.bluez_args(radio_hci)) as client:
        data_buffer = ""
        reset_session_stats()
        log.start(getattr(device, "name", None), device.address)
        debug["connected_at"] = time.time()
        debug["gatt"]["mtu"] = getattr(client, "mtu_size", None)
        logging.info("Connected! Subscribing to Nordic UART TX characteristic...")
        await client.start_notify(UART_TX_CHAR_UUID, handle_rx)
        set_status("streaming", device.address)
        last_device.remember(device.address, getattr(device, "name", None), transport="ble")
        logging.info(f"!!! TELEMETRY ACTIVE: recording to {log.dir} !!!\n")

        last_message_time = time.time()
        while time.time() - last_message_time < WATCHDOG_S:
            await asyncio.sleep(1)
            refresh_rates()
            set_status("streaming", device.address)   # 1 Hz debug refresh
        logging.warning("\nConnection Dropped (Data flow stopped). Returning to Scanner...")


async def tcp_session(host, port):
    """Wi-Fi stream (e.g. the Atmos-Sphere S4's CSV on port 8080). Bytes go through
    the same handle_rx() path as Bluetooth, so parsing, logging and stats are shared.
    Returns when data stops (watchdog); raises if the connection fails."""
    global last_message_time, data_buffer
    reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), TCP_CONNECT_TIMEOUT_S)
    peer = f"{host}:{port}"
    try:
        data_buffer = ""
        reset_session_stats()
        log.start(f"Wi-Fi {peer}", f"tcp://{peer}")
        debug.update(transport="tcp", adapter=None, connected_at=time.time(),
                     device={"name": f"Wi-Fi {peer}", "address": peer, "rssi": None})
        debug["gatt"]["mtu"] = None
        set_status("streaming", peer)
        last_device.remember(f"tcp://{peer}", f"Wi-Fi {peer}", transport="tcp", host=host, port=port)
        logging.info(f"!!! TELEMETRY ACTIVE over Wi-Fi ({peer}): recording to {log.dir} !!!\n")
        last_message_time = time.time()
        next_status = 0.0
        while time.time() - last_message_time < WATCHDOG_S:
            try:
                chunk = await asyncio.wait_for(reader.read(1024), 1.0)
            except asyncio.TimeoutError:
                chunk = None
            if chunk == b"":
                logging.warning("\nWi-Fi stream closed by the device.")
                break
            if chunk:
                handle_rx(None, chunk)
            if time.time() >= next_status:
                refresh_rates()
                set_status("streaming", peer)          # 1 Hz debug refresh
                next_status = time.time() + 1
        else:
            logging.warning("\nWi-Fi stream went quiet (watchdog). Returning to scanner...")
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


async def main():
    global selected_target_mac
    logging.info("Hardware Engine Online.")
    load_profile_config(force=True)

    _write_json(DEVICES_FILE, [])
    set_status("scanning")
    if os.path.exists(COMMAND_FILE):
        try: os.remove(COMMAND_FILE)
        except Exception: pass

    while True:
        if not selected_target_mac:
            await scan_and_list_devices()
            continue

        mac = selected_target_mac
        debug["connected_at"] = None

        if mac.startswith("tcp://"):
            host, _, port = mac[len("tcp://"):].rpartition(":")
            for attempt in range(1, CONNECT_ATTEMPTS + 1):
                set_status("connecting", f"{host}:{port}", attempt)
                logging.info(f"Connecting to {host}:{port} over Wi-Fi (attempt {attempt}/{CONNECT_ATTEMPTS})...")
                try:
                    await tcp_session(host, int(port))
                    break
                except Exception as e:
                    counters["link_failures"] += 1
                    logging.warning(f"\n⚠️ Wi-Fi link error on attempt {attempt}/{CONNECT_ATTEMPTS}: {e!r}")
                    await asyncio.sleep(2)
            selected_target_mac = None
            debug.update(connected_at=None, transport=None)
            reset_session_stats()
            log.close()
            continue

        debug["transport"] = "ble"
        device = ble_device_cache.get(mac)
        if not device:
            try:
                # Reuse the BLEDevice from the scan; only look it up if we don't have one
                device = await BleakScanner.find_device_by_address(mac, timeout=5.0, **bt_debug.bluez_args(radio_hci))
            except Exception as e:
                logging.error(f"Lookup error: {e}")

        if not device:
            logging.warning("Target device vanished. Returning to radar scan.")
        else:
            debug["adapter"] = bt_debug.adapter_of(device)
            debug["device"] = {"name": device.name, "address": device.address, "rssi": last_rssi.get(device.address)}
            if debug["adapter"]:
                logging.info(f"Using Bluetooth radio {debug['adapter']}")
            for attempt in range(1, CONNECT_ATTEMPTS + 1):
                set_status("connecting", mac, attempt)
                logging.info(f"Connecting to {device.name} [{device.address}] (attempt {attempt}/{CONNECT_ATTEMPTS})...")
                try:
                    await stream_session(device)
                    break   # session ran, then data stopped -> rescan
                except Exception as e:
                    counters["link_failures"] += 1
                    logging.warning(f"\n⚠️ Link error on attempt {attempt}/{CONNECT_ATTEMPTS}: {e!r}")
                    await asyncio.sleep(2)

        selected_target_mac = None
        debug["connected_at"] = None
        reset_session_stats()
        log.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("\nHalting streaming contexts. Closing open file descriptors...")
        log.close()
        logging.info("Logs closed cleanly. Safe to exit.")
