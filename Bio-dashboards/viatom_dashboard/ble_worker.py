# ble_worker.py
import asyncio
import time
import csv
import datetime
import json
import logging
import os
import sys
from bleak import BleakClient, BleakScanner
from bleak.backends.scanner import AdvertisementData
from bleak.backends.device import BLEDevice
from importlib.metadata import version as _pkg_version

import bt_debug

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)]
)

# Known hardcoded fallback match
# Optional: pin a specific band by address (it is found by name -- "Band-WU", "O2", ... -- anyway).
KNOWN_MAC = os.environ.get("BIODASH_VIATOM_MAC", "").strip()
HEALTH_SERVICE_UUID = "14839ac4-7d7e-415c-9a42-167340cf2339"
WRITE_CHAR_UUID = "8b00ace7-eb0a-49b0-b977-10a8d4d5e82f" 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data.json")
DEVICES_FILE = os.path.join(BASE_DIR, "devices.json")
COMMAND_FILE = os.path.join(BASE_DIR, "command.json")
DEBUG_FILE = os.path.join(BASE_DIR, "debug.json")
# Which Bluetooth radio to use (set from the debug panel), stored by adapter address.
# auto prefers an internal radio, leaving any external dongle to the Polar H10.
radio_pin = bt_debug.RadioPin(os.path.join(BASE_DIR, "radio_config.json"), prefer="internal")
last_device = bt_debug.LastDevice(os.path.join(BASE_DIR, "last_device.json"))
radio_hci = None


current_metrics = {"spo2": 0, "hr": 0, "battery": 100, "status": "Scanning"}
selected_target_mac = None
ble_device_cache = {}

# --- Debug telemetry (shown in the dashboard's debug panel) ---
POLL_INTERVAL_S = 2.0
# Readings only arrive every ~2 s, so rates use a longer window than the Polar's
meters = {k: bt_debug.RateMeter(window=20) for k in ("notifications", "readings", "polls")}
counters = {"fragments_skipped": 0, "calibrating_packets": 0, "out_of_range": 0,
            "write_failures": 0, "link_failures": 0}
last_rssi = {}
debug = {
    "state": "Scanning",
    "adapters": [],
    "adapter": None,
    "device": None,
    "connected_at": None,
    "last_reading_at": None,
    "gatt": {},
    "nominal": {"poll_interval_s": POLL_INTERVAL_S, "readings_per_s": 1 / POLL_INTERVAL_S},
    "rates": {},
    "counters": counters,
    "versions": {},
    "radio": {"config": "auto", "hci": None, "reason": None},
}
for _pkg in ("bleak", "dbus-fast", "flask"):
    try:
        debug["versions"][_pkg] = _pkg_version(_pkg)
    except Exception:
        debug["versions"][_pkg] = None
debug["versions"]["python"] = sys.version.split()[0]


def reset_session_stats():
    for m in meters.values():
        m.reset()
    for k in ("fragments_skipped", "calibrating_packets", "out_of_range", "write_failures"):
        counters[k] = 0
    debug["rates"] = {}
    debug["last_reading_at"] = None


def resolve_radio():
    """Updates radio_hci from the pin + current adapters; returns True if it changed."""
    global radio_hci
    hci, reason = radio_pin.resolve(debug["adapters"])
    debug["radio"] = {"config": radio_pin.config(), "hci": hci, "reason": reason}
    changed = hci != radio_hci
    radio_hci = hci
    return changed


async def debug_loop():
    """Writes debug.json once a second; refreshes the radio list every 10 s."""
    last_adapters = 0.0
    while True:
        if time.time() - last_adapters > 10:
            debug["adapters"] = await bt_debug.list_adapters()
            last_adapters = time.time()
        resolve_radio()          # every second, so a new pin applies quickly
        debug["state"] = current_metrics["status"]
        debug["rates"] = {
            "notifications_per_s": round(meters["notifications"].rate(), 2),
            "readings_per_s": round(meters["readings"].rate(), 2),
            "polls_per_s": round(meters["polls"].rate(), 2),
        }
        try:
            with open(DEBUG_FILE + ".tmp", "w") as f:
                json.dump(debug, f)
            os.replace(DEBUG_FILE + ".tmp", DEBUG_FILE)
        except Exception:
            pass
        await asyncio.sleep(1)

def save_to_json_dashboard():
    data = {
        "spo2": current_metrics["spo2"] if current_metrics["status"] == "Connected" else 0,
        "hr": current_metrics["hr"] if current_metrics["status"] == "Connected" else 0,
        "battery": current_metrics["battery"],
        "status": current_metrics["status"]
    }
    temp_file = DATA_FILE + ".tmp"
    try:
        with open(temp_file, "w") as f: json.dump(data, f)
        os.replace(temp_file, DATA_FILE)
    except Exception: pass

# Recordings: <Documents>/Bio-dash/Viatom O2/<device>/<date>/<time>_vitals.csv
# (bt_debug.SessionFiles), one file per connection, a row every 5 s while readings are valid.
session = bt_debug.SessionFiles("Viatom O2")
SESSION_STREAMS = {"vitals": ["Timestamp_Epoch_ms", "SpO2_pct", "HR_BPM", "Battery_pct"]}

def append_to_csv_log():
    if current_metrics["status"] != "Connected" or not session.active: return
    try:
        session.writerows("vitals", [[int(time.time() * 1000), current_metrics["spo2"],
                                      current_metrics["hr"], current_metrics["battery"]]])
        session.flush("vitals")
    except Exception: pass

def parse_checkme_notification(sender, data):
    if len(data) == 0: return
    meters["notifications"].add()
    if len(data) == 1 and data[0] == 0xA5:
        counters["calibrating_packets"] += 1
        if current_metrics["status"] != "Calibrating":
            current_metrics["status"] = "Calibrating"
            save_to_json_dashboard()
        return
    # Real-time frames start with 0x55; skip continuation fragments of longer frames
    if data[0] != 0x55:
        counters["fragments_skipped"] += 1
        return
    try:
        data_dict = {}
        if len(data) >= 8: data_dict['spo2'] = int(data[7])
        if len(data) >= 9: data_dict['bpm'] = int(data[8])
        if len(data) >= 15: data_dict['battery'] = int(data[14])

        if data_dict and not 40 <= data_dict.get('spo2', 0) <= 100:
            counters["out_of_range"] += 1
        if data_dict and 40 <= data_dict.get('spo2', 0) <= 100:
            meters["readings"].add()
            debug["last_reading_at"] = time.time()
            current_metrics["spo2"] = data_dict['spo2']
            current_metrics["hr"] = data_dict.get('bpm', current_metrics["hr"])
            current_metrics["battery"] = data_dict.get('battery', current_metrics["battery"])
            current_metrics["status"] = "Connected"
            print(f"🩸 LIVE BLE -> SpO2: {current_metrics['spo2']}% | HR: {current_metrics['hr']} BPM", flush=True)
            save_to_json_dashboard()
    except Exception: pass

async def log_timer_loop():
    while True:
        await asyncio.sleep(5)
        append_to_csv_log()

SEEN_TTL_S = 30  # keep a device listed this long after its last advertisement

async def scan_and_list_devices():
    """Scans CONTINUOUSLY until the UI picks a device.

    Wearables like the O2 Ultra advertise slowly or in bursts; short start/stop
    scan windows kept missing them. One long-lived scanner plus a last-seen TTL
    catches slow advertisers and keeps their card from flickering in and out.
    Returns once a connect command arrives.
    """
    global selected_target_mac
    current_metrics["status"] = "Scanning"
    save_to_json_dashboard()

    logging.info("Scanning for active Checkme wrist devices (continuous)...")
    seen = {}  # address -> (info, last_seen)

    def detection_callback(device: BLEDevice, adv_data: AdvertisementData):
        addr = device.address.upper()
        name = device.name or adv_data.local_name or ""
        uuids = [u.lower() for u in (adv_data.service_uuids or [])]

        # Identification: MAC matches, UUID matches, or text identifiers match
        is_match = (
            (KNOWN_MAC and addr == KNOWN_MAC.upper()) or
            HEALTH_SERVICE_UUID.lower() in uuids or
            any(x in name.upper() for x in ["O2", "CHECKME", "VIATOM", "BAND-WU"])
        )
        if not is_match:
            return

        ble_device_cache[device.address] = device
        last_rssi[device.address] = adv_data.rssi
        prev = seen.get(device.address)
        info = {
            "name": name or (prev[0]["name"] if prev else f"Checkme Wrist Unit ({addr[-5:]})"),
            "address": device.address,
            "rssi": adv_data.rssi if adv_data.rssi else -100,
        }
        if not prev:
            logging.info(f"Spotted {info['name']} [{device.address}] at {info['rssi']} dBm")
        seen[device.address] = (info, time.time())

    last_count = -1
    try:
        resolve_radio()
        scan_hci = radio_hci
        logging.info(f"Scanning on {scan_hci or 'the default radio'}")
        async with BleakScanner(detection_callback, **bt_debug.bluez_args(scan_hci)):
            while True:
                await asyncio.sleep(1.0)
                if radio_hci != scan_hci:
                    logging.info(f"Radio changed to {radio_hci or 'default'}: restarting the scan there")
                    return   # the main loop calls us again on the new radio

                now = time.time()
                for a in [a for a, (_, t) in seen.items() if now - t > SEEN_TTL_S]:
                    logging.info(f"Lost {seen[a][0]['name']} [{a}] (no adverts for {SEEN_TTL_S}s)")
                    del seen[a]

                targets = sorted((i for i, _ in seen.values()), key=lambda x: x["rssi"], reverse=True)
                with open(DEVICES_FILE + ".tmp", "w") as f:
                    json.dump(targets, f)
                os.replace(DEVICES_FILE + ".tmp", DEVICES_FILE)

                if len(targets) != last_count:
                    logging.info(f"{len(targets)} matching wrist monitor(s) in range.")
                    last_count = len(targets)

                if os.path.exists(COMMAND_FILE):
                    try:
                        with open(COMMAND_FILE, 'r') as f:
                            cmd = json.load(f)
                        os.remove(COMMAND_FILE)
                        if cmd.get("action") == "connect":
                            selected_target_mac = cmd.get("address")
                            logging.info(f"UI Selection registered! Target locked: {selected_target_mac}")
                            return  # leaving the context stops the scan before we connect
                    except Exception:
                        pass
                else:
                    # Auto-reconnect: the last band we streamed from is advertising again
                    last = last_device.target()
                    if last and last["address"] in seen:
                        selected_target_mac = last["address"]
                        logging.info(f"Auto-reconnecting to last device {last.get('name') or ''} [{selected_target_mac}]")
                        return
    except Exception as e:
        logging.error(f"Radar error: {e}")
        await asyncio.sleep(2)

async def run():
    global selected_target_mac
    log_task = asyncio.create_task(log_timer_loop())  # keep a reference so it isn't GC'd
    debug_task = asyncio.create_task(debug_loop())
    if os.path.exists(COMMAND_FILE):
        try: os.remove(COMMAND_FILE)
        except Exception: pass

    while True:
        if not selected_target_mac:
            await scan_and_list_devices()
            continue
            
        logging.info(f"Connecting to user selected device: {selected_target_mac}")
        current_metrics["status"] = "Connecting"
        save_to_json_dashboard()
        
        device = ble_device_cache.get(selected_target_mac, selected_target_mac)
        reset_session_stats()
        debug["gatt"] = {}
        debug["adapter"] = bt_debug.adapter_of(device) if not isinstance(device, str) else None
        debug["device"] = {"name": getattr(device, "name", None), "address": selected_target_mac,
                           "rssi": last_rssi.get(selected_target_mac)}
        if debug["adapter"]:
            logging.info(f"Using Bluetooth radio {debug['adapter']}")
        try:
            async with BleakClient(device, timeout=10.0, **bt_debug.bluez_args(radio_hci)) as client:
                debug["connected_at"] = time.time()
                # Use a slightly softer service discovery check once connected
                services = client.services.get_service(HEALTH_SERVICE_UUID)
                
                # If the service cache lookup fails on this Linux build, extract characteristics directly
                chars = services.characteristics if services else []
                target_notify_uuid = next((c.uuid for c in chars if "notify" in c.properties), None)
                target_write_uuid = next((c.uuid for c in chars if "write" in c.properties or "write-without-response" in c.properties), WRITE_CHAR_UUID)
                if not target_notify_uuid:
                    # (the old fallback subscribed to the SERVICE uuid, which can never work)
                    raise RuntimeError("Viatom service/notify characteristic not found -- GATT cache stale?")

                debug["gatt"] = {"notify_uuid": target_notify_uuid, "write_uuid": target_write_uuid,
                                 "mtu": getattr(client, "mtu_size", None), "write_mode": "with response"}
                await client.start_notify(target_notify_uuid, parse_checkme_notification)
                last_device.remember(selected_target_mac, (debug.get("device") or {}).get("name"))
                session.start((debug.get("device") or {}).get("name"), selected_target_mac, SESSION_STREAMS)
                debug["recording"] = session.dir
                logging.info(f"Recording to {session.dir}")
                write_bytes = bytearray([0xAA, 0x17, 0xE8, 0x00, 0x00, 0x00, 0x00, 0x1B])
                
                use_response = True
                misses = 0
                while client.is_connected:
                    try:
                        await client.write_gatt_char(target_write_uuid, write_bytes, response=use_response)
                        meters["polls"].add()
                        misses = 0
                    except Exception:
                        # one failed poll shouldn't kill the link; flip write mode and retry
                        counters["write_failures"] += 1
                        use_response = not use_response
                        debug["gatt"]["write_mode"] = "with response" if use_response else "without response"
                        misses += 1
                        if misses >= 3:
                            raise
                    await asyncio.sleep(POLL_INTERVAL_S)

                logging.warning("Connection dropped. Returning to radar scan.")
                selected_target_mac = None
                
        except Exception as e:
            logging.error(f"Link failed: {e}. Resetting target alignment.")
            counters["link_failures"] += 1
            selected_target_mac = None
            await asyncio.sleep(2)
        debug["connected_at"] = None
        session.close()                    # a session's file ends with its connection
        debug["recording"] = None

if __name__ == "__main__":
    try: asyncio.run(run())
    except KeyboardInterrupt: logging.info("Clean shutdown.")
