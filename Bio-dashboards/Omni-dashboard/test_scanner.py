# Usage: python3 test_scanner.py [band-address]   (address optional; bands are matched by name / service too)
import asyncio
import sys
from bleak import BleakScanner

# Known Viatom/Wellue indicators
VIATOM_MAC = (sys.argv[1] if len(sys.argv) > 1 else "").upper()
VIATOM_SVC_UUID = "14839ac4-7d7e-415c-9a42-167340cf2339".lower()
VIATOM_KEYWORDS = ["O2", "CHECKME", "VIATOM", "RING", "PC-60", "BABY", "SLEEP", "WEAR", "FS20", "POD", "BODI", "OXYGEN"]

async def run_deep_scan():
    print("🚀 Starting 10-second deep scan on hci1...")
    print("Keep your rings awake! Put them on your fingers so the screens light up.")
    print("-" * 60)
    
    found_devices = []

    def detection_callback(device, adv_data):
        name = (device.name or adv_data.local_name or "UNKNOWN").upper()
        addr = device.address.upper()
        uuids = [u.lower() for u in (adv_data.service_uuids or [])]
        rssi = adv_data.rssi or -100
        
        # --- DIAGNOSTIC MODE ---
        # If you still see nothing, UNCOMMENT the line below to see literally everything the antenna picks up
        # print(f"[RAW DUMP] MAC: {addr} | RSSI: {rssi} | Name: {name} | UUIDs: {uuids}")

        is_match = False
        if (VIATOM_MAC and addr == VIATOM_MAC):
            is_match = True
        elif VIATOM_SVC_UUID in uuids:
            is_match = True
        elif any(k in name for k in VIATOM_KEYWORDS):
            is_match = True
            
        if is_match and addr not in [d['address'] for d in found_devices]:
            found_devices.append({"name": name, "address": addr, "rssi": rssi, "uuids": uuids})
            print(f"✅ TARGET ACQUIRED: {name} ({addr}) | RSSI: {rssi}")

    try:
        # Forcing the scan through hci1 (your external USB dongle)
        async with BleakScanner(detection_callback, passive=False, bluez={"adapter": "hci1"}):
            await asyncio.sleep(10.0)
    except Exception as e:
        print(f"\n❌ CRITICAL RADAR FAILURE: {e}")
        print("Is hci1 plugged in and awake? Try running: sudo hciconfig hci1 up")
        return

    print("-" * 60)
    print(f"🏁 Scan complete. Found {len(found_devices)} Viatom-compatible devices.")
    for d in found_devices:
        print(f"   -> {d['name']} [{d['address']}]")

if __name__ == "__main__":
    try:
        asyncio.run(run_deep_scan())
    except KeyboardInterrupt:
        print("\nScan aborted by user.")
