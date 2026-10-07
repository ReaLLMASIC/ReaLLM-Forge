# dump_gatt.py
# Usage: python3 dump_gatt.py <device-address>
import asyncio
import sys
from bleak import BleakClient

if len(sys.argv) != 2:
    sys.exit("usage: python3 dump_gatt.py <device-address>   (find it with: bluetoothctl devices)")
TARGET_MAC = sys.argv[1]

async def run():
    print(f"Connecting to {TARGET_MAC} to map services...")
    async with BleakClient(TARGET_MAC, timeout=15.0) as client:
        if client.is_connected:
            print("\n=== GATT PROFILE MAP DISCOVERED ===")
            for service in client.services:
                print(f"\n[Service] {service.uuid} ({service.description})")
                for char in service.characteristics:
                    print(f"  └── [Characteristic] {char.uuid} | Properties: {char.properties}")
            print("\n===================================")
        else:
            print("Failed to connect.")

if __name__ == "__main__":
    asyncio.run(run())
