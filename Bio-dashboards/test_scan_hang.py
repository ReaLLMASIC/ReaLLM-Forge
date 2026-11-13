#!/usr/bin/env python3
"""bt_debug.scanning against a Bluetooth stack that never answers: each hung start gives up with ScanHung
after the timeout, a scan that works resets the count, and the third hang in a row exits the worker (code 75)
so server.py starts it again. No Bluetooth needed (a stand-in BleakScanner).

    python3 test_scan_hang.py
"""
import asyncio
import os
import subprocess
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))

CHILD = r"""
import asyncio, sys, types
sys.path.insert(0, sys.argv[1])
mode = {"hang": True}

class FakeScanner:
    def __init__(self, cb, **kw): pass
    async def __aenter__(self):
        if mode["hang"]:
            await asyncio.sleep(3600)
        return self
    async def __aexit__(self, *a): pass

sys.modules["bleak"] = types.SimpleNamespace(BleakScanner=FakeScanner)
import bt_debug
bt_debug.SCAN_TIMEOUT_S = 0.2

async def scan():
    async with bt_debug.scanning(lambda d, a: None, "hci9", timeout=0.2):
        pass

async def main():
    for plan in sys.argv[2]:                    # h = hang, o = ok
        mode["hang"] = plan == "h"
        try:
            await scan()
            print("ok", flush=True)
        except bt_debug.ScanHung as e:
            print("hung:", e, flush=True)

asyncio.run(main())
print("finished", flush=True)
"""

ok = 0


def check(name, cond, detail=""):
    global ok
    if not cond:
        raise SystemExit(f"FAIL {name} {detail}")
    ok += 1


def run(plan):
    p = subprocess.run([sys.executable, "-c", CHILD, HERE, plan], capture_output=True, text=True, timeout=30)
    return p.returncode, p.stdout.splitlines()


code, out = run("hhoh")
check("a hung start gives up instead of waiting forever", out[0].startswith("hung:") and "hci9" in out[0], out)
check("a scan that works resets the count", out[2] == "ok" and out[3].startswith("hung:") and code == 0 and out[-1] == "finished", out)
code, out = run("hhho")
check("three hangs in a row: the worker exits so it's started again", code == 75 and "restarting this worker" in out[-1]
      and "finished" not in out, (code, out))
print(f"{ok} checks passed")
