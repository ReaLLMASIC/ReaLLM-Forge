#!/usr/bin/env python3
"""How server.py shares the Bluetooth radios out between several people (plan_radios), without any radios.

    python3 test_radios.py
"""
import os
import sys

sys.argv = [sys.argv[0]]
os.environ.setdefault("BIODASH_APP_CONFIG", os.devnull)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server import plan_radios  # noqa: E402

ok = 0


def check(name, cond, detail=""):
    global ok
    if not cond:
        raise SystemExit(f"FAIL {name} {detail}")
    ok += 1


DONGLE = {"name": "hci1", "address": "RADIO-DONGLE", "placement": "external"}
BUILTIN = {"name": "hci0", "address": "RADIO-BUILTIN", "placement": "internal"}
DONGLE2 = {"name": "hci2", "address": "RADIO-DONGLE-2", "placement": "external"}
streaming = lambda hci: {"state": "streaming", "radio": {"hci": hci}}
looking = {"state": "scanning", "radio": {"hci": None}}
two = [("heart", "heart"), ("heart-p2", "heart")]

# what happened on meap-mainframe: the first strap already streaming on the dongle when a second person is added
p = plan_radios(two, [BUILTIN, DONGLE], {"heart": streaming("hci1"), "heart-p2": looking})
check("a streaming strap keeps its radio", p["heart"] == DONGLE["address"], p)
check("the second strap gets the free radio, not the busy dongle", p["heart-p2"] == BUILTIN["address"], p)

# same, with the first strap streaming on the built-in radio
p = plan_radios(two, [BUILTIN, DONGLE], {"heart": streaming("hci0"), "heart-p2": looking})
check("...and the other way round", p == {"heart": BUILTIN["address"], "heart-p2": DONGLE["address"]}, p)

# nobody connected yet: dongles first
p = plan_radios(two, [BUILTIN, DONGLE], {})
check("straps take the dongle first, then the next free radio", p == {"heart": DONGLE["address"], "heart-p2": BUILTIN["address"]}, p)

# three straps, two radios: one radio has to take two
three = two + [("heart-p3", "heart")]
p = plan_radios(three, [BUILTIN, DONGLE], {"heart": streaming("hci1")})
check("more straps than radios: shared, at most two per radio", sorted(p.values()).count(DONGLE["address"]) <= 2
      and sorted(p.values()).count(BUILTIN["address"]) <= 2 and p["heart"] == DONGLE["address"], p)

# three radios: each strap its own
p = plan_radios(three, [BUILTIN, DONGLE, DONGLE2], {"heart-p2": streaming("hci0")})
check("three radios, three straps: one each", len(set(p.values())) == 3 and p["heart-p2"] == BUILTIN["address"], p)

# rings go where the straps aren't
both = two + [("oxygen", "oxygen"), ("oxygen-p2", "oxygen")]
p = plan_radios(both, [BUILTIN, DONGLE, DONGLE2], {"heart": streaming("hci1"), "heart-p2": streaming("hci2")})
check("rings use the radio no strap uses", p["oxygen"] == p["oxygen-p2"] == BUILTIN["address"], p)

# a connected ring stays put too
p = plan_radios(both, [BUILTIN, DONGLE], {"oxygen": streaming("hci1")})
check("a connected ring keeps its radio", p["oxygen"] == DONGLE["address"], p)

# a strap streaming on a radio that has since gone (unplugged) is planned like one still looking
p = plan_radios(two, [BUILTIN], {"heart": streaming("hci1")})
check("a radio that's gone is not planned", set(p.values()) == {BUILTIN["address"]}, p)

check("no radios: no plan", plan_radios(two, [], {}) == {})
print(f"{ok} checks passed")
