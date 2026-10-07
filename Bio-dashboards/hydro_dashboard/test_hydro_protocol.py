"""Tests for hydro_protocol.py.  Run:  python3 -m pytest hydro_dashboard   (or: python3 hydro_dashboard/test_hydro_protocol.py)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hydro_protocol as hp

NOW = 1790000000.0      # fixed clock
CAP = 621               # 21 oz bottle


def epoch_frame(pending, ts, ml, total=0):
    return bytes([pending, 0]) + total.to_bytes(2, "big") + int(ts).to_bytes(4, "big") + ml.to_bytes(2, "big") + bytes(10)


def percent_frame(pending, pct, secs_ago, total=0):
    return bytes([pending, pct]) + total.to_bytes(2, "big") + bytes([0]) + secs_ago.to_bytes(4, "big") + bytes(11)


def test_empty_and_pending():
    assert hp.parse_sip_frame(bytes(20), CAP, NOW)["kind"] == "empty"
    assert hp.parse_sip_frame(bytes([3, 0, 0]), CAP, NOW) == {"kind": "pending", "remaining": 3}
    assert hp.parse_sip_frame(b"", CAP, NOW)["kind"] == "unknown"


def test_epoch_layout():
    r = hp.parse_sip_frame(epoch_frame(2, NOW - 3600, 42, total=500), CAP, NOW)
    assert r["kind"] == "sip" and r["layout"] == "epoch"
    assert r["volume_ml"] == 42 and r["ts"] == NOW - 3600 and r["remaining"] == 2 and r["total_reported"] == 500


def test_percent_layout():
    r = hp.parse_sip_frame(percent_frame(1, 8, 90, total=250), CAP, NOW)
    assert r["kind"] == "sip" and r["layout"] == "percent"
    assert r["volume_ml"] == round(CAP * 0.08) and r["ts"] == NOW - 90 and r["pct"] == 8


def test_layouts_never_both_plausible():
    # an epoch frame read as "percent" gives a seconds-ago far beyond 10 years, and vice versa
    for ts in (NOW - 10, NOW - 86400 * 30, 1500000000):
        for ml in (5, 120, 600):
            assert not hp.parse_sip_frame(epoch_frame(1, ts, ml), CAP, NOW).get("ambiguous")
    for pct in (1, 50, 100):
        for ago in (0, 60, 86400 * 5):
            r = hp.parse_sip_frame(percent_frame(1, pct, ago), CAP, NOW)
            assert r["layout"] == "percent" and not r["ambiguous"]


def test_garbage_is_unknown():
    junk = bytes([4, 0xFF]) + bytes([0xFF] * 18)        # pct 255, ts in the future
    assert hp.parse_sip_frame(junk, CAP, NOW)["kind"] == "unknown"
    future = epoch_frame(1, NOW + 10 * 86400, 50)       # epoch beyond tomorrow: rejected
    assert hp.parse_sip_frame(future, CAP, NOW)["kind"] == "unknown"


def test_cap_weight_battery():
    assert hp.parse_cap(bytes.fromhex("81020000")) is True
    assert hp.parse_cap(bytes.fromhex("80020000")) is False
    assert hp.parse_cap(b"") is None
    assert hp.parse_weight(bytes.fromhex("8a7b")) == 0x8a7b and hp.parse_weight(b"\x01") is None
    assert hp.parse_battery(bytes([87])) == 87 and hp.parse_battery(bytes([200])) is None


def test_stable_weight():
    s = hp.StableWeight()
    assert [s.add(v, NOW + i) for i, v in enumerate([500, 640, 300, 301, 302])] == [None, None, None, None, 302]
    assert s.stable == 302


def test_refill_detector():
    d = hp.RefillDetector()
    d.on_cap(True, 300, NOW)                     # open with 300 in the bottle
    d.on_cap(False, None, NOW + 20)              # closed
    assert d.on_stable(310, NOW + 25) is None    # small rise: not a refill (yet)
    ev = d.on_stable(480, NOW + 30)
    assert ev == {"pre": 300, "post": 480, "rise": 180}
    assert d.on_stable(500, NOW + 31) is None    # reported once
    d.on_cap(True, 480, NOW + 100); d.on_cap(False, None, NOW + 110)
    assert d.on_stable(470, NOW + 115) is None   # opened to drink: no refill
    assert d.on_stable(600, NOW + 150) is None   # outside the 30 s window


def test_fill_from_weight():
    s = {"capacity_ml": 600, "weight_full_raw": 900, "weight_empty_raw": 300}
    assert hp.fill_from_weight(600, s) == (300, 50)
    assert hp.fill_from_weight(950, s) == (600, 100) and hp.fill_from_weight(100, s) == (0, 0)
    assert hp.fill_from_weight(800, {"capacity_ml": 600, "weight_full_raw": 900}) == (500, 83)   # full anchor only
    assert hp.fill_from_weight(800, {"capacity_ml": 600}) == (None, None)                       # not calibrated
    assert hp.fill_from_weight(None, s) == (None, None)


def test_dedupe():
    d = hp.SipDeduper()
    d.seed([(NOW - 100, 40)])
    assert d.is_new(NOW - 99, 40) is False       # same sip replayed (within 2 s)
    assert d.is_new(NOW - 99, 41) is True        # different volume
    assert d.is_new(NOW - 50, 40) is True        # same volume, different time
    assert d.is_new(NOW - 50, 40) is False


def test_pro2_real_frames():
    # captured from a HidrateSpark PRO 2 (21 oz, firmware 100.64.0)
    r = hp.parse_sip_frame(bytes.fromhex("13116600491c01001080be85d085bb8400000000"), 621, NOW)
    assert r["kind"] == "sip" and r["layout"] == "pro2" and r["remaining"] == 19
    assert r["volume_ml"] == 106 and r["pct"] == 17 and r["ts"] == NOW - 72777
    assert r["total_reported"] == 102 and (r["weight_before"], r["weight_after"]) == (34256, 33979)
    live = hp.parse_sip_frame(bytes.fromhex("0103d400010000001080be8583824a8200000000"), 621, NOW)
    assert live["volume_ml"] == 19 and live["ts"] == NOW - 1
    big = hp.parse_sip_frame(bytes.fromhex("0a20c200b1ed00001080be856b828d8000000000"), 621, NOW)
    assert big["volume_ml"] == 199 and big["ts"] == NOW - 60849
    assert hp.parse_sip_frame(bytes.fromhex("0100000000000000000000000000000000000000"), 621, NOW) == {"kind": "pending", "remaining": 1}


def test_drainer_one_at_a_time():
    import asyncio
    sent = []
    async def write(): sent.append(1)
    d = hp.Drainer(write)
    async def go():
        assert await d.request(NOW) is True
        assert await d.request(NOW + 0.1) is False      # still waiting for the bottle's frame
        d.on_frame(NOW + 0.2)
        assert await d.request(NOW + 0.3) is True
        assert await d.request(NOW + 3.0) is True        # timed out waiting -> may ask again
    asyncio.run(go())
    assert len(sent) == 3


def test_is_bottle():
    assert hp.is_bottle("h2o 1A2B") and hp.is_bottle("HidrateSpark PRO") and hp.is_bottle("H2O")
    assert hp.is_bottle("", [hp.SERVICE_REF]) and not hp.is_bottle("Polar H10 1234")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
    print(f"{len(fns)} tests passed")
