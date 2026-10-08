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
    # captured from a HidrateSpark PRO 2 (21 oz, firmware 100.64.0); MIN 32784, MAX 34238
    r = hp.parse_sip_frame(bytes.fromhex("13116600491c01001080be85d085bb8400000000"), 621, NOW)
    assert r["kind"] == "sip" and r["layout"] == "pro2" and r["remaining"] == 19
    assert r["pct"] == 17 and r["ts"] == NOW - 72777
    assert r["total_reported"] == 102 and (r["weight_before"], r["weight_after"]) == (34256, 33979)
    assert (r["cal_min"], r["cal_max"]) == (32784, 34238)
    # volume from the weights (before is above MAX, so it clamps to full): 110.6 -> 111 mL
    assert r["volume_ml"] == 111 and r["volume_source"] == "weights"
    live = hp.parse_sip_frame(bytes.fromhex("0103d400010000001080be8583824a8200000000"), 621, NOW)
    assert live["volume_ml"] == 24 and live["ts"] == NOW - 1            # byte 1 alone would say 19
    big = hp.parse_sip_frame(bytes.fromhex("0a20c200b1ed00001080be856b828d8000000000"), 621, NOW)
    assert big["volume_ml"] == 204 and big["ts"] == NOW - 60849
    assert hp.parse_sip_frame(bytes.fromhex("0100000000000000000000000000000000000000"), 621, NOW) == {"kind": "pending", "remaining": 1}


def test_pro2_falls_back_to_percent_byte():
    # no usable calibration in the record (MIN = MAX = 0): the whole-percent byte is used
    f = bytearray.fromhex("0103d400010000001080be8583824a8200000000")
    f[8:12] = b"\x00\x00\x00\x00"
    r = hp.parse_sip_frame(bytes(f), 621, NOW)
    assert r["volume_ml"] == 19 and r["volume_source"] == "percent" and r["cal_min"] is None
    # weights that disagree wildly with byte 1 (calibration off): byte 1 wins
    f = bytearray.fromhex("0103d400010000001080be8583824a8200000000")
    f[12:14] = (34200).to_bytes(2, "little")
    r = hp.parse_sip_frame(bytes(f), 621, NOW)
    assert r["volume_ml"] == 19 and r["volume_source"] == "percent"


def test_calibration_and_fill():
    assert hp.valid_calibration(32784, 34238) and not hp.valid_calibration(0, 0) and not hp.valid_calibration(100, 150)
    assert hp.fill_from_calibration(33979, 32784, 34238, 621) == (510, 82)
    assert hp.fill_from_calibration(32909, 32784, 34238, 621) == (53, 9)
    assert hp.fill_from_calibration(40000, 32784, 34238, 621) == (621, 100)       # clamps
    assert hp.fill_from_calibration(33000, None, None, 621) == (None, None)
    assert hp.weight_in_band(33500, 32784, 34238) and hp.weight_in_band(34400, 32784, 34238)
    assert not hp.weight_in_band(35451, 32784, 34238) and not hp.weight_in_band(31626, 32784, 34238)
    assert hp.parse_bottle_size(bytes.fromhex("6d02")) == 621 and hp.parse_bottle_size(b"\x00\x00") is None


def test_family_and_drain_mode():
    assert hp.bottle_family("100.64.0") == "telink" and hp.bottle_family("80.18") == "nordic"
    assert hp.bottle_family("53.64") == "cypress" and hp.bottle_family("") == "unknown"
    assert hp.drain_mode_for("100.64.0") == "ack" and hp.drain_mode_for("80.18") == "fast"
    assert hp.drain_mode_for(None) == "fast"


def test_dedupe_volume_tolerance():
    d = hp.SipDeduper(volume_tol=9)
    d.seed([(NOW - 100, 19)])                       # logged earlier from the percent byte
    assert d.is_new(NOW - 100, 24) is False         # same sip, replayed with the weight-based volume
    assert d.is_new(NOW - 100, 60) is True          # a different sip in the same second (unlikely, but kept)


def test_drainer_one_at_a_time():
    import asyncio
    sent = []
    async def write(b): sent.append(b)
    d = hp.Drainer(write)
    async def go():
        assert await d.request(NOW) is True
        assert await d.request(NOW + 0.1) is False      # still waiting for the bottle's frame
        d.on_frame(NOW + 0.2)
        assert await d.request(NOW + 0.3) is True
        assert await d.request(NOW + 3.0) is True        # timed out waiting -> may ask again
    asyncio.run(go())
    assert sent == [hp.READY_FAST] * 3


def test_drainer_ack_mode_and_fallback():
    import asyncio
    sent = []
    async def write(b): sent.append(b)
    d = hp.Drainer(write, mode="ack")
    async def go():
        assert await d.request(NOW) is True                     # first request: bare READY
        d.on_frame(NOW + 0.2)
        assert await d.request(NOW + 0.3, ack=True) is True     # acknowledge, then ask for the next
        assert await d.request(NOW + 0.4, ack=True) is False    # one at a time
    asyncio.run(go())
    assert sent == [hp.READY, hp.ACK, hp.READY] and d.acks == 1
    # three requests in a row with no answer while records are waiting -> fall back to fast mode
    sent.clear()
    d = hp.Drainer(write, mode="ack")
    async def silent():
        t = NOW
        for _ in range(4):
            await d.request(t)
            t += 3
        await d.request(t)
    asyncio.run(silent())
    assert d.fell_back and d.mode == "fast" and sent[-1] == hp.READY_FAST


def test_reminder_times():
    assert hp.reminder_times({"enabled": False}) == []
    t = hp.reminder_times({"enabled": True, "from": "08:00", "to": "22:00", "every_min": 60})
    assert len(t) == 14 and t[0] == 9 * 3600 and t[-1] == 22 * 3600
    late = hp.reminder_times({"enabled": True, "from": "20:00", "to": "01:00", "every_min": 120})
    assert late == [22 * 3600, 0]                                   # wraps past midnight
    assert hp.reminder_times({"enabled": True, "every_min": 5}) == []   # too frequent: ignored


def test_build_sync():
    s = {"capacity_ml": 621, "goal_ml": 2500, "goal_glow": True,
         "reminders": {"enabled": True, "from": "08:00", "to": "22:00", "every_min": 60, "always": False, "sound": True}}
    w = hp.build_sync(s, "100.64.0", 813, 15 * 3600 + 18 * 60)
    assert w[0] == (hp.CHAR_DEBUG, bytes.fromhex("210082"))            # 813 mL = 130 % of 621
    assert w[1] == (hp.CHAR_SET_POINT, bytes.fromhex("933d"))
    assert w[2] == (hp.CHAR_DEBUG, bytes.fromhex("220192"))            # 2500 mL = 402 %
    assert w[3] == (hp.CHAR_SET_POINT, bytes.fromhex("7700000028d70000"))   # 15:18:00 = 55080 s
    slots = w[4:]
    assert len(slots) == 48 and all(c == hp.CHAR_SET_POINT and len(b) == 10 for c, b in slots)
    first = slots[0][1]                                                # 09:00, 1/15 of the goal = 26 %
    assert first == bytes([0, 0x34, 26, 0]) + (9 * 3600).to_bytes(4, "little") + bytes([1, 1])
    assert slots[13][1][4:8] == (22 * 3600).to_bytes(4, "little") and slots[14][1] == bytes([14]) + bytes(9)
    # "glow regardless", sip glow set, older bottle: 12 slots of 8 bytes
    s2 = {**s, "sip_glow": False, "reminders": {**s["reminders"], "always": True}}
    w2 = hp.build_sync(s2, "80.18", 0, 0)
    assert w2[4] == (hp.CHAR_DEBUG, hp.SIP_GLOW_OFF)
    assert len(w2[5:]) == 12 and len(w2[5][1]) == 8 and w2[5][1][2:4] == b"\xff\xff"
    assert hp.glow_now_bytes("100.64.0") == bytes([71]) and hp.glow_now_bytes("80.18") == bytes([48])


def _pb_decode(data):
    """Minimal protobuf reader for the tests: [(field, int | bytes)]."""
    out, i = [], 0
    def varint():
        nonlocal i
        n = shift = 0
        while True:
            b = data[i]; i += 1
            n |= (b & 0x7F) << shift; shift += 7
            if not b & 0x80:
                return n
    while i < len(data):
        key = varint()
        if key & 7 == 0:
            out.append((key >> 3, varint()))
        else:
            n = varint(); out.append((key >> 3, data[i:i + n])); i += n
    return out


def test_glow_pattern_encoding():
    assert hp.led_count("Sensor-SM-V1") == 10 and hp.led_count("Sensor-LG-V1") == 13 and hp.led_count(None) == 13
    pat = hp.glow_pattern("pulse", ["#2bff00", "#1499ff"], 10)
    data = hp.encode_glow_pattern(pat)
    top = _pb_decode(data)
    assert [f for f, _ in top] == [1, 2, 2, 2, 3, 4] and top[0][1] == 3 and top[4][1] == 1
    assert _pb_decode(top[1][1]) == [(1, 1), (2, 100), (4, 5)]          # repeats 0 is left out
    group = _pb_decode(top[5][1])
    assert group[0] == (1, 1) and group[1] == (2, 3) and len(group) == 5
    lit = _pb_decode(group[3][1])                                         # the middle frame
    assert lit[0] == (2, 30) and lit[1] == (3, 90) and len(lit) == 12     # 10 LEDs
    assert _pb_decode(lit[2][1]) == [(1, 0x2b), (2, 0xff), (3, 1)]        # #2bff00, blue 0 sent as 1
    dark = _pb_decode(group[2][1])
    assert _pb_decode(dark[2][1]) == [(1, 1), (2, 1), (3, 1)]
    spin = hp.glow_pattern("spin", ["#ff0000", "#0000ff"], 10)
    assert len(spin["groups"][1]) == 10 and spin["groups"][1][1][2][1] == spin["groups"][1][0][2][0]   # rotates by one LED
    count, packets = hp.glow_packets(hp.encode_glow_pattern(spin))
    assert count[0] == len(packets) and all(len(p) == 20 for p in packets[:-1]) and b"".join(packets) == hp.encode_glow_pattern(spin)
    for style in hp.GLOW_STYLES:
        for n in (10, 13):
            assert 0 < len(hp.encode_glow_pattern(hp.glow_pattern(style, ["#ffffff", "#000000"], n))) <= hp.GLOW_MAX_BYTES


def test_is_bottle():
    assert hp.is_bottle("h2o 1A2B") and hp.is_bottle("HidrateSpark PRO") and hp.is_bottle("H2O")
    assert hp.is_bottle("", [hp.SERVICE_REF]) and not hp.is_bottle("Polar H10 1234")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
    print(f"{len(fns)} tests passed")
