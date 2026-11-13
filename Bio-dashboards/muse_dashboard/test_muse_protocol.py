"""Tests for muse_protocol.py.  Run:  python3 -m pytest muse_dashboard   (or: python3 muse_dashboard/test_muse_protocol.py)

The decoder is checked against OpenMuse's own decoder: test_data/openmuse_reference.json holds random
packets and what OpenMuse's parse_message made of them (so OpenMuse isn't needed to run this)."""
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import muse_protocol as mp

HERE = os.path.dirname(os.path.abspath(__file__))


def test_commands():
    assert mp.encode_command("p1041") == b"\x06p1041\n"
    assert mp.encode_command("dc001") == b"\x06dc001\n"
    assert mp.encode_command("h") == b"\x02h\n"
    assert all(t.format(preset="p1041") for t in mp.START_SEQUENCE)


def test_pack_unpack_round_trip():
    for width, count, nbytes in ((14, 16, 28), (20, 16, 40), (20, 12, 30)):
        vals = [(i * 7919 + 13) % (1 << width) for i in range(count)]
        assert mp.unpack(mp.pack(vals, width, nbytes), width, count) == vals


def test_bit_order_lsb_first():
    # value 1 in the first 14-bit slot sets bit 0 of byte 0; value 1 in the second sets bit 14 = byte 1, bit 6
    assert mp.pack([1], 14, 28)[0] == 1
    assert mp.pack([0, 1], 14, 28)[1] == 0b0100_0000


def test_scales():
    eeg = mp.decode_block(0x12, mp.pack([16383] + [0] * 15, 14, 28))
    assert eeg[0][0] == 1450.0 and eeg[0][1] == 0.0
    imu = mp.decode_block(0x47, struct.pack("<18h", 16384, 0, 0, 100, 0, 0, *([0] * 12)))
    assert abs(imu[0][0] - 1.0) < 1e-3 and abs(imu[0][3] - (-0.748)) < 1e-3       # 1 g; gyro scale is negative
    assert mp.decode_block(0x98, struct.pack("<H", 87 * 256 + 128) + bytes(18)) == [[87.5]]


def test_against_openmuse():
    ref = json.load(open(os.path.join(HERE, "test_data", "openmuse_reference.json")))
    rows = 0
    for case in ref["cases"]:
        mine = mp.parse_notification(bytes.fromhex(case["hex"]))
        for kind, blocks in case["openmuse"].items():
            got = [b["rows"] for b in mine if b["kind"] == kind]
            assert len(got) == len(blocks), (kind, case["hex"][:40])
            for a, b in zip(blocks, got):
                for ra, rb in zip(a, b):
                    rows += 1
                    tol = 0.01 if kind == "battery" else 1e-3
                    assert all(abs(x - y) <= tol * max(1, abs(x)) for x, y in zip(ra, rb)), (kind, ra[:3], rb[:3])
    assert rows > 200


def test_several_packets_and_junk():
    p1 = mp.build_packet(0x47, 1, 1000, bytes(36))
    p2 = mp.build_packet(0x12, 2, 2000, bytes(28), [(0x12, 3, bytes(28))])
    got = mp.parse_notification(p1 + p2)
    assert [b["kind"] for b in got] == ["imu", "eeg", "eeg"] and got[1]["tick"] == 2000 and got[1]["pkt_index"] == 2
    assert mp.parse_notification(p1 + b"\x05\x00") == mp.parse_notification(p1)          # a cut-off tail is dropped
    assert mp.parse_notification(b"") == []
    # an unknown tag ends its packet; the next packet still decodes
    odd = mp.build_packet(0x47, 4, 0, bytes(36), [(0x47, 5, bytes(36))])
    odd = odd[:14 + 36] + b"\xEE" + odd[14 + 37:]
    assert [b["kind"] for b in mp.parse_notification(odd + p2)] == ["imu", "eeg", "eeg"]


def test_control_answers():
    text = '{"hn":"MuseS-1A2B","fw":"3.2.1","bp":87.5}' + '{"rc":0}'
    assert mp.parse_control(text) == {"hn": "MuseS-1A2B", "fw": "3.2.1", "bp": 87.5, "rc": 0.0}


def test_clock_counts_samples_and_gaps():
    c = mp.MuseClock()
    t = 1_790_000_000_000.0
    a = c.stamps("eeg", 0, 2, t)
    b = c.stamps("eeg", 0, 2, t + 1)                   # same packet time: the next samples, not the same ones
    assert b[0] - a[0] in (7, 8) and a[1] - a[0] in (3, 4)
    tick = int(1.0 * mp.CLOCK_HZ)                       # 1 s later by the headband's clock: 252 samples missing
    d = c.stamps("eeg", tick, 2, t + 1000)
    assert c.dropped["eeg"] == 256 - 4 and abs(d[0] - (a[0] + 1000)) <= 1


def test_clock_shares_one_offset_and_wraps():
    c = mp.MuseClock()
    t = 1_790_000_000_000.0
    e = c.stamps("eeg", 256_000, 2, t)
    o = c.stamps("optics", 256_000, 1, t + 3)          # same device time on another stream -> same epoch time
    assert abs(o[0] - e[0]) <= 1
    near = (1 << 32) - 2560                             # 10 ms before the 32-bit clock wraps
    c2 = mp.MuseClock()
    x = c2.stamps("imu", near, 3, t)
    y = c2.stamps("imu", 2560 + 3 * 4923, 3, t + 60)    # just after it: forward, not 4.7 h back
    assert 0 < y[0] - x[0] < 200 and c2.reanchors == 0


def test_clock_follows_drift_slowly_and_resets_on_jumps():
    c = mp.MuseClock()
    t = 1_790_000_000_000.0
    c.stamps("eeg", 0, 2, t)
    off = c.offset_ms
    c.stamps("eeg", 512, 2, t + 600)                    # arrivals 600 ms late: moves 1 ms, not 600
    assert abs(c.offset_ms - off) == 1
    c.stamps("eeg", 1024, 2, t + 20_000)                # 20 s off: the headband restarted its clock
    assert c.reanchors == 1


if __name__ == "__main__":
    n = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); n += 1; print("  ok ", name)
    print(f"{n} tests passed")
