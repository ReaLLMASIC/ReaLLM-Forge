"""Tests for muse_signal.py with made-up signals whose answers are known.
Run:  python3 -m pytest muse_dashboard   (or: python3 muse_dashboard/test_muse_signal.py)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import muse_protocol as mp
import muse_signal as ms

rng = np.random.default_rng(0)


def eeg_rows(seconds=3):
    t = np.arange(seconds * 256) / 256
    return np.stack([725 + 20 * np.sin(2 * np.pi * 10 * t) + 3 * rng.standard_normal(len(t)),      # alpha
                     725 + 15 * np.sin(2 * np.pi * 20 * t) + 3 * rng.standard_normal(len(t)),      # beta
                     np.full(len(t), 1449.9),                                                     # railed: off the skin
                     725 + 150 * rng.standard_normal(len(t))], 1)                                 # swinging: bad contact


def test_bands_and_contact():
    b = ms.EegBands(mp.MAIN_EEG)
    rows = eeg_rows()
    b.push(rows[:300].tolist())
    assert b.compute() is None                                    # not 2 s yet
    b.push(rows[300:].tolist())
    r = b.compute()
    top = {ch: max(v["rel"], key=v["rel"].get) for ch, v in r.items()}
    assert top["TP9"] == "alpha" and top["AF7"] == "beta"
    assert r["TP9"]["rel"]["alpha"] > 0.8
    assert [r[c]["q"] for c in mp.MAIN_EEG] == [1, 1, 0, 4]
    assert abs(sum(r["TP9"]["rel"].values()) - 1) < 1e-3


def test_band_power_is_in_uv2():
    # a 10 Hz sine of amplitude A has power A^2 / 2: log10(400 / 2) = 2.30 for A = 20 uV
    t = np.arange(512) / 256
    b = ms.EegBands(["X"])
    b.push((725 + 20 * np.sin(2 * np.pi * 10 * t)).reshape(-1, 1).tolist())
    assert abs(b.compute()["X"]["abs"]["alpha"] - np.log10(200)) < 0.05


def pulse_session(bpm=72, seconds=30, channel="LI_IR"):
    chs = mp.OPTICS_BY_COUNT[16]
    n = seconds * 64
    tt = np.arange(n) / 64
    opt = 0.5 + 0.01 * rng.standard_normal((n, 16))
    opt[:, chs.index(channel)] = 0.6 - 0.02 * np.sin(np.pi * bpm / 60 * tt) ** 16 + 0.002 * rng.standard_normal(n)
    h, beats = ms.PpgHeart(chs), []
    for s in range(0, n, 64):
        h.push((1e12 + tt[s:s + 64] * 1000).tolist(), opt[s:s + 64].tolist())
        beats += h.compute()
    return h, beats


def test_pulse_finds_the_channel_and_the_rate():
    h, beats = pulse_session(72, channel="RO_NIR")
    assert h.channel == "RO_NIR" and h.quality > 0.6 and h.hr == 72
    ibi = np.array([b[1] for b in beats])
    assert len(beats) >= 25 and abs(ibi.mean() - 833.3) < 5 and ibi.std() < 15
    assert all(b2[0] > b1[0] for b1, b2 in zip(beats, beats[1:]))                  # each beat once, in order


def test_no_pulse_no_beats():
    chs = mp.OPTICS_BY_COUNT[16]
    h = ms.PpgHeart(chs)
    for s in range(30):
        h.push((1e12 + np.arange(64) * 15.625 + s * 1000).tolist(), (0.5 + 0.01 * rng.standard_normal((64, 16))).tolist())
        assert h.compute() == []
    assert h.hr is None


def test_fnirs_direction():
    chs = mp.OPTICS_BY_COUNT[16]
    f = ms.Fnirs(chs)
    assert f.optodes == {"L": "outer", "R": "outer"}
    assert ms.Fnirs(mp.OPTICS_BY_COUNT[4]).optodes == {"L": "inner", "R": "inner"}
    for s in range(12):
        rows = np.full((64, 16), 0.5)
        if s >= 10:
            rows[:, chs.index("LO_NIR")] *= 0.99            # less 730 nm light back: more deoxy-haemoglobin
        f.push(rows.tolist())
        out = f.compute()
        if s < 10:
            assert out is None                              # the baseline (10 s)
    assert out["L"][1] > 0 and out["L"][0] < 0 and out["R"] == (0.0, 0.0)
    f.rebaseline()
    f.push(np.full((64, 16), 0.5).tolist())
    assert f.compute() is None


if __name__ == "__main__":
    n = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); n += 1; print("  ok ", name)
    print(f"{n} tests passed")
