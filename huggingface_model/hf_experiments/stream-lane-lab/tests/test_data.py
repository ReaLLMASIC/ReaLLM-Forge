"""The synthetic stream must not change when split into live chunks."""
import numpy as np
import pytest

from lane_lab.data import sine_series, sequence_bank


def test_chunks_and_prefixes_match_full_stream_exactly():
    full = sine_series(20001, 300, 3)
    assert np.array_equal(full[:73], sine_series(20001, 73, 3))
    assert np.array_equal(full[73:201], sine_series(20001, 128, 3, start=73))
    joined = np.concatenate([sine_series(20001, 100, 3, start=start) for start in [0, 100, 200]])
    assert np.array_equal(full, joined)


def test_adding_channels_does_not_change_existing_channels():
    one = sine_series(77, 200, 1)
    two = sine_series(77, 200, 2)
    three = sine_series(77, 200, 3)
    assert np.array_equal(one, two[:, :1])
    assert np.array_equal(two, three[:, :2])


def test_seed_changes_phases_and_amplitudes_but_bound_is_fixed():
    a, b = sine_series(1, 1000, 3), sine_series(2, 1000, 3)
    assert a.dtype == np.float32
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert abs(a).max() <= .900001 and abs(b).max() <= .900001
    assert not np.allclose(a, b)
    assert np.all(a.std(axis=0) > .25)


def test_sequence_bank_uses_distinct_reproducible_seeds():
    bank = sequence_bank(30000, 4, 64, 2)
    assert bank.shape == (4, 64, 2)
    for i in range(4):
        assert np.array_equal(bank[i], sine_series(30000 + i, 64, 2))


@pytest.mark.parametrize("frames,channels,start", [(0, 1, 0), (5, 0, 0), (5, 33, 0), (5, 1, -1)])
def test_invalid_stream_parameters_raise(frames, channels, start):
    with pytest.raises(ValueError):
        sine_series(1, frames, channels, start=start)
