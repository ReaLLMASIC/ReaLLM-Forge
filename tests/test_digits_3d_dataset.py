import pickle
import runpy
from pathlib import Path

import numpy as np
import pytest

MODULE = runpy.run_path(Path(__file__).parents[1] / "data/digits_3d/prepare.py")
build_dataset = MODULE["build_dataset"]


def test_letters_are_in_vocab_but_absent_from_splits(tmp_path):
    build_dataset(tmp_path, train_repeats=3, val_repeats=2, num_digits=12, num_letters=6)

    with (tmp_path / "meta.pkl").open("rb") as handle:
        meta = pickle.load(handle)
    train = np.fromfile(tmp_path / "train.bin", dtype=np.uint16)
    val = np.fromfile(tmp_path / "val.bin", dtype=np.uint16)
    held_out_ids = {meta["stoi"][char] for char in "abcdef"}

    assert "".join(meta["itos"][i] for i in range(meta["vocab_size"])) == "0123456789!@abcdef"
    assert held_out_ids.isdisjoint(train)
    assert held_out_ids.isdisjoint(val)
    assert "".join(meta["itos"][int(i)] for i in train) == "0123456789!@" * 3
    assert meta["trained_tokens"] == list("0123456789!@")
    assert meta["unseen_tokens"] == list("abcdef")


@pytest.mark.parametrize(
    ("num_digits", "num_letters", "message"),
    [(0, 4, "num_digits"), (100, 4, "num_digits"), (10, -1, "num_letters"), (10, 100, "num_letters")],
)
def test_symbol_counts_are_validated(tmp_path, num_digits, num_letters, message):
    with pytest.raises(ValueError, match=message):
        build_dataset(tmp_path, 2, 2, num_digits, num_letters)


def test_default_vocabulary_has_ten_trained_and_ten_held_out(tmp_path):
    build_dataset(tmp_path, 2, 2)
    with (tmp_path / "meta.pkl").open("rb") as handle:
        meta = pickle.load(handle)
    assert len(meta["trained_tokens"]) == 10
    assert len(meta["unseen_tokens"]) == 10


def test_dropped_symbols_remain_in_vocab_but_leave_splits(tmp_path):
    build_dataset(tmp_path, 3, 2, num_digits=10, num_letters=2, dropout_count=3)
    with (tmp_path / "meta.pkl").open("rb") as handle:
        meta = pickle.load(handle)
    train = np.fromfile(tmp_path / "train.bin", dtype=np.uint16)
    dropped_ids = {meta["stoi"][token] for token in "789"}
    assert meta["dropped_tokens"] == list("789")
    assert meta["initial_trained_tokens"] == list("0123456789")
    assert dropped_ids.isdisjoint(train)
    assert "".join(meta["itos"][int(i)] for i in train) == "0123456" * 3
