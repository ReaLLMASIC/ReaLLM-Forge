import json
import sys
from types import SimpleNamespace

import pytest

from semantic_factorization import (
    analyze_tokens,
    load_huggingface_vocabulary,
    load_tiktoken_vocabulary,
    load_tokenizer_vocabulary,
)


def test_factorization_rules_and_overlap():
    tokens = {"hello", "▁hello", "Hello", "▁Hello", "NASA", "nasa", "▁NASA", "unrelated"}
    result = analyze_tokens(tokens)
    assert result.leading_space == {"▁hello", "▁Hello", "▁NASA"}
    assert result.initial_capital == {"Hello", "▁Hello"}
    # ▁NASA has no ▁nasa counterpart, so it is removable only by SPACE here.
    assert result.all_caps == {"NASA"}
    assert result.exclusive_counts() == (3, 1, 1)
    assert len(result.removed) == 5
    assert result.factorized_vocabulary_size == 3


def test_all_caps_requires_a_cased_character_and_counterpart():
    assert not analyze_tokens({"123", "!", "ABC", "mixed", "▁MISSING"}).all_caps


def test_common_space_representations_are_supported():
    assert analyze_tokens({"word", "Ġword"}).leading_space == {"Ġword"}
    result = analyze_tokens({"word", " word", "Word", " WORD"})
    assert result.leading_space == {" word"}
    assert result.initial_capital == {"Word"}
    assert result.all_caps == {" WORD"}


def test_load_tokenizer_vocabulary(tmp_path):
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps({"model": {"vocab": {"a": 0, "A": 1}}}), encoding="utf-8")
    assert load_tokenizer_vocabulary(path) == {"a": 0, "A": 1}


def test_load_tokenizer_vocabulary_rejects_wrong_shape(tmp_path):
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps({"model": {"vocab": "wrong"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="model.vocab"):
        load_tokenizer_vocabulary(path)


def test_load_tiktoken_vocabulary_preserves_reserved_head_rows(monkeypatch):
    encoding = SimpleNamespace(
        n_vocab=5,
        _mergeable_ranks={b"word": 0, b" word": 1, b"\xff": 2},
        _special_tokens={"<special>": 4},
    )
    monkeypatch.setitem(sys.modules, "tiktoken", SimpleNamespace(get_encoding=lambda _name: encoding))
    vocabulary = load_tiktoken_vocabulary("fake")
    assert len(vocabulary) == 5
    assert vocabulary["<reserved:3>"] == 3
    assert vocabulary["<invalid-utf8:ff>"] == 2


def test_huggingface_vocabulary_matches_model_head_extent(monkeypatch):
    tokenizer = SimpleNamespace(vocab_size=3, get_vocab=lambda: {"a": 0, "b": 2, "added": 5})
    transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: tokenizer),
        AutoConfig=SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: SimpleNamespace(vocab_size=4)),
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    vocabulary = load_huggingface_vocabulary("fake/model")
    assert len(vocabulary) == 4
    assert vocabulary["<reserved:1>"] == 1
    assert vocabulary["<reserved:3>"] == 3
    assert "added" not in vocabulary
