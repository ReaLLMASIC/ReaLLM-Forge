"""Evaluation checks use analytic probabilities, independent of trained weights."""
import json
import math
from types import SimpleNamespace

import pytest
import torch

from evaluate import (evaluate_multiple_choice, score_continuation, score_text,
                      unicode_byte_chunks)


class ByteTokenizer:
    # Every route is bytes here, to test counting separately from BPE implementation.
    def encode_route(self, text, route, add_bos=False, add_eos=False):
        return ([self.bos_id(route)] if add_bos else []) + list(text.encode("utf-8")) + ([self.eos_id(route)] if add_eos else [])

    def bos_id(self, route):
        return 256 + 3 * int(route)

    def eos_id(self, route):
        return 257 + 3 * int(route)


class UniformModel(torch.nn.Module):
    def __init__(self, context=8):
        super().__init__()
        self.placeholder = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(max_position_embeddings=context)
        self.seen = []

    def forward(self, input_ids, tokenizer_ids, **kwargs):
        self.seen.append(input_ids[0].tolist())
        shape = (*input_ids.shape, 271)
        logits = torch.full(shape, -float("inf"), device=input_ids.device)
        logits[..., :256] = 0
        logits[..., 257 + 3 * int(tokenizer_ids)] = 0
        return SimpleNamespace(logits=logits)


@pytest.mark.parametrize("mode", ["token", "bytes"])
def test_bpb_counts_each_byte_once_and_only_true_eos(mode):
    model, tokenizer = UniformModel(context=8), ByteTokenizer()
    text = "中 A🙂\t\r\n0123456789"
    scored = score_text(model, tokenizer, text, "zh", mode)
    size = len(text.encode("utf-8"))
    assert scored["text_tokens"] == scored["source_bytes"] == size
    assert scored["text_nll"] == pytest.approx(size * math.log(257), rel=1e-6)
    assert scored["bpb"] == pytest.approx(math.log2(257), rel=1e-6)
    assert scored["eos_count"] == 1
    assert scored["eos_nll"] == pytest.approx(math.log(257), rel=1e-6)
    assert scored["fallback_source_byte_fraction"] == 1
    assert all(len(window) <= 8 for window in model.seen)
    assert sum(window.count(tokenizer.eos_id(1)) for window in model.seen) == 1


def test_byte_chunks_preserve_unicode_and_whitespace():
    text = "\r\n𠮷🙂 e\u0301 中  "
    chunks = unicode_byte_chunks(text, 5)
    assert "".join(chunks) == text
    assert all(0 < len(chunk.encode("utf-8")) <= 5 for chunk in chunks)
    with pytest.raises(ValueError):
        unicode_byte_chunks(text, 3)


def test_mc_scores_only_answer_and_rejects_context_overflow():
    model, tokenizer = UniformModel(context=16), ByteTokenizer()
    score = score_continuation(model, tokenizer, "prompt: ", "中", "zh")
    assert score == pytest.approx(-3 * math.log(257), rel=1e-6)
    with pytest.raises(ValueError, match="exceed context"):
        score_continuation(model, tokenizer, "x" * 15, "yes", "en")


def test_mc_shared_coverage_and_protocol_are_explicit(tmp_path):
    path = tmp_path / "mc.jsonl"
    rows = [
        {"language": "en", "prompt": "Q: ", "choices": ["a", "bbb"], "answer_index": 0},
        {"language": "zh", "prompt": "中" * 10, "choices": ["甲", "乙"], "answer_index": 0},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    model, tokenizer = UniformModel(context=16), ByteTokenizer()
    # Test tokenizer accepts both route strings and indices, as the real wrapper does.
    original = tokenizer.encode_route
    tokenizer.encode_route = lambda text, route, **kw: original(text, ("byte", "zh", "ko", "ja", "en").index(route) if isinstance(route, str) else route, **kw)
    result = evaluate_multiple_choice(model, tokenizer, path)
    assert result["scored_count"] == 1
    assert result["skipped_count"] == 1
    assert result["official_benchmark_score"] is False
    assert result["language_metrics"]["en"]["native_accuracy"] == 1
    assert result["language_metrics"]["en"]["byte_accuracy"] == 1


def test_empty_document_has_eos_but_no_invented_bytes():
    scored = score_text(UniformModel(), ByteTokenizer(), "", "byte", "bytes")
    assert scored["source_bytes"] == scored["text_tokens"] == 0
    assert scored["eos_count"] == 1
    assert scored["bpb"] is None
