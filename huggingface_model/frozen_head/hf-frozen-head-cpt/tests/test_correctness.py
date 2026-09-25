"""Tests protect the scientific comparison: objective, freezing, swaps, split isolation, and resume."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast,
                          Qwen2Config, Qwen2ForCausalLM)

from analyze import matched_point, report
from common import anchor_dir, load_config, read_json, run_dir, write_json
from diagnostics import data_blocks, head_swap
from experiment import choose_stream, latest_checkpoint, train
from model_utils import chunked_loss, configure_mode, frozen_hashes, is_tied
from prepare_data import Packer, TokenBlocks, prepare, route_record

torch.set_num_threads(1)


def small_model(kind="qwen2"):
    torch.manual_seed(7)
    if kind == "qwen2":
        return Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
             num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
             max_position_embeddings=64, tie_word_embeddings=True, attention_dropout=0.0))
    return GPT2LMHeadModel(GPT2Config(vocab_size=32, n_embd=16, n_layer=1, n_head=2, n_positions=64,
                                    resid_pdrop=0, embd_pdrop=0, attn_pdrop=0))


@pytest.mark.parametrize("kind", ["qwen2", "gpt2"])
@pytest.mark.parametrize("mode", ["full", "freeze_head", "freeze_embed", "freeze_both"])
def test_chunked_ce_matches_hf_loss_and_gradients(kind, mode):
    m = small_model(kind)
    configure_mode(m, mode)
    direct = copy.deepcopy(m)
    m.eval()
    direct.eval()
    batch = torch.randint(0, 32, (2, 9))
    our_loss = chunked_loss(m, batch, 3)
    logits = direct(input_ids=batch[:, :-1], use_cache=False).logits
    expected = F.cross_entropy(logits.reshape(-1, 32), batch[:, 1:].reshape(-1))
    torch.testing.assert_close(our_loss, expected, atol=1e-6, rtol=1e-6)
    our_loss.backward()
    expected.backward()
    for (name, p), (name2, q) in zip(m.named_parameters(), direct.named_parameters()):
        assert name == name2
        if not p.requires_grad:
            assert p.grad is None
        else:
            torch.testing.assert_close(p.grad, q.grad, atol=3e-6, rtol=2e-4)


def test_head_only_freeze_does_not_freeze_input_or_backbone():
    m = small_model()
    assert is_tied(m)
    before = m.get_input_embeddings().weight.detach().clone()
    configure_mode(m, "freeze_head")
    assert not is_tied(m)
    hashes = frozen_hashes(m)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=0.01, weight_decay=0.1)
    chunked_loss(m, torch.randint(0, 32, (1, 9)), 3).backward()
    opt.step()
    assert frozen_hashes(m) == hashes
    assert not torch.equal(m.get_input_embeddings().weight, before)


def test_packing_overlap_and_targets(tmp_path):
    p = Packer(tmp_path / "tokens.bin", 4, 8, 32)
    p.add([0, 1, 2])
    p.add([3, 4, 5, 6, 7, 8, 9])
    meta = p.close()
    b = TokenBlocks(tmp_path / "tokens.bin", 4)
    assert b.data.tolist() == [[0, 1, 2, 3, 4], [4, 5, 6, 7, 8]]
    assert meta["target_tokens"] == 8
    assert np.load(tmp_path / "tokens.counts.npy")[1:9].tolist() == [1] * 8


def test_repository_split_and_replay_counts():
    spec = {"text_field": "content", "group_field": "repo_name"}
    assert route_record({"content": "file one", "repo_name": "same/repo"}, spec) == route_record(
        {"content": "other file", "repo_name": "same/repo"}, spec)
    seq = [choose_stream(i, 0.1) for i in range(100)]
    assert [x[1] for x in seq if x[0] == "a"] == list(range(10))
    assert [x[1] for x in seq if x[0] == "b"] == list(range(90))


def fixture_config(tmp_path, output_name="run"):
    source = tmp_path / "source"
    source.mkdir(exist_ok=True)
    if not (source / "config.json").exists():
        model = small_model()
        model.save_pretrained(source)
        vocab = {"[UNK]": 0, "[EOS]": 1, "[PAD]": 2}
        vocab.update({f"t{i}": i + 3 for i in range(29)})
        tok = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
        tok.pre_tokenizer = Whitespace()
        fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", eos_token="[EOS]", pad_token="[PAD]")
        fast.save_pretrained(source)
    data = {}
    for domain in ("a", "b"):
        mapping = {}
        for k, split in enumerate(("train", "validation", "test")):
            path = tmp_path / f"{domain}_{split}.jsonl"
            records = []
            for j in range(30):
                # Distinct strings across all partitions; controlled language shift in the common suffix.
                prefix = f"t{j % 29} t{k + 10} t{0 if domain == 'a' else 1}"
                tail = " ".join(f"t{(i + j) % 10 + (0 if domain == 'a' else 15)}" for i in range(32))
                records.append({"text": prefix + " " + tail})
            path.write_text("".join(json.dumps(x) + "\n" for x in records))
            mapping[split] = str(path)
        data[domain] = {"local_jsonl": mapping, "text_field": "text", "min_chars": 1,
                        "splits": {s: s for s in mapping}, "train_tokens": 256,
                        "validation_tokens": 32, "test_tokens": 32}
    raw = {"model": str(source), "output": str(tmp_path / output_name), "device": "cpu", "seq_len": 8,
           "micro_batch": 1, "accumulation": 2, "head_chunk": 3, "seeds": [12], "lrs": [0.003],
           "a_lr": 0.003, "a_steps": 2, "b_steps": 4, "eval_every": 2, "save_every": 2,
           "log_every": 1, "probe_blocks": 1, "cpu_threads": 1, "deterministic": True,
           "modes": ["full", "freeze_head"], "data": data}
    path = tmp_path / f"{output_name}.json"
    write_json(path, raw)
    return load_config(path)


def test_end_to_end_and_exact_resume(tmp_path):
    c = fixture_config(tmp_path, "continuous")
    prepare(c)
    assert train(c, 12, "a", "full", 0.003) == 0
    assert train(c, 12, "b", "full", 0.003) == 0
    assert train(c, 12, "b", "freeze_head", 0.003) == 0
    root = run_dir(c, 12, "freeze_head", 0.003)
    assert read_json(root / "frozen_initial.json") == read_json(root / "frozen_final.json")
    rows = [json.loads(x) for x in (root / "metrics.jsonl").read_text().splitlines()]
    swap = rows[0]["head_swap"]
    assert max(swap[k] for k in ("w0_h0", "wt_h0", "w0_ht", "wt_ht")) - min(
        swap[k] for k in ("w0_h0", "wt_h0", "w0_ht", "wt_ht")) < 1e-6
    assert rows[-1]["head_geometry"]["relative_frobenius_drift"] == 0
    report(c, gains=[0.001, 100], plots=False)
    assert (Path(c["output"]) / "analysis" / "final_test.csv").exists()
    resumed = fixture_config(tmp_path, "resumed")
    prepare(resumed)
    assert train(resumed, 12, "a", "full", 0.003) == 0
    assert train(resumed, 12, "b", "full", 0.003, stop_after=2) == 75
    assert train(resumed, 12, "b", "full", 0.003) == 0
    original = Qwen2ForCausalLM.from_pretrained(latest_checkpoint(run_dir(c, 12, "full", 0.003)))
    restored = Qwen2ForCausalLM.from_pretrained(latest_checkpoint(run_dir(resumed, 12, "full", 0.003)))
    for key, value in original.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key]), key
    assert read_json(run_dir(c, 12, "full", 0.003) / "test.json") == read_json(
        run_dir(resumed, 12, "full", 0.003) / "test.json")


def test_matching_uses_temporal_crossing():
    rows = [{"step": s, "a": {"loss": a}, "b": {"loss": b}} for s, a, b in
            [(0, 1, 2), (10, 1.2, 1.8), (20, 1.1, 1.9), (30, 1.4, 1.6)]]
    m = matched_point(rows, 0.15)
    assert m["left_step"] == 0 and m["right_step"] == 10
    assert m["forgetting"] == pytest.approx(0.15)
    assert matched_point(rows, 1.0) is None


def test_cuda_device_resolution_and_allocator_budget(monkeypatch):
    from types import SimpleNamespace
    from model_utils import setup_device
    seen = {}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "set_device", lambda d: seen.update(device=d))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda d: SimpleNamespace(total_memory=80 * 2**30))
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda f, d: seen.update(fraction=f))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda d: (78 * 2**30, 80 * 2**30))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda d: "Mock H100; allocation logic only")
    d, cap = setup_device({"device": "cuda", "cpu_threads": 1, "memory_limit_gib": 70, "memory_fraction": 0.9})
    assert d == torch.device("cuda:0")
    assert seen["device"].index == 0
    assert seen["fraction"] == pytest.approx(70 / 80)
    assert cap == 70 * 2**30
