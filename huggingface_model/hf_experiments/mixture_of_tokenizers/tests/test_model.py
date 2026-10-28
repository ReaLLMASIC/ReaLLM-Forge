"""Behavioral checks for routing, shared parameters, HF round trips, and cache."""

import sys
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configuration_mixture import MixtureConfig
from modeling_mixture import MixtureForCausalLM


@pytest.fixture(autouse=True)
def fast_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_model(k=16):
    config = MixtureConfig(
        lexical_size=k,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        initializer_seed=789,
    )
    return MixtureForCausalLM(config).eval()


def examples(config):
    rows = []
    for route in range(5):
        first = 65 if route == 0 else config.lexical_range(route).start
        rows.append([config.bos_id(route), first, 97, config.eos_id(route)])
    return torch.tensor(rows)


def test_global_counts_and_only_byte_intersection():
    config = MixtureConfig(lexical_size=10000)
    assert config.vocab_size == 40271
    for r in range(5):
        assert config.bos_id(r) not in config.allowed_ids(r)
        assert config.pad_id(r) not in config.allowed_ids(r)
        for s in range(r):
            assert set(config.allowed_ids(r)) & set(config.allowed_ids(s)) == set(range(256))
    assert MixtureConfig(lexical_size=20000).vocab_size == 80271


def test_mixed_route_mask_and_invalid_route_inputs():
    model = tiny_model()
    ids = examples(model.config)
    routes = torch.arange(5)
    output = model(ids, tokenizer_ids=routes)
    for route in range(5):
        allowed = model.allowed_output_mask[route]
        assert torch.isfinite(output.logits[route, :, allowed]).all()
        assert torch.isneginf(output.logits[route, :, ~allowed]).all()
    with pytest.raises(ValueError, match="tokenizer_ids is required"):
        model(ids)
    with pytest.raises(ValueError, match="another route"):
        model(ids[1:2], tokenizer_ids=2)


def test_balanced_loss_matches_separate_routes_and_padding_is_ignored():
    model = tiny_model()
    ids = examples(model.config)
    routes = torch.arange(5)
    labels = ids.clone()
    labels[0, 2:] = -100  # unequal target counts must not change route weights
    mixed = model(ids, tokenizer_ids=routes, labels=labels)
    separate = torch.stack([
        model(ids[r:r+1], tokenizer_ids=r, labels=labels[r:r+1]).loss
        for r in range(5)
    ]).mean()
    torch.testing.assert_close(mixed.loss, separate, atol=1e-6, rtol=1e-6)
    assert mixed.route_target_counts.tolist() == [1, 3, 3, 3, 3]
    padded = torch.cat([ids, torch.tensor([[model.config.pad_id(r)] for r in range(5)])], dim=1)
    mask = torch.ones_like(padded)
    mask[:, -1] = 0
    padded_out = model(padded, tokenizer_ids=routes, labels=padded, attention_mask=mask)
    original = model(ids, tokenizer_ids=routes, labels=ids)
    torch.testing.assert_close(padded_out.loss, original.loss, atol=1e-6, rtol=1e-6)


def test_grouped_projection_loss_and_gradients_equal_global_mask():
    model = tiny_model()
    ids = examples(model.config)
    kwargs = dict(tokenizer_ids=torch.arange(5), labels=ids, use_cache=False)
    full = model(ids, **kwargs)
    full.loss.backward()
    full_grad = model.get_input_embeddings().weight.grad.detach().clone()
    model.zero_grad(set_to_none=True)
    compact = model(ids, return_logits=False, **kwargs)
    compact.loss.backward()
    assert compact.logits is None
    torch.testing.assert_close(full.loss, compact.loss, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(full_grad, model.get_input_embeddings().weight.grad, atol=2e-6, rtol=1e-5)


def test_one_shared_byte_table_receives_gradients_from_every_route():
    model = tiny_model()
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    for route in range(5):
        model.zero_grad(set_to_none=True)
        ids = examples(model.config)[route:route+1]
        model(ids, tokenizer_ids=route, labels=ids, return_logits=False).loss.backward()
        gradient = model.get_input_embeddings().weight.grad
        assert gradient[:256].abs().sum() > 0
        if route != 4:
            other = model.config.lexical_range(4)
            assert gradient[other.start:other.stop].abs().sum() == 0


def test_cache_matches_full_prefix_and_route_embedding_is_used_every_step():
    model = tiny_model()
    ids = examples(model.config)[2:3]
    allowed = model.allowed_output_mask[2]
    with torch.no_grad():
        full = model(ids, tokenizer_ids=2, use_cache=False)
        prefix = model(ids[:, :-1], tokenizer_ids=2, use_cache=True)
        step = model(ids[:, -1:], tokenizer_ids=2, past_key_values=prefix.past_key_values, use_cache=True)
    torch.testing.assert_close(full.logits[:, -1, allowed], step.logits[:, -1, allowed], atol=2e-5, rtol=2e-5)
    # A model input hook sees route-added embeddings in the cached one-token call.
    seen = []
    hook = model.gpt_neox.register_forward_pre_hook(lambda module, args, kwargs: seen.append(kwargs["inputs_embeds"].detach()), with_kwargs=True)
    with torch.no_grad():
        model(ids[:, -1:], tokenizer_ids=2, use_cache=False)
    hook.remove()
    expected = model.get_input_embeddings()(ids[:, -1:]) + model.tokenizer_embeddings.weight[2]
    torch.testing.assert_close(seen[0], expected)


def test_hf_generation_keeps_route_and_rejects_mixed_generation_batches():
    model = tiny_model()
    route = 3
    ids = torch.tensor([[model.config.bos_id(route), 65]])
    seen_routes = []
    hook = model.register_forward_pre_hook(lambda module, args, kwargs: seen_routes.append(kwargs["tokenizer_ids"].clone()), with_kwargs=True)
    output = model.generate(ids, tokenizer_ids=route, max_new_tokens=4, do_sample=False, use_cache=True)
    hook.remove()
    assert len(seen_routes) >= 1
    assert all(x.tolist() == [route] for x in seen_routes)
    assert set(output[0, ids.shape[1]:].tolist()) <= set(model.config.allowed_ids(route))
    with pytest.raises(ValueError, match="one route"):
        model.generate(examples(model.config)[:2, :2], tokenizer_ids=torch.tensor([0, 1]), max_new_tokens=2)


def test_save_reload_autoclass_and_tied_weights(tmp_path):
    model = tiny_model()
    ids = examples(model.config)
    expected = model(ids, tokenizer_ids=torch.arange(5)).logits
    model.save_pretrained(tmp_path, safe_serialization=True)
    config = AutoConfig.from_pretrained(tmp_path)
    restored = AutoModelForCausalLM.from_pretrained(tmp_path).eval()
    assert isinstance(config, MixtureConfig)
    assert isinstance(restored, MixtureForCausalLM)
    assert restored.get_input_embeddings().weight is restored.get_output_embeddings().weight
    actual = restored(ids, tokenizer_ids=torch.arange(5)).logits
    torch.testing.assert_close(actual, expected)


def test_backbone_bytes_and_nested_lexical_prefixes_match_across_k():
    small = tiny_model(16)
    large = tiny_model(32)
    for (name_a, param_a), (name_b, param_b) in zip(small.gpt_neox.layers.named_parameters(), large.gpt_neox.layers.named_parameters()):
        assert name_a == name_b
        torch.testing.assert_close(param_a, param_b, rtol=0, atol=0)
    torch.testing.assert_close(small.get_input_embeddings().weight[:256], large.get_input_embeddings().weight[:256], rtol=0, atol=0)
    for route in range(1, 5):
        start_a = small.config.lexical_range(route).start
        start_b = large.config.lexical_range(route).start
        torch.testing.assert_close(small.get_input_embeddings().weight[start_a:start_a+16], large.get_input_embeddings().weight[start_b:start_b+16], rtol=0, atol=0)
