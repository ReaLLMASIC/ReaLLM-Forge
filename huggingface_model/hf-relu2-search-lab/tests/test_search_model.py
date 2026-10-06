"""Independent head widths, exact padding algebra and HF export integration."""
import copy
import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from hf_model.configuration_search import SearchConfig
from hf_model.modeling_search import (SearchForCausalLM, pad_head_features,
                                      search_attention_reference)
from hf_model.configuration_comparison import ComparisonConfig
from hf_model.modeling_comparison import ComparisonForCausalLM
from hf_model.triton_relu2_attention import triton_relu2_attention
from hf_model.triton_relu2linear_fixed import triton_relu2linear16_attention
from experiment.optimizers import partition


def tiny_config(variant="relu2", **extra):
    args = dict(vocab_size=41, hidden_size=32, intermediate_size=48,
                num_hidden_layers=2, num_attention_heads=3,
                n_qk_head_dim=6, n_v_head_dim=10,
                max_position_embeddings=32, variant=variant,
                qk_norm_scale_init=20.0, relu2max_divisor=16.0,
                sdpa_backend="math", require_fused=False,
                relu2max_accelerator="torch", bos_token_id=1, eos_token_id=2,
                pad_token_id=0)
    args.update(extra)
    return SearchConfig(**args)


def independent_attention(q, k, v, scale, variant, mask, offset, divisor=2.0):
    scores = torch.einsum("bhmd,bhnd->bhmn", q, k) * scale
    keep = torch.arange(k.shape[-2], device=q.device)[None] <= torch.arange(q.shape[-2], device=q.device)[:, None] + offset
    keep = keep[None, None] & mask[:, None, None].bool()
    if variant == "softmax_sdpa":
        scores = scores.masked_fill(~keep, -torch.inf)
        scores = torch.where(keep.any(-1, keepdim=True), scores, torch.zeros_like(scores))
        weights = scores.softmax(-1).masked_fill(~keep, 0.0)
    else:
        x = scores.clamp_min(0)
        weights = x.square() if variant == "relu2" else torch.where(x <= 16, x.square(), 32 * x - 256)
        weights = weights.masked_fill(~keep, 0.0) / divisor
    return weights @ v


@pytest.mark.parametrize("variant", ["softmax_sdpa", "relu2", "linear16"])
@pytest.mark.parametrize("dims", [(6, 10), (10, 6), (8, 8)])
def test_padding_matches_independent_outputs_and_all_gradients(variant, dims):
    torch.manual_seed(92)
    dq, dv = dims
    values = [torch.randn(2, 3, 3, dq) * 3, torch.randn(2, 3, 11, dq) * 3,
              torch.randn(2, 3, 11, dv), torch.tensor(-0.7)]
    inputs = [x.requires_grad_() for x in values]
    clones = [x.detach().clone().requires_grad_() for x in inputs]
    mask = torch.ones(2, 11, dtype=torch.bool)
    mask[0] = False
    mask[1, ::3] = False
    qp, kp, vp = pad_head_features(*inputs[:3])
    got = search_attention_reference(qp, kp, vp, inputs[3], variant, divisor=2,
                                     attention_mask=mask, causal_offset=8)[..., :dv]
    expected = independent_attention(*clones, variant, mask, 8)
    torch.testing.assert_close(got, expected, atol=2e-4, rtol=2e-5)
    upstream = torch.randn_like(got)
    got.backward(upstream)
    expected.backward(upstream)
    for actual, ref in zip(inputs, clones):
        assert torch.isfinite(actual.grad).all()
        torch.testing.assert_close(actual.grad, ref.grad, atol=5e-4, rtol=5e-5)
    assert torch.count_nonzero(got[0]) == 0


@pytest.mark.parametrize("variant", ["softmax_sdpa", "relu2", "linear16"])
def test_nondivisible_residual_width_cache_masks_and_generation(variant):
    torch.manual_seed(11)
    model = SearchForCausalLM(tiny_config(variant)).eval()
    assert model.config.hidden_size % model.config.num_attention_heads != 0
    attn = model.model.layers[0].attn
    assert tuple(attn.q_proj.weight.shape) == (18, 32)
    assert tuple(attn.v_proj.weight.shape) == (30, 32)
    assert tuple(attn.out_proj.weight.shape) == (32, 30)
    x = torch.randint(3, 41, (2, 9))
    mask = torch.ones_like(x)
    mask[0, :2] = 0
    full = model(x, attention_mask=mask, use_cache=True)
    cache = model(x[:, :5], attention_mask=mask[:, :5], use_cache=True).past_key_values
    assert cache[0][0].shape == (2, 3, 5, 6)
    assert cache[0][1].shape == (2, 3, 5, 10)
    tail = model(x[:, 5:], attention_mask=mask, past_key_values=cache, use_cache=True)
    torch.testing.assert_close(tail.logits, full.logits[:, 5:], atol=2e-6, rtol=2e-5)
    generated = model.generate(x[:1, :3], max_new_tokens=2, do_sample=False)
    assert 3 < generated.shape[-1] <= 5
    assert attn.last_backend == ("sdpa_math_padded" if variant == "softmax_sdpa" else "torch_reference")


@pytest.mark.parametrize("variant", ["softmax_sdpa", "relu2", "linear16"])
def test_equal_head_model_preserves_baseline_math_and_gradients(variant):
    torch.manual_seed(41)
    config = tiny_config(variant, num_attention_heads=4, n_qk_head_dim=8, n_v_head_dim=8)
    search = SearchForCausalLM(config)
    old_args = config.to_dict()
    for key in ("model_type", "n_qk_head_dim", "n_v_head_dim", "padded_head_dim"):
        old_args.pop(key, None)
    baseline = ComparisonForCausalLM(ComparisonConfig(**old_args))
    baseline.load_state_dict(search.state_dict(), strict=True)
    x = torch.randint(0, 41, (1, 7))
    a, b = search(x, labels=x, use_cache=False), baseline(x, labels=x, use_cache=False)
    torch.testing.assert_close(a.logits, b.logits, atol=0, rtol=0)
    a.loss.backward()
    b.loss.backward()
    for (name, p), (other, ref) in zip(search.named_parameters(), baseline.named_parameters()):
        assert name == other
        torch.testing.assert_close(p.grad, ref.grad, atol=0, rtol=0)


def test_checkpoint_recomputation_and_optimizer_partition():
    torch.manual_seed(101)
    model = SearchForCausalLM(tiny_config("linear16", output_norm="capped"))
    checkpointed = copy.deepcopy(model)
    checkpointed.gradient_checkpointing_enable()
    x = torch.randint(0, 41, (1, 7))
    model(x, labels=x, use_cache=False).loss.backward()
    checkpointed(x, labels=x, use_cache=False).loss.backward()
    for (_, p), (_, ref) in zip(model.named_parameters(), checkpointed.named_parameters()):
        assert p.grad is not None and torch.isfinite(p.grad).all()
        torch.testing.assert_close(p.grad, ref.grad, atol=0, rtol=0)
    groups = partition(model, "muon")
    muon = {name for name, _ in groups["muon_decoder"]}
    no_decay = {name for name, _ in groups["adamw_no_decay"]}
    assert len(muon) == 6 * model.config.num_hidden_layers
    assert all(name.startswith("model.layers.") and name.endswith(".weight") for name in muon)
    assert "model.embed_tokens.weight" in {name for name, _ in groups["adamw_decay"]}
    assert "model.layers.0.attn.qk_norm_factor" in no_decay
    assert "model.layers.0.attn_output_norm.gain" in no_decay
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight


def test_fresh_process_auto_model_export(tmp_path):
    torch.manual_seed(7)
    model = SearchForCausalLM(tiny_config("linear16")).eval()
    export = tmp_path / "export"
    model.save_pretrained(export)
    x = torch.tensor([[1, 5, 9, 8]])
    torch.save({"x": x, "expected": model(x).logits.detach()}, tmp_path / "expected.pt")
    script = """
import sys, torch
from transformers import AutoConfig, AutoModelForCausalLM
root=sys.argv[1]
c=AutoConfig.from_pretrained(root+'/export', trust_remote_code=True)
assert c.hidden_size==32 and c.num_attention_heads==3
assert c.n_qk_head_dim==6 and c.n_v_head_dim==10
m=AutoModelForCausalLM.from_pretrained(root+'/export', trust_remote_code=True).eval()
d=torch.load(root+'/expected.pt', weights_only=True)
torch.testing.assert_close(m(d['x']).logits,d['expected'],atol=0,rtol=0)
assert m.get_input_embeddings().weight is m.get_output_embeddings().weight
assert m.generate(d['x'],max_new_tokens=1).shape[1]==5
"""
    env = dict(os.environ, HF_MODULES_CACHE=str(tmp_path / "fresh_modules"),
               HF_HOME=str(tmp_path / "hf_home"), HF_HUB_OFFLINE="1")
    env.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)], cwd=tmp_path,
                            env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("kwargs", [dict(n_qk_head_dim=257), dict(n_v_head_dim=0),
    dict(n_qk_head_dim=7), dict(rope_length=8, n_qk_head_dim=6),
    dict(num_key_value_heads=1), dict(linear_threshold=8), dict(dropout=.1),
    dict(use_absolute_position_embeddings=True), dict(causal_length_alpha=.5)])
def test_unsupported_shapes_and_treatments_fail_explicitly(kwargs):
    with pytest.raises(ValueError):
        tiny_config(**kwargs)


def test_partial_rope_and_original_scale_without_learned_scale():
    config = tiny_config(n_qk_head_dim=7, n_v_head_dim=11, rope_length=6,
                         use_qk_norm_scale=False)
    model = SearchForCausalLM(config)
    scales = []
    model.model.layers[0].attn.diagnostic_callback = lambda q, k, scale, past: scales.append(scale)
    model(torch.tensor([[1, 2, 3]])).logits.sum().backward()
    assert scales == pytest.approx([7 ** -0.5], rel=1e-15)
    assert model.config.padded_head_dim == 11


def test_all_ones_generation_masks_preserve_compact_sdpa(monkeypatch):
    from torch.nn import functional as functional
    original = functional.scaled_dot_product_attention
    calls = []
    def record(*args, **kwargs):
        calls.append((kwargs.get("attn_mask"), kwargs.get("is_causal")))
        return original(*args, **kwargs)
    monkeypatch.setattr(functional, "scaled_dot_product_attention", record)
    model = SearchForCausalLM(tiny_config("softmax_sdpa")).eval()
    x = torch.tensor([[1, 2, 3]])
    prefix = model(x, attention_mask=torch.ones_like(x), use_cache=True)
    assert all(mask is None and causal for mask, causal in calls)
    calls.clear()
    model(x[:, :1], attention_mask=torch.ones(1, 4),
          past_key_values=prefix.past_key_values, use_cache=True)
    assert all(mask is None and not causal for mask, causal in calls)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_flash_generation_with_tokenizer_style_all_ones_mask():
    config = tiny_config("softmax_sdpa", n_qk_head_dim=32, n_v_head_dim=64,
                         sdpa_backend="flash", require_fused=True)
    model = SearchForCausalLM(config).cuda().to(torch.bfloat16).eval()
    x = torch.tensor([[1, 5, 7]], device="cuda")
    out = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=2)
    assert 3 < out.shape[1] <= 5
    assert all(layer.attn.last_backend == "sdpa_flash_padded" for layer in model.model.layers)
    with pytest.raises(ValueError, match="key-padding masks"):
        model(x, attention_mask=torch.tensor([[0, 1, 1]], device="cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("variant", ["relu2", "linear16"])
@pytest.mark.parametrize("dims", [(32, 64), (64, 32)])
@pytest.mark.parametrize("lengths", [(7, 37), (3, 1057)])
def test_cuda_padded_fused_output_and_every_gradient(variant, dims, lengths):
    torch.manual_seed(731)
    dq, dv = dims
    m, n = lengths
    inputs = [torch.randn(1, 3, m, dq, device="cuda", dtype=torch.bfloat16).requires_grad_(),
              torch.randn(1, 3, n, dq, device="cuda", dtype=torch.bfloat16).requires_grad_(),
              torch.randn(1, 3, n, dv, device="cuda", dtype=torch.bfloat16).requires_grad_(),
              torch.tensor(-2., device="cuda", requires_grad=True)]
    refs = [x.detach().float().requires_grad_() for x in inputs]
    mask = torch.ones(1, n, device="cuda", dtype=torch.bool)
    mask[:, ::5] = False
    qp, kp, vp = pad_head_features(*inputs[:3])
    fn = triton_relu2_attention if variant == "relu2" else triton_relu2linear16_attention
    got = fn(qp, kp, vp, scale=inputs[3], divisor=256, attention_mask=mask)[..., :dv]
    expected = independent_attention(*refs, variant, mask, n-m, 256)
    upstream = torch.randn_like(got)
    got.backward(upstream)
    expected.backward(upstream.float())
    for a, b in [(got, expected)] + [(x.grad, ref.grad) for x, ref in zip(inputs, refs)]:
        assert torch.isfinite(a).all()
        error = a.float() - b.float()
        rms = b.float().square().mean().sqrt().clamp_min(1e-6)
        # BF16 tile-weight/derivative casts differ from the FP32 oracle.
        assert error.square().mean().sqrt() <= .012 * rms + 2e-5
        assert error.abs().max() <= .12 * rms + 2e-4
