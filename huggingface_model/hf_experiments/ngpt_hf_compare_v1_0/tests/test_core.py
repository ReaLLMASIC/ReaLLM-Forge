import copy
import math
import pytest
import torch
from torch.nn import functional as F
from model import Block, ModelConfig, RMSNorm, Transformer, unit
from compare_ngpt import learning_rate, optimizer_for, parser, train_one
from data import load_tokens, starts_for_step, synthetic_data, validation_document

torch.set_num_threads(2)


def model(variant="ngpt", **kwargs):
    return Transformer(ModelConfig(vocab_size=32, width=16, layers=2, heads=2,
                                   context=16, variant=variant, **kwargs))


@pytest.mark.parametrize("variant", ["gpt", "ngpt"])
def test_shapes_untied_no_bias(variant):
    m = model(variant)
    assert m(torch.randint(32, (2, 8))).shape == (2, 8, 32)
    assert m.embed.weight is not m.lm_head.weight
    assert all(mod.bias is None for mod in m.modules() if isinstance(mod, torch.nn.Linear))
    assert sum(isinstance(mod, RMSNorm) for mod in m.modules()) == (5 if variant == "gpt" else 0)


def test_initial_scales():
    m = model()
    for b in m.blocks:
        torch.testing.assert_close(b.alpha_attn * .05*math.sqrt(16), torch.full((16,), .05))
        torch.testing.assert_close(b.alpha_mlp * .05*math.sqrt(16), torch.full((16,), .05))
        torch.testing.assert_close(b.s_qk * math.sqrt(16), torch.ones((2, 8)))
        assert bool((b.s_u == 1).all() and (b.s_v == 1).all())
    torch.testing.assert_close(m.s_z * math.sqrt(16), torch.ones(32))


def test_projection_axes_and_optimizer_identity():
    m = model()
    opt = optimizer_for(m, "ngpt", .003)
    ids = {id(p) for g in opt.param_groups for p in g["params"]}
    tokens = torch.randint(32, (2, 9))
    m.next_token_loss(tokens).backward()
    opt.step()
    moments = {id(p): s["exp_avg"].clone() for p, s in opt.state.items()}
    m.project_weights_()
    assert ids == {id(p) for p in m.parameters()}
    for name, w, dim in m.constrained_weights():
        torch.testing.assert_close(w.norm(dim=dim), torch.ones_like(w.norm(dim=dim)), atol=2e-6, rtol=0)
        if "down_proj" in name:
            assert dim == 0 and w.shape == (16, 64)
    for p, state in opt.state.items():
        torch.testing.assert_close(moments[id(p)], state["exp_avg"], rtol=0, atol=0)


@pytest.mark.parametrize("variant", ["gpt", "ngpt"])
def test_future_tokens_cannot_change_past(variant):
    m = model(variant).eval()
    x = torch.randint(32, (2, 12))
    y = x.clone()
    y[:, 6:] = (y[:, 6:]+3) % 32
    torch.testing.assert_close(m(x)[:, :6], m(y)[:, :6], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(m(x[:, :6]), m(x)[:, :6], atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("variant", ["gpt", "ngpt"])
def test_chunked_loss_and_all_gradients_match_dense(variant):
    a = model(variant)
    b = copy.deepcopy(a)
    x = torch.randint(32, (2, 10))
    la = a.next_token_loss(x, chunk_tokens=7)
    logits = b(x[:, :-1])
    lb = F.cross_entropy(logits.reshape(-1, 32).float(), x[:, 1:].reshape(-1))
    torch.testing.assert_close(la, lb)
    la.backward()
    lb.backward()
    for (name, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert pa.grad is not None, name
        assert bool(torch.isfinite(pa.grad).all()), name
        torch.testing.assert_close(pa.grad, pb.grad, atol=2e-6, rtol=2e-5, msg=name)


@pytest.mark.parametrize("variant", ["gpt", "ngpt"])
def test_activation_checkpointing_gradients(variant):
    a = model(variant)
    b = model(variant, activation_checkpointing=True)
    b.load_state_dict(a.state_dict())
    x = torch.randint(32, (2, 8))
    a.next_token_loss(x).backward()
    b.next_token_loss(x).backward()
    for pa, pb in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(pa.grad, pb.grad)


def test_shared_initial_directions():
    torch.manual_seed(18)
    a = model("gpt")
    torch.manual_seed(18)
    b = model("ngpt")
    aw = {n: (p, dim) for n, p, dim in a.constrained_weights()}
    for n, p, dim in b.constrained_weights():
        torch.testing.assert_close(unit(aw[n][0], dim), p, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("variant", ["gpt", "ngpt"])
def test_attention_scaling_and_rope(monkeypatch, variant):
    m = model(variant)
    original = F.scaled_dot_product_attention
    captured = []
    def wrapped(q, k, v, **kwargs):
        captured.append((q.detach(), k.detach(), kwargs))
        return original(q, k, v, **kwargs)
    monkeypatch.setattr(F, "scaled_dot_product_attention", wrapped)
    x = torch.randint(32, (1, 8))
    m(x)
    q, k, kw = captured[0]
    assert kw["is_causal"] and kw["dropout_p"] == 0.0
    assert kw["scale"] == pytest.approx(math.sqrt(8) if variant == "ngpt" else 1/math.sqrt(8))
    if variant == "ngpt":
        torch.testing.assert_close(q.norm(dim=-1), torch.ones_like(q[..., 0]), atol=2e-6, rtol=0)
    block = m.blocks[0]
    h = m.embed(x)
    if variant == "gpt":
        h = block.attn_norm(h)
    raw = block.q_proj(h).view(1, 8, 2, 8).transpose(1, 2)
    pos = torch.arange(8).float()[:, None]
    angles = pos * m.inv_freq[None]
    # Position-zero is unrotated; norm of correct RoPE is preserved at every position.
    if variant == "gpt":
        torch.testing.assert_close(q.norm(dim=-1), raw.norm(dim=-1), atol=2e-6, rtol=2e-6)
    expected0 = unit(raw[:, :, 0]) if variant == "ngpt" else raw[:, :, 0]
    torch.testing.assert_close(q[:, :, 0], expected0)
    a, b = raw[..., 0::2], raw[..., 1::2]
    expected = torch.stack((a*angles.cos()-b*angles.sin(), a*angles.sin()+b*angles.cos()), -1).flatten(-2)
    if variant == "ngpt":
        expected = unit(expected)
    torch.testing.assert_close(q, expected, atol=2e-6, rtol=2e-6)


def test_ngpt_hidden_norms():
    m = model()
    states = []
    hooks = [b.register_forward_hook(lambda module, inp, out: states.append(out.detach())) for b in m.blocks]
    m(torch.randint(32, (2, 8)))
    for h in states:
        torch.testing.assert_close(h.norm(dim=-1), torch.ones_like(h[..., 0]), atol=2e-6, rtol=0)
    for h in hooks:
        h.remove()


def test_schedules_and_optimizers():
    assert learning_rate(1, 100, .003, 0) == .003
    assert learning_rate(1, 100, .003, 20) == .003/20
    assert learning_rate(20, 100, .003, 20) == .003
    assert learning_rate(100, 100, .003, 0) == 0
    assert learning_rate(100, 100, .003, 20) == 0
    for variant in ("gpt", "ngpt"):
        opt = optimizer_for(model(variant), variant, .003)
        assert opt.defaults["betas"] == (.9, .95)
        assert opt.defaults["eps"] == 1e-8
        assert opt.param_groups[0]["weight_decay"] == (.1 if variant == "gpt" else 0.)
        assert opt.param_groups[1]["weight_decay"] == 0.


def test_data_routing_and_batch_repeatability():
    text = "An exact duplicate should always be routed into the same split."
    assert validation_document(text, 0) == validation_document(text, 0)
    a = starts_for_step(1000, 8, 16, 1234, 5, 0)
    b = starts_for_step(1000, 8, 16, 1234, 5, 0)
    assert (a == b).all()
    assert not (a == starts_for_step(1000, 8, 16, 1234, 6, 0)).all()


@pytest.mark.parametrize("variant", ["gpt", "ngpt"])
def test_reference_bf16_storage_cpu_autocast(variant):
    m = model(variant, weight_storage="reference_bf16")
    assert m.blocks[0].q_proj.weight.dtype == torch.bfloat16
    assert m.embed.weight.dtype == m.lm_head.weight.dtype == torch.float32
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = m.next_token_loss(torch.randint(32, (2, 9)), chunk_tokens=5)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in m.parameters())
    m.project_weights_()


@pytest.mark.parametrize("variant", ["gpt", "ngpt"])
def test_exact_cpu_checkpoint_resume(tmp_path, variant):
    d = tmp_path/"data"
    synthetic_data(d)
    meta, arrays = load_tokens(d, 8)
    common = ["train", "--core-only", "--device", "cpu", "--precision", "fp32", "--width", "16",
              "--layers", "1", "--heads", "2", "--context", "8", "--steps", "6",
              "--warmup-gpt", "2", "--batch-size", "2", "--accumulation", "2", "--eval-every", "2",
              "--eval-batches", "1", "--save-every", "2", "--loss-chunk", "8", "--data", str(d)]
    full = parser().parse_args(common+["--out", str(tmp_path/"full")])
    train_one(full, variant, 0, meta, arrays, torch.device("cpu"))
    part = parser().parse_args(common+["--out", str(tmp_path/"part"), "--stop-after-steps", "2"])
    train_one(part, variant, 0, meta, arrays, torch.device("cpu"))
    part.resume, part.stop_after_steps = True, None
    train_one(part, variant, 0, meta, arrays, torch.device("cpu"))
    a = torch.load(tmp_path/f"full/{variant}_seed0/last.pt", weights_only=True)
    b = torch.load(tmp_path/f"part/{variant}_seed0/last.pt", weights_only=True)
    for name in a["model"]:
        torch.testing.assert_close(a["model"][name], b["model"][name], atol=0, rtol=0)
    for ra, rb in zip(a["history"], b["history"]):
        for key in ("step", "train_loss", "val_loss", "train_probe_loss", "lr", "tokens"):
            assert ra.get(key) == rb.get(key)
