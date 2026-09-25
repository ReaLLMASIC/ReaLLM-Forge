"""Full-parameter CPT with FP32 AdamW and memory-bounded BF16 forward passes."""
from __future__ import annotations

import contextlib
import hashlib
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM

SUPPORTED = {"qwen2", "llama", "mistral", "gpt_neox", "gpt2"}


def seed_all(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False


def autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if torch.device(device).type == "cuda" else contextlib.nullcontext()


def setup_device(c):
    torch.set_num_threads(c["cpu_threads"])
    d = torch.device(c["device"])
    if d.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable. Use the offline CPU smoke test, or run this preset on your H100.")
        if d.index is None:
            d = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(d)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("This preset requires a BF16-capable GPU")
        total = torch.cuda.get_device_properties(d).total_memory
        cap = min(c["memory_limit_gib"] * 2**30, total * c["memory_fraction"])
        torch.cuda.set_per_process_memory_fraction(cap / total, d)
        free, _ = torch.cuda.mem_get_info(d)
        print(f"GPU: {torch.cuda.get_device_name(d)}; total={total / 2**30:.2f} GiB; "
              f"free={free / 2**30:.2f} GiB; allocator ceiling={cap / 2**30:.2f} GiB", flush=True)
        return d, min(cap, free * 0.95)
    return d, float("inf")


def load_model(path, c, revision=None):
    # FP32 parameters => FP32 AdamW states, no silently low-precision optimizer moments.
    m = AutoModelForCausalLM.from_pretrained(
        path, revision=revision, dtype=torch.float32,
        attn_implementation=c["attn_implementation"], trust_remote_code=False,
    )
    if m.config.model_type not in SUPPORTED:
        raise ValueError(f"Unsupported model_type={m.config.model_type}. Supported: {sorted(SUPPORTED)}. "
                         "Other architectures may apply logit scaling/softcapping outside their base model.")
    if not isinstance(m.get_output_embeddings(), torch.nn.Linear):
        raise ValueError("Expected a linear output head")
    m.config.use_cache = False
    if c["disable_dropout"]:
        for module in m.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
            if hasattr(module, "attention_dropout"):
                module.attention_dropout = 0.0
    return m


def is_tied(m):
    return m.get_input_embeddings().weight.data_ptr() == m.get_output_embeddings().weight.data_ptr()


def configure_mode(m, mode):
    for p in m.parameters():
        p.requires_grad_(True)
    tied = is_tied(m)
    if mode.startswith("tied_"):
        if not tied:
            raise ValueError("Tied controls require an anchor with genuinely tied embeddings")
    elif mode != "anchor":
        if tied:
            m.get_output_embeddings().weight = torch.nn.Parameter(m.get_output_embeddings().weight.detach().clone())
        m.config.tie_word_embeddings = False
    if mode in {"freeze_head", "freeze_both", "freeze_head_norm", "tied_freeze_both"}:
        m.get_output_embeddings().requires_grad_(False)
    if mode in {"freeze_embed", "freeze_both", "tied_freeze_both"}:
        m.get_input_embeddings().requires_grad_(False)
    if mode == "freeze_head_norm":
        attr = {"gpt2": "ln_f", "gpt_neox": "final_layer_norm"}.get(m.config.model_type, "norm")
        getattr(m.base_model, attr).requires_grad_(False)
    return {"anchor_was_tied": tied, "now_tied": is_tied(m)}


def memory_estimate(m, c, mode):
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    head = sum(p.numel() for p in m.get_output_embeddings().parameters())
    # weights + trainable gradients and two FP32 moments. Add BF16 weight casts,
    # one original head for diagnostics, activation/workspace safety allowance.
    persistent = 4 * total + 12 * trainable
    estimated = persistent + 2 * total + (0 if mode == "anchor" else 4 * head) + c["activation_reserve_gib"] * 2**30
    return {"parameters": total, "trainable_parameters": trainable,
            "persistent_gib": persistent / 2**30,
            "estimate_with_reserve_gib": estimated / 2**30,
            "note": "Conservative estimate, not a measured peak; run preflight first."}


def place_model(m, c, device, cap, mode):
    estimate = memory_estimate(m, c, mode)
    print(f"Memory precheck ({mode}): {estimate}", flush=True)
    if estimate["estimate_with_reserve_gib"] * 2**30 > cap:
        raise RuntimeError("Estimated training memory exceeds available budget. Choose a smaller model. "
                           "Reducing sequence length cannot remove AdamW's parameter-state memory.")
    m.to(device)
    if c["gradient_checkpointing"]:
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return estimate


def hidden(m, batch):
    return m.base_model(input_ids=batch[:, :-1], use_cache=False, return_dict=True).last_hidden_state


def ce_sum(h, w, b, target):
    logits = F.linear(h, w, b)
    return F.cross_entropy(logits.float(), target, reduction="sum")


def chunked_loss(m, batch, chunk_size):
    """Exact full-vocabulary CE; checkpoint chunks so their logits are NOT all retained."""
    h = hidden(m, batch).reshape(-1, m.get_output_embeddings().in_features)
    target = batch[:, 1:].reshape(-1)
    head = m.get_output_embeddings()
    total = h.new_zeros((), dtype=torch.float32)
    for start in range(0, len(target), chunk_size):
        end = start + chunk_size
        if torch.is_grad_enabled():
            value = checkpoint(ce_sum, h[start:end], head.weight, head.bias, target[start:end],
                               use_reentrant=False)
        else:
            value = ce_sum(h[start:end], head.weight, head.bias, target[start:end])
        total = total + value
    return total / len(target)


def make_optimizer(m, c, lr, mode):
    head_ids = {id(p) for p in m.get_output_embeddings().parameters()}
    groups = {}
    for name, p in m.named_parameters():
        if not p.requires_grad:
            continue
        multiplier = c["slow_head_multiplier"] if mode == "slow_head" and id(p) in head_ids else 1.0
        wd = c["weight_decay"] if p.ndim >= 2 else 0.0
        groups.setdefault((multiplier, wd), []).append(p)
    return torch.optim.AdamW([
        {"params": params, "lr": lr * mult, "base_lr": lr * mult, "weight_decay": wd}
        for (mult, wd), params in groups.items()
    ], betas=(0.9, 0.95), eps=1e-8, foreach=False)


def lr_factor(step, total, c):
    warm = int(total * c["warmup_fraction"])
    if warm and step <= warm:
        return step / warm
    progress = min(1.0, max(0.0, (step - warm) / max(1, total - warm)))
    return c["min_lr_fraction"] + (1 - c["min_lr_fraction"]) * 0.5 * (1 + math.cos(math.pi * progress))


def tensor_hash(t):
    h = hashlib.sha256()
    flat = t.detach().reshape(-1)
    for start in range(0, flat.numel(), 1 << 20):
        h.update(flat[start:start + (1 << 20)].cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def frozen_hashes(m):
    # All frozen parameters including the optional final norm. Checks the entire tensor.
    return {n: tensor_hash(p) for n, p in m.named_parameters() if not p.requires_grad}


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(s):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def peak_memory(device):
    if device.type != "cuda":
        return {"peak_allocated_gib": 0.0, "peak_reserved_gib": 0.0}
    return {"peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30}
