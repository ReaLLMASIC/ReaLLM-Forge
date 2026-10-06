#!/usr/bin/env python3
"""
Recurrent Attention-Residual Module Mesh
========================================

A single-file Hugging Face / PyTorch research implementation of a synchronous
module mesh.  At iteration zero every module reads the token embedding.  At
later iterations every attention and FFN module independently aggregates

    [token embedding, all attention outputs, all FFN outputs]

from the previous iteration, then all modules execute concurrently.  The
default aggregator is Attention Residuals-style depth/module attention: each
destination has a learned pseudo-query, keys are RMS-normalized source states,
and values are the unnormalized source states.  A final pseudo-query readout
aggregates the embedding and all terminal module outputs before the tied LM
head.

The implementation packs all attention weights and all FFN weights into
batched tensors.  Attention modules are folded into the batch dimension for
one PyTorch SDPA call, allowing fused attention kernels on NVIDIA GPUs.  The
default ``a100-80gb`` profile is a 305M-parameter, BF16/TF32 configuration
with strict SM80, 80 GB HBM, and Flash-SDPA checks.  Strict profiles are also
provided for H100 SXM/PCIe 80 GB and RTX 4090 24 GB devices, alongside a
conservative generic-CUDA profile.  It supports:

  * arbitrary counts of attention and FFN modules;
  * arbitrary recurrent iterations;
  * tied or untied weights across iterations;
  * destination-specific or shared AttnRes, learned-static, uniform, and
    identity/no-cross routing;
  * single- or multi-head routing over the module axis;
  * BF16, torch.compile, fused AdamW, Accelerate/DDP, and checkpoint chunks;
  * training, synthetic benchmarking, a correctness smoke test, and grid
    sweeps from this one file.

Recommended environment (API target):

    pip install "torch>=2.5" "transformers==5.16.1" \
        "datasets>=4.0" "accelerate>=1.10" safetensors

Quick checks:

    python recurrent_attnres_mesh.py --mode smoke
    python recurrent_attnres_mesh.py --mode inspect
    python recurrent_attnres_mesh.py --mode benchmark \
        --hardware-profile a100-80gb

One A100 80 GB training run (32,768 tokens/update):

    accelerate launch recurrent_attnres_mesh.py --mode train \
        --dataset roneneldan/TinyStories --tokenizer gpt2 \
        --hardware-profile a100-80gb

The profile resolves to D=2048, 32 heads, SwiGLU width 5504, four attention
modules, four FFN modules, six recurrent steps, sequence length 2048,
microbatch 8, and two gradient-accumulation steps.  Explicit command-line
values and sweep run-config values always override profile defaults.  Use
``--hardware-profile cuda-generic`` on an otherwise unsupported NVIDIA GPU,
or ``--hardware-profile portable`` for CPU checks.

Sequential grid sweep on one A100:

    python recurrent_attnres_mesh.py --mode sweep \
        --hardware-profile a100-80gb \
        --sweep-attention-modules 1,2,4 \
        --sweep-ffn-modules 1,2,4 \
        --sweep-iterations 1,2,4,6,8 \
        --sweep-router-types attnres \
        --max-steps 1000 --output-dir runs/mesh_sweep

The sweep deliberately logs parameter count, analytical FLOPs, tokens/s, and
peak memory because changing module count changes both parameters and compute,
whereas changing recurrent iterations with tied weights changes compute but
not the heavy branch parameter count.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import itertools
import json
import math
import os
import random
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    from torch.utils.data import DataLoader, IterableDataset
except ImportError as exc:
    raise SystemExit(
        "PyTorch is required. Install the dependencies shown in the module "
        "docstring before running this script."
    ) from exc


A100_80GB_PROFILE: dict[str, Any] = {
    # A large but conservative screening point for one 80 GB A100.  The
    # module and iteration values remain ordinary CLI values, so every one can
    # be independently overridden by an ablation or sweep child.
    "hidden_size": 2_048,
    "num_attention_modules": 4,
    "num_ffn_modules": 4,
    "num_iterations": 6,
    "num_attention_heads": 32,
    "intermediate_size": 5_504,
    "max_position_embeddings": 4_096,
    "sequence_length": 2_048,
    "num_workers": 8,
    "prefetch_factor": 4,
    "persistent_workers": True,
    "micro_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "mixed_precision": "bf16",
    "tf32": True,
    "compile_model": True,
    "compile_mode": "default",
    "gradient_checkpointing": True,
    "checkpoint_chunk_steps": 2,
    "sdpa_backend": "flash",
    "require_fused_sdpa": True,
    "fused_adamw": True,
    "strict_hardware_profile": True,
    "allocator_expandable_segments": True,
    "target_peak_memory_fraction": 0.90,
    "memory_reserve_gb": 8.0,
    "throughput_warmup_steps": 1,
    "bf16_peak_tflops": 312.0,
}

# H100 profiles deliberately retain the A100 scientific baseline and memory
# policy so an A100/H100 comparison changes hardware rather than architecture.
# The profile name selects the appropriate dense BF16 (non-sparse, non-FP8)
# denominator and fused SDPA policy.
H100_SXM_80GB_PROFILE: dict[str, Any] = {
    **A100_80GB_PROFILE,
    "sdpa_backend": "fused",
    "bf16_peak_tflops": 989.0,
}

H100_PCIE_80GB_PROFILE: dict[str, Any] = {
    **A100_80GB_PROFILE,
    "sdpa_backend": "fused",
    "bf16_peak_tflops": 756.0,
}

RTX4090_24GB_PROFILE: dict[str, Any] = {
    # Sized to keep the configured working budget at about 20 GiB while still
    # exercising the same four-attention/four-FFN mesh topology.
    "hidden_size": 1_024,
    "num_attention_modules": 4,
    "num_ffn_modules": 4,
    "num_iterations": 4,
    "num_attention_heads": 16,
    "intermediate_size": 2_752,
    "max_position_embeddings": 2_048,
    "sequence_length": 1_024,
    "num_workers": 4,
    "prefetch_factor": 2,
    "persistent_workers": True,
    "micro_batch_size": 4,
    "gradient_accumulation_steps": 8,
    "mixed_precision": "bf16",
    "tf32": True,
    "compile_model": True,
    "compile_mode": "default",
    "gradient_checkpointing": True,
    "checkpoint_chunk_steps": 2,
    "sdpa_backend": "flash",
    "require_fused_sdpa": True,
    "fused_adamw": True,
    "strict_hardware_profile": True,
    "allocator_expandable_segments": True,
    "target_peak_memory_fraction": 0.85,
    "memory_reserve_gb": 3.0,
    "throughput_warmup_steps": 1,
    "bf16_peak_tflops": 165.2,
}

CUDA_GENERIC_PROFILE: dict[str, Any] = {
    # Broadly compatible fallback: FP16 and automatic SDPA allow PyTorch to
    # choose an available kernel. Override dimensions/batch size and provide
    # --bf16-peak-tflops when characterizing a known BF16-capable GPU.
    "hidden_size": 768,
    "num_attention_modules": 2,
    "num_ffn_modules": 2,
    "num_iterations": 4,
    "num_attention_heads": 12,
    "intermediate_size": 2_048,
    "max_position_embeddings": 2_048,
    "sequence_length": 1_024,
    "num_workers": 4,
    "prefetch_factor": 2,
    "persistent_workers": True,
    "micro_batch_size": 2,
    "gradient_accumulation_steps": 16,
    "mixed_precision": "fp16",
    "tf32": True,
    "compile_model": True,
    "compile_mode": "default",
    "gradient_checkpointing": True,
    "checkpoint_chunk_steps": 2,
    "sdpa_backend": "auto",
    "require_fused_sdpa": False,
    "fused_adamw": True,
    "strict_hardware_profile": True,
    "allocator_expandable_segments": True,
    "target_peak_memory_fraction": 0.80,
    "memory_reserve_gb": 2.0,
    "throughput_warmup_steps": 1,
    "bf16_peak_tflops": 0.0,
}

PORTABLE_PROFILE: dict[str, Any] = {
    # Intentionally small and accelerator-free so inspect/benchmark checks are
    # useful on laptops and CI runners without CUDA.
    "hidden_size": 256,
    "num_attention_modules": 1,
    "num_ffn_modules": 1,
    "num_iterations": 2,
    "num_attention_heads": 8,
    "intermediate_size": 688,
    "max_position_embeddings": 512,
    "sequence_length": 256,
    "num_workers": 0,
    "prefetch_factor": 2,
    "persistent_workers": False,
    "micro_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "mixed_precision": "no",
    "tf32": False,
    "compile_model": False,
    "compile_mode": "default",
    "gradient_checkpointing": False,
    "checkpoint_chunk_steps": 1,
    "sdpa_backend": "auto",
    "require_fused_sdpa": False,
    "fused_adamw": False,
    "strict_hardware_profile": False,
    "allocator_expandable_segments": False,
    "target_peak_memory_fraction": 0.90,
    "memory_reserve_gb": 2.0,
    "throughput_warmup_steps": 0,
    "bf16_peak_tflops": 0.0,
}

HARDWARE_PROFILES = {
    "a100-80gb": A100_80GB_PROFILE,
    "h100-sxm-80gb": H100_SXM_80GB_PROFILE,
    "h100-pcie-80gb": H100_PCIE_80GB_PROFILE,
    "rtx4090-24gb": RTX4090_24GB_PROFILE,
    "cuda-generic": CUDA_GENERIC_PROFILE,
    "portable": PORTABLE_PROFILE,
}

# CUDA device properties do not expose the H100 PCIe-versus-SXM form factor
# portably, so both H100 contracts validate the SKU and SM version; selecting
# the form-factor profile also selects its dense-BF16 denominator.
GPU_PROFILE_CONTRACTS: dict[str, dict[str, Any]] = {
    "a100-80gb": {
        "name_token": "A100",
        "compute_capability": (8, 0),
        "min_hbm_gb": 75.0,
        "require_bf16": True,
    },
    "h100-sxm-80gb": {
        "name_token": "H100",
        "compute_capability": (9, 0),
        "min_hbm_gb": 75.0,
        "require_bf16": True,
    },
    "h100-pcie-80gb": {
        "name_token": "H100",
        "compute_capability": (9, 0),
        "min_hbm_gb": 75.0,
        "require_bf16": True,
    },
    "rtx4090-24gb": {
        "name_token": "RTX 4090",
        "compute_capability": (8, 9),
        "min_hbm_gb": 20.0,
        "require_bf16": True,
    },
    "cuda-generic": {
        "name_token": None,
        "compute_capability": None,
        "min_hbm_gb": 0.0,
        "require_bf16": False,
    },
}
CUDA_OOM_EXIT_CODE = 42

try:
    import transformers
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        GenerationMixin,
        PretrainedConfig,
        PreTrainedModel,
        get_cosine_schedule_with_warmup,
    )
    from transformers.utils import ModelOutput
except ImportError as exc:
    raise SystemExit(
        "Hugging Face Transformers is required. Install transformers==5.16.1."
    ) from exc


# ---------------------------------------------------------------------------
# Hugging Face configuration and output
# ---------------------------------------------------------------------------


class RecurrentAttnResConfig(PretrainedConfig):
    """Configuration for the recurrent attention-residual module mesh."""

    model_type = "recurrent_attnres_mesh"

    def __init__(
        self,
        vocab_size: int = 50_304,
        valid_vocab_size: Optional[int] = None,
        hidden_size: int = 768,
        num_attention_modules: int = 2,
        num_ffn_modules: int = 2,
        num_iterations: int = 4,
        num_attention_heads: int = 12,
        intermediate_size: int = 2_048,
        max_position_embeddings: int = 2_048,
        rope_theta: float = 10_000.0,
        router_type: str = "attnres",
        router_heads: int = 1,
        readout_type: str = "attnres",
        input_injection: str = "every",
        share_iteration_weights: bool = True,
        use_step_embeddings: bool = False,
        branch_output: str = "delta",
        exclude_self: bool = False,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        embedding_dropout: float = 0.0,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        sdpa_backend: str = "auto",
        require_fused_sdpa: Optional[bool] = None,
        require_flash: Optional[bool] = None,
        checkpoint_chunk_steps: int = 1,
        tie_word_embeddings: bool = True,
        bos_token_id: Optional[int] = 1,
        eos_token_id: Optional[int] = 2,
        pad_token_id: Optional[int] = 0,
        **kwargs: Any,
    ) -> None:
        # These are derived properties.  Ignore serialized values from older
        # checkpoints rather than allowing structural overrides to make them
        # stale.
        kwargs.pop("num_modules", None)
        kwargs.pop("num_hidden_layers", None)
        self.vocab_size = vocab_size
        self.valid_vocab_size = (
            vocab_size if valid_vocab_size is None else valid_vocab_size
        )
        self.hidden_size = hidden_size
        self.num_attention_modules = num_attention_modules
        self.num_ffn_modules = num_ffn_modules
        self.num_iterations = num_iterations
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.router_type = router_type
        self.router_heads = router_heads
        self.readout_type = readout_type
        self.input_injection = input_injection
        self.share_iteration_weights = share_iteration_weights
        self.use_step_embeddings = use_step_embeddings
        self.branch_output = branch_output
        self.exclude_self = exclude_self
        self.attention_dropout = attention_dropout
        self.residual_dropout = residual_dropout
        self.embedding_dropout = embedding_dropout
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        # ``require_flash`` is the serialized spelling used by the original
        # A100-only release.  Loading such a checkpoint still forces Flash;
        # new checkpoints record the more general backend policy explicitly.
        if require_fused_sdpa is None:
            require_fused_sdpa = bool(require_flash)
        if require_flash and sdpa_backend == "auto":
            sdpa_backend = "flash"
        elif require_fused_sdpa and sdpa_backend == "auto":
            sdpa_backend = "fused"
        if sdpa_backend not in {"auto", "fused", "cudnn", "flash"}:
            raise ValueError(f"Unsupported SDPA backend policy: {sdpa_backend!r}")
        self.sdpa_backend = sdpa_backend
        self.require_fused_sdpa = bool(require_fused_sdpa)
        # Compatibility field for older tooling. It describes a flash-only
        # policy, while require_fused_sdpa also covers the H100 fused policy.
        self.require_flash = bool(
            self.require_fused_sdpa and sdpa_backend == "flash"
        )
        self.checkpoint_chunk_steps = checkpoint_chunk_steps
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )
        self.use_cache = False
        self.is_decoder = True
        self.is_encoder_decoder = False
        self._validate_structure()

    @property
    def num_modules(self) -> int:
        return self.num_attention_modules + self.num_ffn_modules

    @property
    def num_hidden_layers(self) -> int:
        # Useful metadata for HF tooling even though generation deliberately
        # recomputes prefixes instead of maintaining a KV cache.
        return self.num_iterations * self.num_attention_modules

    def _validate_structure(self) -> None:
        if self.num_attention_modules < 0 or self.num_ffn_modules < 0:
            raise ValueError("Module counts cannot be negative.")
        if self.num_modules < 1:
            raise ValueError("At least one attention or FFN module is required.")
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be positive.")
        if not 0 < self.valid_vocab_size <= self.vocab_size:
            raise ValueError(
                "valid_vocab_size must be positive and no larger than vocab_size."
            )
        if self.num_attention_heads < 1:
            raise ValueError("num_attention_heads must be positive.")
        if self.router_heads < 1:
            raise ValueError("router_heads must be positive.")
        if self.hidden_size % max(self.num_attention_heads, 1) != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads.")
        if self.hidden_size % self.router_heads != 0:
            raise ValueError("hidden_size must be divisible by router_heads.")
        if self.num_iterations < 1:
            raise ValueError("num_iterations must be at least 1.")
        if self.checkpoint_chunk_steps < 1:
            raise ValueError("checkpoint_chunk_steps must be at least 1.")
        if self.router_type not in {
            "attnres",
            "shared_attnres",
            "static",
            "uniform",
            "identity",
        }:
            raise ValueError(
                "router_type must be attnres, shared_attnres, static, "
                "uniform, or identity."
            )
        if self.readout_type not in {"attnres", "static", "uniform"}:
            raise ValueError("readout_type must be attnres, static, or uniform.")
        if self.input_injection not in {"every", "initial_only"}:
            raise ValueError("input_injection must be every or initial_only.")
        if self.branch_output not in {"delta", "residual"}:
            raise ValueError("branch_output must be delta or residual.")
        if self.exclude_self and self.router_type == "identity":
            raise ValueError(
                "exclude_self is incompatible with identity routing, whose "
                "definition is to retain each module's own state."
            )
        if (
            self.exclude_self
            and self.input_injection == "initial_only"
            and self.num_modules == 1
        ):
            raise ValueError(
                "exclude_self leaves no routing source when there is one module "
                "and input_injection='initial_only'."
            )
        if self.intermediate_size < 1 and self.num_ffn_modules:
            raise ValueError("intermediate_size must be positive.")
        if self.max_position_embeddings < 1:
            raise ValueError("max_position_embeddings must be positive.")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive.")
        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be positive.")
        for name, probability in (
            ("attention_dropout", self.attention_dropout),
            ("residual_dropout", self.residual_dropout),
            ("embedding_dropout", self.embedding_dropout),
        ):
            if not 0.0 <= probability < 1.0:
                raise ValueError(f"{name} must be in [0, 1).")

    @classmethod
    def from_dict(
        cls, config_dict: dict[str, Any], **kwargs: Any
    ) -> Any:
        result = super().from_dict(config_dict, **kwargs)
        config = result[0] if isinstance(result, tuple) else result
        config._validate_structure()
        return result


@dataclass
class RecurrentMeshCausalLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Any] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    router_entropies: Optional[tuple[torch.FloatTensor, ...]] = None


# ---------------------------------------------------------------------------
# Numerically simple, compiler-friendly building blocks
# ---------------------------------------------------------------------------


def rms_norm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    eps: float,
) -> torch.Tensor:
    """RMS normalization with accumulation in fp32."""
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    y = x * torch.rsqrt(variance + eps).to(dtype=x.dtype)
    return y if weight is None else y * weight


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.eps)


class BatchedRMSNorm(nn.Module):
    """A distinct affine RMSNorm for each packed module."""

    def __init__(self, num_modules: int, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_modules, hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [M, B, L, D]
        return rms_norm(x, self.weight[:, None, None, :], self.eps)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE head_dim must be even.")
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        # Persistent is important for Transformers 5's meta-device
        # from_pretrained path: custom nonpersistent buffers are otherwise not
        # materialized by the generic loader.
        self.register_buffer("inv_freq", inv_freq, persistent=True)

    def forward(
        self, position_ids: torch.LongTensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = position_ids.float().unsqueeze(-1) * self.inv_freq.float()
        return freqs.cos().to(dtype=dtype), freqs.sin().to(dtype=dtype)


def apply_rotary(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    # x: [M, B, H, L, Dh]; cos/sin: [B, L, Dh/2]
    even, odd = x[..., 0::2], x[..., 1::2]
    cos = cos[None, :, None, :, :]
    sin = sin[None, :, None, :, :]
    rotated_even = even * cos - odd * sin
    rotated_odd = even * sin + odd * cos
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


def sdpa_backend_context(
    policy: str,
) -> contextlib.AbstractContextManager[Any]:
    """Restrict SDPA to the requested policy without enabling math fallback."""
    if policy == "auto":
        return contextlib.nullcontext()
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "This PyTorch build cannot explicitly select SDPA backends."
        ) from exc

    if policy == "flash":
        backends = [SDPBackend.FLASH_ATTENTION]
    elif policy == "cudnn":
        try:
            backends = [SDPBackend.CUDNN_ATTENTION]
        except AttributeError as exc:
            raise RuntimeError(
                "This PyTorch build does not expose the cuDNN SDPA backend."
            ) from exc
    elif policy == "fused":
        try:
            backends = [
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.FLASH_ATTENTION,
            ]
        except AttributeError as exc:
            raise RuntimeError(
                "The fused SDPA policy needs both cuDNN and Flash backend enums."
            ) from exc
    else:  # Configuration and argparse validate this before model execution.
        raise ValueError(f"Unsupported SDPA backend policy: {policy!r}")
    return sdpa_kernel(backends)


class BatchedCausalSelfAttention(nn.Module):
    """Distinct attention modules executed as packed batched GEMMs + one SDPA."""

    def __init__(self, config: RecurrentAttnResConfig) -> None:
        super().__init__()
        self.num_modules = config.num_attention_modules
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.attention_dropout = config.attention_dropout
        self.residual_dropout = config.residual_dropout
        self.sdpa_backend = config.sdpa_backend
        self.require_fused_sdpa = config.require_fused_sdpa
        self.require_flash = config.require_flash  # legacy inspection alias
        self.rope = RotaryEmbedding(self.head_dim, config.rope_theta)

        self.qkv_weight = nn.Parameter(
            torch.empty(self.num_modules, self.hidden_size, 3 * self.hidden_size)
        )
        self.out_weight = nn.Parameter(
            torch.empty(self.num_modules, self.hidden_size, self.hidden_size)
        )
        nn.init.normal_(self.qkv_weight, std=config.initializer_range)
        output_std = config.initializer_range / math.sqrt(
            max(2 * config.num_iterations, 1)
        )
        nn.init.normal_(self.out_weight, std=output_std)

    def _sdpa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        dropout_p = self.attention_dropout if self.training else 0.0
        if self.sdpa_backend != "auto":
            if not query.is_cuda:
                raise RuntimeError(
                    f"SDPA backend policy {self.sdpa_backend!r} requires CUDA."
                )
        context = sdpa_backend_context(self.sdpa_backend)

        if self.require_fused_sdpa and attention_mask is not None:
            raise RuntimeError(
                "A required fused-SDPA policy currently needs unpadded, "
                "fixed-length batches; a custom padding mask was supplied."
            )

        with context:
            return F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=dropout_p,
                is_causal=attention_mask is None,
            )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: [A, B, L, D]
        a, batch, length, hidden = x.shape
        if a != self.num_modules:
            raise ValueError(f"Expected {self.num_modules} attention modules, got {a}.")

        flat = x.reshape(a, batch * length, hidden)
        qkv = torch.bmm(flat, self.qkv_weight).view(
            a, batch, length, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=3)
        query = query.permute(0, 1, 3, 2, 4)
        key = key.permute(0, 1, 3, 2, 4)
        value = value.permute(0, 1, 3, 2, 4)

        cos, sin = self.rope(position_ids, query.dtype)
        query = apply_rotary(query, cos, sin)
        key = apply_rotary(key, cos, sin)

        query = query.reshape(a * batch, self.num_heads, length, self.head_dim)
        key = key.reshape(a * batch, self.num_heads, length, self.head_dim)
        value = value.reshape(a * batch, self.num_heads, length, self.head_dim)
        query, key, value = query.contiguous(), key.contiguous(), value.contiguous()

        sdpa_mask = None
        if attention_mask is not None:
            valid_keys = attention_mask.to(dtype=torch.bool)
            causal = torch.ones(
                length, length, dtype=torch.bool, device=x.device
            ).tril()
            allowed = causal[None, None, :, :] & valid_keys[:, None, None, :]
            # Queries are flattened in module-major order: (a0,b0),
            # (a0,b1), ..., (a1,b0), ... .  Preserve that ordering here.
            sdpa_mask = (
                allowed[None, :, :, :, :]
                .expand(a, batch, 1, length, length)
                .reshape(a * batch, 1, length, length)
            )

        attended = self._sdpa(query, key, value, sdpa_mask)
        attended = (
            attended.view(a, batch, self.num_heads, length, self.head_dim)
            .permute(0, 1, 3, 2, 4)
            .reshape(a, batch * length, hidden)
        )
        output = torch.bmm(attended, self.out_weight).view(a, batch, length, hidden)
        return F.dropout(output, p=self.residual_dropout, training=self.training)


class BatchedSwiGLU(nn.Module):
    """Distinct SwiGLU modules executed with two packed batched GEMMs."""

    def __init__(self, config: RecurrentAttnResConfig) -> None:
        super().__init__()
        self.num_modules = config.num_ffn_modules
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.residual_dropout = config.residual_dropout
        self.gate_up_weight = nn.Parameter(
            torch.empty(
                self.num_modules,
                self.hidden_size,
                2 * self.intermediate_size,
            )
        )
        self.down_weight = nn.Parameter(
            torch.empty(
                self.num_modules,
                self.intermediate_size,
                self.hidden_size,
            )
        )
        nn.init.normal_(self.gate_up_weight, std=config.initializer_range)
        output_std = config.initializer_range / math.sqrt(
            max(2 * config.num_iterations, 1)
        )
        nn.init.normal_(self.down_weight, std=output_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [F, B, L, D]
        ffn, batch, length, hidden = x.shape
        if ffn != self.num_modules:
            raise ValueError(f"Expected {self.num_modules} FFN modules, got {ffn}.")
        flat = x.reshape(ffn, batch * length, hidden)
        gate_up = torch.bmm(flat, self.gate_up_weight)
        gate, up = gate_up.chunk(2, dim=-1)
        activated = F.silu(gate) * up
        output = torch.bmm(activated, self.down_weight).view(
            ffn, batch, length, hidden
        )
        return F.dropout(output, p=self.residual_dropout, training=self.training)


class CrossModuleMixer(nn.Module):
    """
    Mix an embedding and module-indexed states independently at every token.

    Dynamic AttnRes routing uses a learned pseudo-query per destination and
    RMS-normalized source keys.  The query is initialized to zero, exactly
    yielding uniform source weights at initialization.
    """

    def __init__(
        self,
        config: RecurrentAttnResConfig,
        num_destinations: int,
        mixer_type: str,
        allow_identity: bool,
    ) -> None:
        super().__init__()
        self.num_modules = config.num_modules
        self.num_destinations = num_destinations
        self.hidden_size = config.hidden_size
        self.num_heads = config.router_heads
        self.head_dim = config.hidden_size // config.router_heads
        self.mixer_type = mixer_type
        self.exclude_self = config.exclude_self and allow_identity
        self.allow_identity = allow_identity
        self.eps = config.rms_norm_eps

        if mixer_type in {"attnres", "shared_attnres"}:
            query_count = 1 if mixer_type == "shared_attnres" else num_destinations
            self.pseudo_queries = nn.Parameter(
                torch.zeros(query_count, self.num_heads, self.head_dim)
            )
            self.register_parameter("static_logits", None)
        elif mixer_type == "static":
            self.static_logits = nn.Parameter(
                torch.zeros(
                    num_destinations, self.num_modules + 1, self.num_heads
                )
            )
            self.register_parameter("pseudo_queries", None)
        else:
            self.register_parameter("pseudo_queries", None)
            self.register_parameter("static_logits", None)

        if mixer_type == "identity" and (
            not allow_identity or num_destinations != self.num_modules
        ):
            raise ValueError("identity routing is only valid for the module mixer.")

    def _mask_self(
        self, logits: torch.Tensor, include_embedding: bool
    ) -> torch.Tensor:
        if not self.exclude_self:
            return logits
        source_offset = 1 if include_embedding else 0
        mask = torch.zeros(
            self.num_destinations,
            logits.shape[1],
            dtype=torch.bool,
            device=logits.device,
        )
        indices = torch.arange(self.num_destinations, device=logits.device)
        mask[indices, indices + source_offset] = True
        extra_dims = (None,) * (logits.ndim - 2)
        return logits.masked_fill(mask[(slice(None), slice(None), *extra_dims)], -torch.inf)

    def forward(
        self,
        embedding: torch.Tensor,
        states: torch.Tensor,
        include_embedding: bool,
        return_entropy: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # embedding: [B, L, D]; states: [M, B, L, D]
        if self.mixer_type == "identity":
            output = states
            if include_embedding:
                output = 0.5 * (states + embedding.unsqueeze(0))
            entropy = embedding.new_zeros(()) if return_entropy else None
            return output, entropy

        values = (
            torch.cat((embedding.unsqueeze(0), states), dim=0)
            if include_embedding
            else states
        )
        sources, batch, length, hidden = values.shape
        values_h = values.view(
            sources, batch, length, self.num_heads, self.head_dim
        )

        if self.mixer_type == "uniform":
            if self.exclude_self:
                weights = values.new_ones(self.num_destinations, sources)
                source_offset = 1 if include_embedding else 0
                indices = torch.arange(
                    self.num_destinations, device=values.device
                )
                weights[indices, indices + source_offset] = 0
                weights = weights / weights.sum(dim=1, keepdim=True)
                output = torch.einsum("ms,sblhd->mblhd", weights, values_h)
                entropy_sources = sources - 1
            else:
                output = values_h.mean(dim=0, keepdim=True).expand(
                    self.num_destinations, -1, -1, -1, -1
                )
                entropy_sources = sources
            entropy = (
                embedding.new_tensor(math.log(entropy_sources))
                if return_entropy
                else None
            )
            return output.reshape(
                self.num_destinations, batch, length, hidden
            ), entropy

        if self.mixer_type == "static":
            logits = (
                self.static_logits
                if include_embedding
                else self.static_logits[:, 1:, :]
            )
            logits = self._mask_self(logits, include_embedding)
            weights = torch.softmax(logits.float(), dim=1).to(values.dtype)
            output = torch.einsum("msh,sblhd->mblhd", weights, values_h)
            entropy = None
            if return_entropy:
                p = weights.float().clamp_min(1e-20)
                entropy = -(p * p.log()).sum(dim=1).mean()
            return output.reshape(
                self.num_destinations, batch, length, hidden
            ), entropy

        # Attention Residuals-style dynamic routing.
        keys = rms_norm(values, None, self.eps).view(
            sources, batch, length, self.num_heads, self.head_dim
        )
        queries = (
            self.pseudo_queries.expand(self.num_destinations, -1, -1)
            if self.mixer_type == "shared_attnres"
            else self.pseudo_queries
        )
        logits = torch.einsum("mhd,sblhd->msblh", queries, keys)
        logits = self._mask_self(logits, include_embedding)
        weights = torch.softmax(logits.float(), dim=1).to(values.dtype)
        output = torch.einsum("msblh,sblhd->mblhd", weights, values_h)
        entropy = None
        if return_entropy:
            p = weights.float().clamp_min(1e-20)
            entropy = -(p * p.log()).sum(dim=1).mean()
        return output.reshape(
            self.num_destinations, batch, length, hidden
        ), entropy


class RecurrentMeshCell(nn.Module):
    """One synchronous update of every attention and FFN module."""

    def __init__(
        self, config: RecurrentAttnResConfig, needs_mixer: bool = True
    ) -> None:
        super().__init__()
        self.config = config
        self.input_norm = BatchedRMSNorm(
            config.num_modules, config.hidden_size, config.rms_norm_eps
        )
        self.mixer = (
            CrossModuleMixer(
                config,
                num_destinations=config.num_modules,
                mixer_type=config.router_type,
                allow_identity=True,
            )
            if needs_mixer
            else None
        )
        self.attention = (
            BatchedCausalSelfAttention(config)
            if config.num_attention_modules
            else None
        )
        self.ffn = BatchedSwiGLU(config) if config.num_ffn_modules else None

    def forward(
        self,
        previous_states: torch.Tensor,
        embedding: torch.Tensor,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        iteration: int,
        step_embedding: Optional[torch.Tensor],
        return_entropy: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if iteration == 0:
            routed = embedding.unsqueeze(0).expand(
                self.config.num_modules, -1, -1, -1
            )
            entropy = embedding.new_zeros(()) if return_entropy else None
        else:
            if self.mixer is None:
                raise RuntimeError("This iteration cell has no cross-module mixer.")
            routed, entropy = self.mixer(
                embedding,
                previous_states,
                include_embedding=self.config.input_injection == "every",
                return_entropy=return_entropy,
            )

        if step_embedding is not None:
            routed = routed + step_embedding[None, None, None, :]
        module_inputs = self.input_norm(routed)
        outputs: list[torch.Tensor] = []
        a = self.config.num_attention_modules
        if self.attention is not None:
            outputs.append(
                self.attention(module_inputs[:a], position_ids, attention_mask)
            )
        if self.ffn is not None:
            outputs.append(self.ffn(module_inputs[a:]))
        branch_outputs = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
        if self.config.branch_output == "residual":
            branch_outputs = routed + branch_outputs
        return branch_outputs, entropy


# ---------------------------------------------------------------------------
# Hugging Face causal language model
# ---------------------------------------------------------------------------


class RecurrentAttnResPreTrainedModel(PreTrainedModel):
    config_class = RecurrentAttnResConfig
    base_model_prefix = "mesh"
    supports_gradient_checkpointing = True
    _supports_cache_class = False

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class RecurrentAttnResForCausalLM(
    RecurrentAttnResPreTrainedModel, GenerationMixin
):
    """
    A Hugging Face-compatible causal LM.

    Generation intentionally sets use_cache=False and recomputes the prefix.
    Training/ablation throughput is the target.  A correct high-performance KV
    cache needs a distinct cache slot for every (iteration, attention module)
    pair and is best added only after the architecture is validated.
    """

    # Transformers 5 uses an explicit target -> source mapping.
    _tied_weights_keys = {"lm_head.weight": "embed_tokens.weight"}

    def __init__(self, config: RecurrentAttnResConfig) -> None:
        super().__init__(config)
        # Do not set padding_idx here. Decoder tokenizers commonly alias PAD to
        # EOS; padding_idx would then freeze and zero a real EOS embedding.
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embedding_dropout = nn.Dropout(config.embedding_dropout)
        if config.share_iteration_weights:
            self.shared_cell = RecurrentMeshCell(
                config, needs_mixer=config.num_iterations > 1
            )
            self.cells = None
        else:
            self.shared_cell = None
            self.cells = nn.ModuleList(
                RecurrentMeshCell(config, needs_mixer=iteration > 0)
                for iteration in range(config.num_iterations)
            )
        if config.use_step_embeddings:
            self.step_embeddings = nn.Parameter(
                torch.empty(config.num_iterations, config.hidden_size)
            )
            nn.init.normal_(self.step_embeddings, std=config.initializer_range)
        else:
            self.register_parameter("step_embeddings", None)
        self.readout = CrossModuleMixer(
            config,
            num_destinations=1,
            mixer_type=config.readout_type,
            allow_identity=False,
        )
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )
        self.gradient_checkpointing = False
        self.gradient_checkpointing_kwargs: dict[str, Any] = {
            "use_reentrant": False
        }
        self.post_init()
        self.tie_weights()

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.lm_head = new_embeddings

    def resize_token_embeddings(
        self,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        mean_resizing: bool = True,
    ) -> nn.Embedding:
        old_valid_size = self.config.valid_vocab_size
        embeddings = super().resize_token_embeddings(
            new_num_tokens=new_num_tokens,
            pad_to_multiple_of=pad_to_multiple_of,
            mean_resizing=mean_resizing,
        )
        self.config.valid_vocab_size = (
            min(old_valid_size, self.config.vocab_size)
            if new_num_tokens is None
            else int(new_num_tokens)
        )
        self.config._validate_structure()
        return embeddings

    def save_pretrained(
        self,
        save_directory: str | os.PathLike[str],
        is_main_process: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self.config.auto_map = {
            "AutoConfig": "recurrent_attnres_mesh.RecurrentAttnResConfig",
            "AutoModelForCausalLM": (
                "recurrent_attnres_mesh.RecurrentAttnResForCausalLM"
            ),
        }
        result = super().save_pretrained(
            save_directory, is_main_process, *args, **kwargs
        )
        if is_main_process:
            destination = Path(save_directory) / "recurrent_attnres_mesh.py"
            source = Path(__file__).resolve()
            if source != destination.resolve():
                shutil.copy2(source, destination)
        return result

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("inputs_embeds") is not None:
            raise NotImplementedError(
                "Generation from inputs_embeds is unsupported because "
                "Transformers requires a cache for decoder-only embedding "
                "prompts. Pass input_ids instead."
            )
        if kwargs.get("use_cache"):
            warnings.warn(
                "The recurrent mesh has no KV cache; forcing use_cache=False.",
                stacklevel=2,
            )
        kwargs["use_cache"] = False
        return super().generate(*args, **kwargs)

    def gradient_checkpointing_enable(
        self,
        gradient_checkpointing_kwargs: Optional[dict[str, Any]] = None,
        every_n_layers: int = 1,
    ) -> None:
        checkpoint_kwargs = dict(gradient_checkpointing_kwargs or {})
        checkpoint_kwargs.setdefault("use_reentrant", False)
        self.gradient_checkpointing_kwargs = checkpoint_kwargs
        if every_n_layers < 1:
            raise ValueError("every_n_layers must be positive.")
        if every_n_layers != 1:
            warnings.warn(
                "every_n_layers is a sequential-layer concept and is ignored; "
                "set config.checkpoint_chunk_steps to control recurrent "
                "checkpoint granularity.",
                stacklevel=2,
            )
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False

    def _cell(self, iteration: int) -> RecurrentMeshCell:
        if self.shared_cell is not None:
            return self.shared_cell
        assert self.cells is not None
        return self.cells[iteration]

    def _run_iteration_range(
        self,
        states: torch.Tensor,
        embedding: torch.Tensor,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        start: int,
        end: int,
    ) -> torch.Tensor:
        for iteration in range(start, end):
            step_embedding = (
                self.step_embeddings[iteration]
                if self.step_embeddings is not None
                else None
            )
            states, _ = self._cell(iteration)(
                states,
                embedding,
                position_ids,
                attention_mask,
                iteration,
                step_embedding,
                False,
            )
        return states

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: int | torch.Tensor = 0,
        output_router_stats: bool = False,
        **kwargs: Any,
    ) -> RecurrentMeshCausalLMOutput | tuple[torch.Tensor, ...]:
        num_items_in_batch = kwargs.pop("num_items_in_batch", None)
        del kwargs, output_attentions
        if past_key_values is not None:
            raise NotImplementedError(
                "past_key_values are not implemented. Use generate(..., "
                "use_cache=False)."
            )
        if use_cache:
            warnings.warn(
                "KV caching is not implemented for the recurrent module mesh; "
                "the full prefix will be recomputed.",
                stacklevel=2,
            )
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids and inputs_embeds.")
        return_dict = (
            self.config.return_dict if return_dict is None else return_dict
        )
        output_hidden_states = (
            self.config.output_hidden_states
            if output_hidden_states is None
            else output_hidden_states
        )
        embedding = (
            self.embed_tokens(input_ids)
            if inputs_embeds is None
            else inputs_embeds
        )
        embedding = self.embedding_dropout(embedding)
        batch, length, _ = embedding.shape
        if length > self.config.max_position_embeddings:
            raise ValueError(
                f"Sequence length {length} exceeds max_position_embeddings="
                f"{self.config.max_position_embeddings}."
            )
        if position_ids is None:
            if attention_mask is None:
                position_ids = torch.arange(
                    length, device=embedding.device, dtype=torch.long
                )[None, :].expand(batch, -1)
            else:
                position_ids = attention_mask.long().cumsum(dim=-1) - 1
                position_ids = position_ids.clamp_min(0)

        # Generation APIs commonly supply an all-ones mask.  It contains no
        # information, and converting it to a dense [A*B,1,L,L] mask prevents
        # the simplest causal Flash-SDPA path.  Avoid a device synchronization
        # while torch.compile is tracing; packed training already supplies no
        # mask at all.
        compiling = bool(
            hasattr(torch, "compiler") and torch.compiler.is_compiling()
        )
        if (
            attention_mask is not None
            and not compiling
            and bool(attention_mask.to(dtype=torch.bool).all().item())
        ):
            attention_mask = None

        states = embedding.new_zeros(
            self.config.num_modules, batch, length, self.config.hidden_size
        )
        hidden_history: list[torch.Tensor] = []
        entropy_history: list[torch.Tensor] = []
        collect_diagnostics = output_hidden_states or output_router_stats

        if self.gradient_checkpointing and self.training and not collect_diagnostics:
            chunk = self.config.checkpoint_chunk_steps
            for start in range(0, self.config.num_iterations, chunk):
                end = min(start + chunk, self.config.num_iterations)

                def run_chunk(
                    current_states: torch.Tensor,
                    current_embedding: torch.Tensor,
                    chunk_start: int = start,
                    chunk_end: int = end,
                ) -> torch.Tensor:
                    return self._run_iteration_range(
                        current_states,
                        current_embedding,
                        position_ids,
                        attention_mask,
                        chunk_start,
                        chunk_end,
                    )

                states = checkpoint(
                    run_chunk,
                    states,
                    embedding,
                    **self.gradient_checkpointing_kwargs,
                )
        else:
            for iteration in range(self.config.num_iterations):
                step_embedding = (
                    self.step_embeddings[iteration]
                    if self.step_embeddings is not None
                    else None
                )
                states, entropy = self._cell(iteration)(
                    states,
                    embedding,
                    position_ids,
                    attention_mask,
                    iteration,
                    step_embedding,
                    output_router_stats,
                )
                if output_hidden_states:
                    diagnostic = self.final_norm(embedding + states.mean(dim=0))
                    hidden_history.append(diagnostic)
                if entropy is not None:
                    entropy_history.append(entropy)

        readout, readout_entropy = self.readout(
            embedding, states, include_embedding=True, return_entropy=output_router_stats
        )
        hidden = self.final_norm(readout.squeeze(0))
        if attention_mask is not None:
            hidden = hidden.masked_fill(
                ~attention_mask.to(dtype=torch.bool).unsqueeze(-1), 0.0
            )
        if readout_entropy is not None:
            entropy_history.append(readout_entropy)

        hidden_for_logits = hidden
        if labels is None:
            if isinstance(logits_to_keep, int) and logits_to_keep > 0:
                hidden_for_logits = hidden[:, -logits_to_keep:, :]
            elif isinstance(logits_to_keep, torch.Tensor):
                hidden_for_logits = hidden.index_select(1, logits_to_keep)
        logits = self.lm_head(hidden_for_logits)
        if self.config.valid_vocab_size < self.config.vocab_size:
            # Retain a Tensor-Core-friendly padded projection while preventing
            # synthetic rows, which the tokenizer cannot decode, from receiving
            # probability mass or being selected by generate().
            logits[..., self.config.valid_vocab_size :] = torch.finfo(
                logits.dtype
            ).min

        loss = None
        if labels is not None:
            if logits.shape[1] != labels.shape[1]:
                raise ValueError("labels require full-sequence logits.")
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = labels[:, 1:].contiguous()
            if attention_mask is not None:
                shift_labels = shift_labels.masked_fill(
                    ~attention_mask[:, 1:].to(dtype=torch.bool), -100
                )
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="sum" if num_items_in_batch is not None else "mean",
            )
            if num_items_in_batch is not None:
                denominator = torch.as_tensor(
                    num_items_in_batch,
                    device=loss.device,
                    dtype=loss.dtype,
                ).clamp_min(1)
                loss = loss / denominator

        hidden_tuple = tuple(hidden_history) if output_hidden_states else None
        entropy_tuple = tuple(entropy_history) if output_router_stats else None
        if not return_dict:
            output: tuple[torch.Tensor, ...] = (logits,)
            if hidden_tuple is not None:
                output += (hidden_tuple,)  # type: ignore[assignment]
            return ((loss,) + output) if loss is not None else output
        return RecurrentMeshCausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=hidden_tuple,
            attentions=None,
            router_entropies=entropy_tuple,
        )

def register_auto_classes() -> None:
    """Register this in-process custom architecture with Hugging Face Auto APIs."""
    try:
        AutoConfig.register(RecurrentAttnResConfig.model_type, RecurrentAttnResConfig)
    except ValueError:
        pass
    try:
        AutoModelForCausalLM.register(
            RecurrentAttnResConfig, RecurrentAttnResForCausalLM
        )
    except ValueError:
        pass


register_auto_classes()


# ---------------------------------------------------------------------------
# Dataset packing and analytical accounting
# ---------------------------------------------------------------------------


class PackedTextDataset(IterableDataset):
    """Tokenize a Hugging Face dataset stream into fixed, no-padding blocks."""

    def __init__(
        self,
        source: Iterable[dict[str, Any]],
        tokenizer: Any,
        text_column: str,
        sequence_length: int,
        eos_token_id: int,
        max_examples: int = 0,
    ) -> None:
        super().__init__()
        self.source = source
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.sequence_length = sequence_length
        self.eos_token_id = eos_token_id
        self.max_examples = max_examples

    def set_epoch(self, epoch: int) -> None:
        """Forward epochs so a shuffled HF stream changes order on restart."""
        if hasattr(self.source, "set_epoch"):
            self.source.set_epoch(epoch)

    def _worker_source(self) -> Iterable[dict[str, Any]]:
        source = self.source
        worker = torch.utils.data.get_worker_info()
        if worker is None or isinstance(source, IterableDataset):
            # Hugging Face IterableDataset already reads PyTorch worker_info
            # and shards its ex_iterable.  Sharding it again here silently
            # drops data (and can fail when there are fewer shards than
            # workers).
            return source
        if hasattr(source, "shard"):
            try:
                return source.shard(
                    num_shards=worker.num_workers, index=worker.id
                )
            except (IndexError, TypeError, ValueError):
                pass
        # Generic map-style/iterable fallback: assign every W-th record to a
        # worker so wrapping it in this IterableDataset does not duplicate it.
        source = itertools.islice(source, worker.id, None, worker.num_workers)
        return source

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        token_buffer: list[int] = []
        cursor = 0
        seen = 0
        worker = torch.utils.data.get_worker_info()
        example_limit = self.max_examples
        if example_limit and worker is not None:
            active_workers = worker.num_workers
            if isinstance(self.source, IterableDataset):
                source_shards = getattr(self.source, "num_shards", active_workers)
                try:
                    active_workers = min(active_workers, max(int(source_shards), 1))
                except (TypeError, ValueError):
                    active_workers = worker.num_workers
            if worker.id >= active_workers:
                return
            quotient, remainder = divmod(example_limit, active_workers)
            example_limit = quotient + int(worker.id < remainder)
            if example_limit == 0:
                return
        for example in self._worker_source():
            if example_limit and seen >= example_limit:
                break
            seen += 1
            text = example.get(self.text_column)
            if not isinstance(text, str) or not text:
                continue
            ids = self.tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"]
            token_buffer.extend(ids)
            token_buffer.append(self.eos_token_id)

            while len(token_buffer) - cursor >= self.sequence_length:
                block = torch.tensor(
                    token_buffer[cursor : cursor + self.sequence_length],
                    dtype=torch.long,
                )
                cursor += self.sequence_length
                yield {"input_ids": block, "labels": block.clone()}

            # Avoid quadratic front-deletion while keeping the buffer bounded.
            if cursor > 1_000_000 or cursor > len(token_buffer) // 2:
                token_buffer = token_buffer[cursor:]
                cursor = 0


def round_up(value: int, multiple: int) -> int:
    return value if multiple <= 1 else ((value + multiple - 1) // multiple) * multiple


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable


def estimate_parameter_count(config: RecurrentAttnResConfig) -> int:
    """Exactly count trainable scalars without materializing the model."""
    d = config.hidden_size
    a = config.num_attention_modules
    f = config.num_ffn_modules
    m = config.num_modules

    # One cell contains a packed RMSNorm, packed QKV/output projections, and
    # packed SwiGLU gate/up/down projections.
    cell_without_mixer = m * d + 4 * a * d * d + 3 * f * d * config.intermediate_size

    def mixer_parameters(destinations: int, mixer_type: str) -> int:
        if mixer_type == "attnres":
            return destinations * d
        if mixer_type == "shared_attnres":
            return d
        if mixer_type == "static":
            return destinations * (m + 1) * config.router_heads
        return 0

    if config.share_iteration_weights:
        cell_parameters = cell_without_mixer
        if config.num_iterations > 1:
            cell_parameters += mixer_parameters(m, config.router_type)
    else:
        cell_parameters = config.num_iterations * cell_without_mixer
        cell_parameters += max(config.num_iterations - 1, 0) * mixer_parameters(
            m, config.router_type
        )

    embeddings = config.vocab_size * d
    if not config.tie_word_embeddings:
        embeddings *= 2
    step_embeddings = config.num_iterations * d if config.use_step_embeddings else 0
    readout = mixer_parameters(1, config.readout_type)
    final_norm = d
    return embeddings + cell_parameters + step_embeddings + readout + final_norm


def estimate_forward_flops(
    config: RecurrentAttnResConfig,
    batch_size: int,
    sequence_length: int,
    include_lm_head: bool = True,
    gradient_checkpointing: bool = False,
) -> dict[str, float]:
    """
    Approximate dense matmul FLOPs for one forward pass.

    The attention score term uses the exact number of causal query/key pairs.
    Elementwise operations, normalization, softmax, and optimizer work are not
    included.  Training matmul FLOPs are approximated as forward + two
    backward matmuls.  With recurrent-cell checkpointing, the checkpointed
    recurrent body is also counted once for backward recomputation.
    """
    b, length, d = batch_size, sequence_length, config.hidden_size
    a, f, m = (
        config.num_attention_modules,
        config.num_ffn_modules,
        config.num_modules,
    )
    causal_pairs = length * (length + 1) / 2
    attention_projection = a * 8.0 * b * length * d * d
    attention_scores = a * 4.0 * b * causal_pairs * d
    ffn = f * 6.0 * b * length * d * config.intermediate_size
    branch_per_iteration = attention_projection + attention_scores + ffn

    router = 0.0
    if config.router_type in {"attnres", "shared_attnres"}:
        # Key/query scores plus value aggregation for iterations after the first.
        sources = m + (1 if config.input_injection == "every" else 0)
        router = (
            max(config.num_iterations - 1, 0)
            * 4.0
            * b
            * length
            * m
            * sources
            * d
        )
    elif config.router_type == "static":
        sources = m + (1 if config.input_injection == "every" else 0)
        router = (
            max(config.num_iterations - 1, 0)
            * 2.0
            * b
            * length
            * m
            * sources
            * d
        )
    readout = (
        4.0 if config.readout_type == "attnres" else 2.0
    ) * b * length * (m + 1) * d
    recurrent_body = config.num_iterations * branch_per_iteration + router
    body = recurrent_body + readout
    lm_head = (
        2.0 * b * length * d * config.vocab_size if include_lm_head else 0.0
    )
    forward = body + lm_head
    recompute = recurrent_body if gradient_checkpointing else 0.0
    return {
        "forward_flops": forward,
        "training_matmul_flops": 3.0 * forward + recompute,
        "checkpoint_recompute_flops": recompute,
        "body_forward_flops": body,
        "lm_head_forward_flops": lm_head,
        "module_evaluations": float(config.num_iterations * m),
        "attention_evaluations": float(config.num_iterations * a),
        "ffn_evaluations": float(config.num_iterations * f),
    }


def build_config(
    args: argparse.Namespace,
    tokenizer: Optional[Any] = None,
) -> RecurrentAttnResConfig:
    tokenizer_size = len(tokenizer) if tokenizer is not None else None
    if args.vocab_size > 0:
        if tokenizer_size is not None and args.vocab_size < tokenizer_size:
            raise ValueError(
                f"--vocab-size={args.vocab_size} is smaller than tokenizer "
                f"size {tokenizer_size}."
            )
        vocab_size = args.vocab_size
    elif tokenizer_size is not None:
        vocab_size = round_up(tokenizer_size, args.pad_vocab_multiple)
    else:
        vocab_size = 50_304
    valid_vocab_size = tokenizer_size if tokenizer_size is not None else vocab_size
    max_positions = max(args.max_position_embeddings, args.sequence_length)
    eos = getattr(tokenizer, "eos_token_id", None) if tokenizer is not None else 2
    bos = getattr(tokenizer, "bos_token_id", None) if tokenizer is not None else 1
    pad = getattr(tokenizer, "pad_token_id", None) if tokenizer is not None else 0
    if eos is None:
        eos = pad if pad is not None else 0
    if pad is None:
        pad = eos
    return RecurrentAttnResConfig(
        vocab_size=vocab_size,
        valid_vocab_size=valid_vocab_size,
        hidden_size=args.hidden_size,
        num_attention_modules=args.num_attention_modules,
        num_ffn_modules=args.num_ffn_modules,
        num_iterations=args.num_iterations,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=max_positions,
        rope_theta=args.rope_theta,
        router_type=args.router_type,
        router_heads=args.router_heads,
        readout_type=args.readout_type,
        input_injection=args.input_injection,
        share_iteration_weights=args.share_iteration_weights,
        use_step_embeddings=args.use_step_embeddings,
        branch_output=args.branch_output,
        exclude_self=args.exclude_self,
        attention_dropout=args.attention_dropout,
        residual_dropout=args.residual_dropout,
        embedding_dropout=args.embedding_dropout,
        initializer_range=args.initializer_range,
        rms_norm_eps=args.rms_norm_eps,
        sdpa_backend=args.sdpa_backend,
        require_fused_sdpa=args.require_fused_sdpa,
        checkpoint_chunk_steps=args.checkpoint_chunk_steps,
        tie_word_embeddings=args.tie_word_embeddings,
        bos_token_id=bos,
        eos_token_id=eos,
        pad_token_id=pad,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_hardware_profile(args: argparse.Namespace) -> argparse.Namespace:
    """Fill only unset arguments, preserving CLI and run-config ablations."""
    try:
        profile = HARDWARE_PROFILES[args.hardware_profile]
    except KeyError as exc:
        raise ValueError(f"Unknown hardware profile: {args.hardware_profile!r}") from exc
    for destination, value in profile.items():
        if getattr(args, destination, None) is None:
            setattr(args, destination, value)
    return args


def configure_runtime_environment(args: argparse.Namespace) -> None:
    """Set allocator policy before the first CUDA context/allocation."""
    if not args.allocator_expandable_segments:
        return
    option = "expandable_segments:True"
    configured = (
        os.environ.get("PYTORCH_ALLOC_CONF")
        or os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
        or ""
    )
    if "expandable_segments" not in configured:
        configured = f"{configured},{option}" if configured else option
    # PYTORCH_ALLOC_CONF is the current spelling; the CUDA-prefixed alias
    # keeps the single file useful with the older PyTorch API floor it accepts.
    os.environ["PYTORCH_ALLOC_CONF"] = configured
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = configured
    if torch.cuda.is_initialized():
        warnings.warn(
            "CUDA was already initialized, so the expandable-segments allocator "
            "setting may not take effect in this process.",
            stacklevel=2,
        )


def configure_torch(tf32: bool) -> None:
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if tf32 else "highest")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(True)


def select_execution_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not 0 <= local_rank < torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} does not identify a visible CUDA device."
        )
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank)


@torch.no_grad()
def probe_sdpa_backend(
    device: torch.device,
    head_dim: int,
    dtype: torch.dtype,
    policy: str,
) -> None:
    """Execute the configured head shape/dtype under the selected SDPA policy."""
    with torch.cuda.device(device):
        query = torch.randn(1, 1, 64, head_dim, device=device, dtype=dtype)
        with sdpa_backend_context(policy):
            output = F.scaled_dot_product_attention(
                query, query, query, dropout_p=0.0, is_causal=True
            )
        # Surface an asynchronous kernel failure during preflight, not after
        # model construction or compilation.
        output.sum().item()
        del output, query
        torch.cuda.empty_cache()


def validate_execution_hardware(
    args: argparse.Namespace,
    device: torch.device,
    config: Optional[RecurrentAttnResConfig] = None,
) -> dict[str, Any]:
    """Validate the selected device contract and return stable run metadata."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    attention_modules = (
        config.num_attention_modules
        if config is not None
        else args.num_attention_modules
    )
    info: dict[str, Any] = {
        "hardware_profile": args.hardware_profile,
        "device": str(device),
        "device_name": "CPU",
        "compute_capability": None,
        "total_memory_gb": 0.0,
        "free_memory_gb_at_start": 0.0,
        "memory_budget_gb": 0.0,
        "torch_cuda_version": torch.version.cuda,
        "bf16_supported": False,
        "sdpa_backend_policy": args.sdpa_backend,
        "fused_sdpa_required": bool(
            args.require_fused_sdpa and attention_modules > 0
        ),
        "sdpa_backend_probe": "not_applicable",
        "sdpa_probe_dtype": "not_applicable",
        # Backward-compatible A100-era fields.
        "flash_sdpa_required": bool(
            args.require_fused_sdpa
            and args.sdpa_backend == "flash"
            and attention_modules > 0
        ),
        "flash_sdpa_probe": "not_applicable",
        "flash_sdpa_dtype": "not_applicable",
        "world_size": world_size,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        with torch.cuda.device(device):
            free_bytes, total_bytes = torch.cuda.mem_get_info()
        capability = torch.cuda.get_device_capability(device)
        info.update(
            {
                "device_name": properties.name,
                "compute_capability": list(capability),
                "total_memory_gb": total_bytes / 2**30,
                "free_memory_gb_at_start": free_bytes / 2**30,
                "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            }
        )
        info["memory_budget_gb"] = max(
            0.0,
            min(
                info["total_memory_gb"] * args.target_peak_memory_fraction,
                info["free_memory_gb_at_start"] - args.memory_reserve_gb,
            ),
        )

    profile_errors: list[str] = []
    contract = GPU_PROFILE_CONTRACTS.get(args.hardware_profile)
    info["gpu_contract"] = contract
    if contract is not None:
        if args.hardware_profile != "cuda-generic" and world_size != 1:
            profile_errors.append(
                "WORLD_SIZE must be 1 for a single-GPU hardware profile, "
                f"got {world_size}"
            )
        if device.type != "cuda":
            profile_errors.append("CUDA is unavailable")
        else:
            name_token = contract["name_token"]
            if name_token and name_token.upper() not in str(
                info["device_name"]
            ).upper():
                profile_errors.append(
                    f"device name is {info['device_name']!r}; expected token "
                    f"{name_token!r}"
                )
            expected_capability = contract["compute_capability"]
            if (
                expected_capability is not None
                and info["compute_capability"] != list(expected_capability)
            ):
                profile_errors.append(
                    "compute capability is "
                    f"{info['compute_capability']}, expected "
                    f"{list(expected_capability)}"
                )
            min_hbm_gb = float(contract["min_hbm_gb"])
            if info["total_memory_gb"] < min_hbm_gb:
                profile_errors.append(
                    f"visible GPU memory is {info['total_memory_gb']:.2f} GiB, "
                    f"expected >={min_hbm_gb:g}"
                )
            if contract["require_bf16"] and not info["bf16_supported"]:
                profile_errors.append("CUDA device does not report BF16 support")
            if info["memory_budget_gb"] <= 0.0:
                profile_errors.append(
                    "free HBM does not exceed the configured memory reserve"
                )
    info["profile_valid"] = not profile_errors
    info["profile_errors"] = profile_errors
    if profile_errors and args.strict_hardware_profile:
        raise RuntimeError(
            f"{args.hardware_profile} hardware preflight failed: "
            + "; ".join(profile_errors)
        )
    if profile_errors:
        warnings.warn(
            "Hardware profile mismatch (strict checking disabled): "
            + "; ".join(profile_errors),
            stacklevel=2,
        )

    backend_probe_required = bool(
        attention_modules > 0
        and (args.require_fused_sdpa or args.sdpa_backend != "auto")
    )
    if backend_probe_required:
        if device.type != "cuda":
            raise RuntimeError(
                "The selected SDPA backend policy needs CUDA when attention "
                "modules are enabled."
            )
        hidden_size = config.hidden_size if config is not None else args.hidden_size
        num_heads = (
            config.num_attention_heads
            if config is not None
            else args.num_attention_heads
        )
        head_dim = hidden_size // num_heads
        probe_dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "no": torch.float32,
        }[args.mixed_precision]
        info["sdpa_probe_dtype"] = str(probe_dtype)
        try:
            probe_sdpa_backend(
                device,
                head_dim,
                probe_dtype,
                args.sdpa_backend,
            )
        except Exception as exc:
            info["sdpa_backend_probe"] = "failed"
            raise RuntimeError(
                f"SDPA backend preflight ({args.sdpa_backend}) failed for "
                f"head_dim={head_dim}, dtype={probe_dtype}."
            ) from exc
        info["sdpa_backend_probe"] = "passed"
        if args.sdpa_backend == "flash":
            info["flash_sdpa_probe"] = "passed"
            info["flash_sdpa_dtype"] = str(probe_dtype)
    return info


def cuda_memory_metrics(
    device: torch.device,
    args: argparse.Namespace,
    hardware: dict[str, Any],
) -> dict[str, float | bool]:
    if device.type != "cuda":
        return {
            "current_allocated_gb": 0.0,
            "current_reserved_gb": 0.0,
            "peak_allocated_gb": 0.0,
            "peak_reserved_gb": 0.0,
            "peak_memory_gb": 0.0,
            "free_memory_gb": 0.0,
            "peak_reserved_fraction": 0.0,
            "memory_budget_gb": 0.0,
            "within_memory_budget": True,
        }
    with torch.cuda.device(device):
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated(device) / 2**30
    reserved = torch.cuda.memory_reserved(device) / 2**30
    peak_allocated = torch.cuda.max_memory_allocated(device) / 2**30
    peak_reserved = torch.cuda.max_memory_reserved(device) / 2**30
    total_gb = total_bytes / 2**30
    budget = float(hardware.get("memory_budget_gb", 0.0))
    return {
        "current_allocated_gb": allocated,
        "current_reserved_gb": reserved,
        "peak_allocated_gb": peak_allocated,
        "peak_reserved_gb": peak_reserved,
        # Backward-compatible name used by the existing sweep CSV.
        "peak_memory_gb": peak_allocated,
        "free_memory_gb": free_bytes / 2**30,
        "peak_reserved_fraction": peak_reserved / max(total_gb, 1e-9),
        "memory_budget_gb": budget,
        "within_memory_budget": peak_reserved <= budget,
    }


def estimated_dense_bf16_mfu(
    achieved_tflops: float,
    args: argparse.Namespace,
    hardware: dict[str, Any],
    num_devices: int = 1,
) -> float:
    recorded_device = hardware.get("device")
    if not (
        hardware.get("profile_valid")
        and (recorded_device is None or str(recorded_device).startswith("cuda"))
        and args.mixed_precision == "bf16"
        and args.bf16_peak_tflops > 0
    ):
        return math.nan
    return achieved_tflops / (args.bf16_peak_tflops * num_devices)


def estimated_a100_bf16_mfu(
    achieved_tflops: float,
    args: argparse.Namespace,
    hardware: dict[str, Any],
    num_devices: int = 1,
) -> float:
    """Compatibility field for A100 analyses; use the dense metric elsewhere."""
    if args.hardware_profile != "a100-80gb":
        return math.nan
    return estimated_dense_bf16_mfu(
        achieved_tflops, args, hardware, num_devices
    )


def resolved_run_metadata(
    args: argparse.Namespace,
    config: Optional[RecurrentAttnResConfig] = None,
) -> dict[str, Any]:
    config = config or build_config(args)
    parameters = estimate_parameter_count(config)
    estimates = estimate_forward_flops(
        config,
        args.micro_batch_size,
        args.sequence_length,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    return {
        "hardware_profile": args.hardware_profile,
        "hidden_size": config.hidden_size,
        "num_attention_heads": config.num_attention_heads,
        "head_dim": config.hidden_size // config.num_attention_heads,
        "intermediate_size": config.intermediate_size,
        "num_attention_modules": config.num_attention_modules,
        "num_ffn_modules": config.num_ffn_modules,
        "num_iterations": config.num_iterations,
        "share_iteration_weights": config.share_iteration_weights,
        "sequence_length": args.sequence_length,
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_tokens_per_update": (
            args.micro_batch_size
            * args.sequence_length
            * args.gradient_accumulation_steps
        ),
        "mixed_precision": args.mixed_precision,
        "tf32": args.tf32,
        "compile_model": args.compile_model,
        "compile_mode": args.compile_mode,
        "gradient_checkpointing": args.gradient_checkpointing,
        "checkpoint_chunk_steps": args.checkpoint_chunk_steps,
        "sdpa_backend": config.sdpa_backend,
        "require_fused_sdpa": config.require_fused_sdpa,
        "bf16_peak_tflops": args.bf16_peak_tflops,
        # Backward-compatible interpretation: true means flash-only policy.
        "require_flash": config.require_flash,
        "estimated_parameters": parameters,
        "estimated_fp32_adam_training_state_gb": parameters * 16 / 2**30,
        **estimates,
    }


def inspect_configuration(args: argparse.Namespace) -> dict[str, Any]:
    result = resolved_run_metadata(args)
    result.update(
        {
            "strict_hardware_profile": args.strict_hardware_profile,
            "gpu_contract": GPU_PROFILE_CONTRACTS.get(args.hardware_profile),
            "allocator_expandable_segments": args.allocator_expandable_segments,
            "target_peak_memory_fraction": args.target_peak_memory_fraction,
            "memory_reserve_gb": args.memory_reserve_gb,
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor,
            "persistent_workers": args.persistent_workers,
        }
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def make_optimizer(
    model: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    no_decay_terms = (
        "bias",
        "norm",
        "pseudo_queries",
        "static_logits",
        "step_embeddings",
    )
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or any(term in name for term in no_decay_terms):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups = [
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    optimizer_kwargs: dict[str, Any] = {
        "lr": args.learning_rate,
        "betas": (args.adam_beta1, args.adam_beta2),
        "eps": args.adam_epsilon,
    }
    if args.fused_adamw and device.type == "cuda":
        optimizer_kwargs["fused"] = True
    try:
        return torch.optim.AdamW(groups, **optimizer_kwargs)
    except TypeError:
        optimizer_kwargs.pop("fused", None)
        return torch.optim.AdamW(groups, **optimizer_kwargs)


def data_loader_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "batch_size": args.micro_batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": True,
    }
    if args.num_workers > 0:
        kwargs["prefetch_factor"] = args.prefetch_factor
        kwargs["persistent_workers"] = args.persistent_workers
    return kwargs


def set_dataloader_epoch(loader: Any, epoch: int) -> None:
    """Set both Accelerate's shard epoch and the wrapped source epoch."""
    if hasattr(loader, "set_epoch"):
        loader.set_epoch(epoch)
    dataset = getattr(loader, "dataset", None)
    if dataset is not None and hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)


def load_packed_dataset(
    args: argparse.Namespace,
    tokenizer: Any,
    split: str,
    training: bool,
) -> PackedTextDataset:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install the Hugging Face datasets package for training.") from exc

    if training and args.train_file:
        source = load_dataset(
            "text",
            data_files={"train": args.train_file},
            split="train",
            streaming=args.streaming,
        )
    elif not training and args.eval_file:
        source = load_dataset(
            "text",
            data_files={"validation": args.eval_file},
            split="validation",
            streaming=args.streaming,
        )
    else:
        load_kwargs: dict[str, Any] = {
            "path": args.dataset,
            "split": split,
            "streaming": args.streaming,
        }
        if args.dataset_config:
            load_kwargs["name"] = args.dataset_config
        if args.dataset_revision:
            load_kwargs["revision"] = args.dataset_revision
        source = load_dataset(**load_kwargs)

    if training and args.shuffle_buffer > 0:
        if args.streaming:
            source = source.shuffle(
                seed=args.seed, buffer_size=args.shuffle_buffer
            )
        else:
            source = source.shuffle(seed=args.seed)
    return PackedTextDataset(
        source=source,
        tokenizer=tokenizer,
        text_column=args.text_column,
        sequence_length=args.sequence_length,
        eos_token_id=tokenizer.eos_token_id,
        max_examples=args.max_dataset_examples,
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def local_file_identity(path_value: str) -> Optional[dict[str, Any]]:
    if not path_value:
        return None
    path = Path(path_value).resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def tokenizer_identity(tokenizer: Any, args: argparse.Namespace) -> dict[str, Any]:
    vocabulary = sorted(tokenizer.get_vocab().items())
    vocabulary_hash = hashlib.sha256(
        json.dumps(
            vocabulary, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    init_kwargs = getattr(tokenizer, "init_kwargs", {})
    return {
        "requested_name": args.tokenizer,
        "requested_revision": args.tokenizer_revision,
        "resolved_name": getattr(tokenizer, "name_or_path", None),
        "resolved_commit": init_kwargs.get("_commit_hash"),
        "vocab_size": len(tokenizer),
        "vocab_sha256": vocabulary_hash,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "special_tokens_map": {
            key: str(value)
            for key, value in tokenizer.special_tokens_map.items()
        },
    }


def installed_package_version(name: str) -> str:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return "unknown"


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model: nn.Module,
    eval_loader: DataLoader,
    accelerator: Any,
    num_batches: int,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    iterator = iter(eval_loader)
    for _ in range(num_batches):
        try:
            batch = next(iterator)
        except StopIteration:
            break
        output = model(**batch)
        gathered = accelerator.gather_for_metrics(output.loss.detach().reshape(1))
        losses.extend(gathered.float().cpu().tolist())
    if was_training:
        model.train()
    if not losses:
        return {"eval_loss": math.nan, "eval_perplexity": math.nan}
    loss = sum(losses) / len(losses)
    return {
        "eval_loss": loss,
        "eval_perplexity": math.exp(min(loss, 20.0)),
    }


def unwrap_compiled_model(model: nn.Module, accelerator: Any) -> nn.Module:
    unwrapped = accelerator.unwrap_model(model)
    return getattr(unwrapped, "_orig_mod", unwrapped)


def train(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from accelerate import Accelerator
        from accelerate.utils import GradientAccumulationPlugin
    except ImportError as exc:
        raise SystemExit("Install Hugging Face Accelerate for training.") from exc

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    configure_torch(args.tf32)
    execution_device = select_execution_device()
    preflight_config = (
        RecurrentAttnResConfig.from_pretrained(args.load_model)
        if args.load_model
        else None
    )
    hardware = validate_execution_hardware(
        args, execution_device, config=preflight_config
    )
    seed_everything(args.seed)
    accumulation = GradientAccumulationPlugin(
        num_steps=args.gradient_accumulation_steps,
        sync_with_dataloader=False,
    )
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_plugin=accumulation,
    )
    if (
        args.mixed_precision == "bf16"
        and accelerator.device.type == "cuda"
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("BF16 was requested, but this CUDA device lacks BF16 support.")
    accelerator_index = (
        accelerator.device.index
        if accelerator.device.index is not None
        else (torch.cuda.current_device() if accelerator.device.type == "cuda" else None)
    )
    if (
        accelerator.device.type != execution_device.type
        or accelerator_index != execution_device.index
    ):
        raise RuntimeError(
            "Accelerate selected a different device than hardware preflight: "
            f"{accelerator.device} versus {hardware['device']}."
        )

    if (
        not args.dataset_revision
        and (not args.train_file or not args.eval_file)
    ):
        try:
            from huggingface_hub import HfApi

            args.dataset_revision = HfApi().dataset_info(args.dataset).sha
        except Exception as exc:  # Cached/offline datasets can still be usable.
            warnings.warn(
                "Could not resolve the Hub dataset to an immutable commit; "
                "exact resume assumes its contents remain unchanged. "
                f"Reason: {exc}",
                stacklevel=2,
            )

    tokenizer_source = args.tokenizer or args.load_model or "gpt2"
    args.tokenizer = tokenizer_source
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        revision=args.tokenizer_revision or None,
        use_fast=True,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must define an EOS token.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer_signature = tokenizer_identity(tokenizer, args)

    if args.load_model:
        model = RecurrentAttnResForCausalLM.from_pretrained(args.load_model)
        config = model.config
        # SDPA enforcement and checkpoint chunking are execution policy, not
        # learned architecture. A checkpoint obeys the active device profile
        # instead of silently retaining an older execution setting.
        config.sdpa_backend = args.sdpa_backend
        config.require_fused_sdpa = args.require_fused_sdpa
        config.require_flash = bool(
            args.require_fused_sdpa and args.sdpa_backend == "flash"
        )
        config.checkpoint_chunk_steps = args.checkpoint_chunk_steps
        for module in model.modules():
            if isinstance(module, BatchedCausalSelfAttention):
                module.sdpa_backend = args.sdpa_backend
                module.require_fused_sdpa = args.require_fused_sdpa
                module.require_flash = bool(
                    args.require_fused_sdpa and args.sdpa_backend == "flash"
                )
        if len(tokenizer) != config.valid_vocab_size:
            raise ValueError(
                f"Tokenizer size {len(tokenizer)} does not match checkpoint "
                f"valid_vocab_size={config.valid_vocab_size}."
            )
        for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
            expected = getattr(config, name, None)
            actual = getattr(tokenizer, name, None)
            if expected is not None and actual != expected:
                raise ValueError(
                    f"Tokenizer {name}={actual!r} does not match checkpoint "
                    f"value {expected!r}."
                )
        expected_vocab_hash = getattr(config, "tokenizer_vocab_sha256", None)
        if (
            expected_vocab_hash is not None
            and tokenizer_signature["vocab_sha256"] != expected_vocab_hash
        ):
            raise ValueError(
                "Tokenizer token-to-ID mapping does not match the checkpoint."
            )
    else:
        config = build_config(args, tokenizer)
        model = RecurrentAttnResForCausalLM(config)
    config.tokenizer_vocab_sha256 = tokenizer_signature["vocab_sha256"]
    config.tokenizer_source = tokenizer_source
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    total_parameters, trainable_parameters = count_parameters(model)
    analytical_parameters = estimate_parameter_count(config)
    if total_parameters != analytical_parameters:
        raise AssertionError(
            "Analytical parameter accounting drifted from the model: "
            f"{analytical_parameters} versus {total_parameters}."
        )
    flop_estimate = estimate_forward_flops(
        config,
        args.micro_batch_size,
        args.sequence_length,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    if accelerator.is_main_process:
        accelerator.print(
            f"Model: {total_parameters / 1e6:.2f}M parameters; "
            f"{flop_estimate['module_evaluations']:.0f} module evaluations/token; "
            f"Transformers {transformers.__version__}"
        )
        accelerator.print(
            json.dumps(
                {
                    **resolved_run_metadata(args, config),
                    **hardware,
                },
                sort_keys=True,
            )
        )

    optimizer = make_optimizer(model, args, accelerator.device)
    train_dataset = load_packed_dataset(
        args, tokenizer, args.train_split, training=True
    )
    eval_dataset = load_packed_dataset(
        args, tokenizer, args.eval_split, training=False
    )
    loader_kwargs = data_loader_kwargs(args)
    train_loader = DataLoader(train_dataset, **loader_kwargs)
    eval_loader = DataLoader(eval_dataset, **loader_kwargs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        # AcceleratedScheduler advances once per process when batches are
        # sharded, so construct the underlying schedule on its expanded step
        # scale. This is a no-op for the intended single-GPU profiles and keeps
        # DDP warmup/decay aligned with optimizer updates.
        num_warmup_steps=args.warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_steps * accelerator.num_processes,
    )

    if args.compile_model:
        if not hasattr(torch, "compile"):
            raise RuntimeError("This PyTorch version does not provide torch.compile.")
        model = torch.compile(
            model,
            mode=args.compile_mode,
            dynamic=False,
            fullgraph=False,
        )

    model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, scheduler
    )

    data_signature = {
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "dataset_revision": args.dataset_revision,
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "train_file": local_file_identity(args.train_file),
        "eval_file": local_file_identity(args.eval_file),
        "source_fingerprint": (
            str(getattr(train_dataset.source, "_fingerprint"))
            if getattr(train_dataset.source, "_fingerprint", None) is not None
            else None
        ),
        "tokenizer": tokenizer_signature,
        "text_column": args.text_column,
        "streaming": args.streaming,
        "shuffle_buffer": args.shuffle_buffer,
        "max_dataset_examples": args.max_dataset_examples,
        "sequence_length": args.sequence_length,
        "micro_batch_size": args.micro_batch_size,
        "num_workers": args.num_workers,
        "seed": args.seed,
    }
    resume_semantics = {
        "model_config": config.to_dict(),
        "optimizer": {
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "adam_beta1": args.adam_beta1,
            "adam_beta2": args.adam_beta2,
            "adam_epsilon": args.adam_epsilon,
            "max_grad_norm": args.max_grad_norm,
            "fused_adamw": args.fused_adamw,
        },
        "schedule": {
            "warmup_steps": args.warmup_steps,
            "max_steps": args.max_steps,
        },
        "execution": {
            "hardware_profile": args.hardware_profile,
            "mixed_precision": args.mixed_precision,
            "tf32": args.tf32,
            "compile_model": args.compile_model,
            "compile_mode": args.compile_mode,
            "gradient_checkpointing": args.gradient_checkpointing,
            "checkpoint_chunk_steps": args.checkpoint_chunk_steps,
            "sdpa_backend": args.sdpa_backend,
            "require_fused_sdpa": args.require_fused_sdpa,
        },
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "accelerate": installed_package_version("accelerate"),
            "datasets": installed_package_version("datasets"),
        },
    }
    resume_semantics_sha256 = hashlib.sha256(
        json.dumps(
            resume_semantics,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    resumed_from_step = 0
    consumed_micro_batches = 0
    data_epoch = 0
    batches_in_epoch = 0
    resume_loader = train_loader
    prefetched_batch: Optional[dict[str, torch.Tensor]] = None
    if args.resume_state:
        progress_path = Path(args.resume_state) / "training_progress.json"
        if not progress_path.is_file():
            raise FileNotFoundError(
                f"Missing resume sidecar {progress_path}; refusing to guess "
                "the optimizer step or data cursor."
            )
        with progress_path.open("r", encoding="utf-8") as handle:
            progress = json.load(handle)
        for key, expected in (
            ("gradient_accumulation_steps", args.gradient_accumulation_steps),
            ("num_processes", accelerator.num_processes),
        ):
            if progress.get(key) != expected:
                raise ValueError(
                    f"Cannot resume: checkpoint {key}={progress.get(key)!r}, "
                    f"current run has {expected!r}."
                )
        if progress.get("data_signature") != data_signature:
            raise ValueError(
                "Cannot resume with changed data, packing, worker, batch, or "
                "seed settings."
            )
        if progress.get("resume_semantics_sha256") != resume_semantics_sha256:
            raise ValueError(
                "Cannot resume with changed model, optimizer, schedule, "
                "precision, compilation, checkpointing, or library-version "
                "semantics."
            )
        resumed_from_step = int(progress["completed_steps"])
        consumed_micro_batches = int(progress["consumed_micro_batches"])
        data_epoch = int(progress["data_epoch"])
        batches_in_epoch = int(progress["batches_in_epoch"])
        if not 0 <= resumed_from_step <= args.max_steps:
            raise ValueError(
                f"Checkpoint step {resumed_from_step} is outside the requested "
                f"0..{args.max_steps} range."
            )
        set_dataloader_epoch(train_loader, data_epoch)
        if batches_in_epoch:
            accelerator.print(
                f"Restoring data cursor: epoch {data_epoch}, skipping "
                f"{batches_in_epoch} prepared microbatches."
            )
            resume_loader = accelerator.skip_first_batches(
                train_loader, batches_in_epoch
            )
            # skip_first_batches creates a new Accelerate loader whose epoch
            # counter starts at zero, so set the desired epoch on that loader.
            set_dataloader_epoch(resume_loader, data_epoch)
        # Actually fetch one post-cursor batch before restoring RNG: merely
        # constructing an iterator is lazy.  This creates/fast-forwards workers
        # first; load_state then restores the exact model/dropout RNG state.
        train_iterator = iter(resume_loader)
        try:
            prefetched_batch = next(train_iterator)
        except StopIteration:
            data_epoch += 1
            batches_in_epoch = 0
            set_dataloader_epoch(train_loader, data_epoch)
            train_iterator = iter(train_loader)
            try:
                prefetched_batch = next(train_iterator)
            except StopIteration as exc:
                raise RuntimeError(
                    "The training dataset produced no full packed batch."
                ) from exc
        accelerator.load_state(args.resume_state)
    else:
        set_dataloader_epoch(train_loader, data_epoch)
        train_iterator = iter(train_loader)

    output_dir = Path(args.output_dir)
    accelerator.wait_for_everyone()
    if output_dir.exists():
        allowed_existing: set[Path] = set()
        if args.run_config:
            allowed_existing.add(Path(args.run_config).resolve())
        unexpected = sorted(
            str(item)
            for item in output_dir.iterdir()
            if item.resolve() not in allowed_existing
        )
        if unexpected:
            raise FileExistsError(
                "Training requires a fresh --output-dir (including when "
                "resuming) so metrics and checkpoints cannot be duplicated or "
                f"overwritten. Existing entries: {unexpected[:5]}"
            )
    accelerator.wait_for_everyone()
    metrics_path = output_dir / "metrics.jsonl"
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "run_args.json", vars(args))
        write_json(output_dir / "model_config.json", config.to_dict())
        write_json(output_dir / "hardware.json", hardware)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    if accelerator.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(accelerator.device)
    completed_steps = resumed_from_step
    local_micro_tokens = 0
    running_loss = torch.zeros((), device=accelerator.device, dtype=torch.float32)
    running_micro_batches = 0
    log_started = time.perf_counter()
    window_overhead_seconds = 0.0
    training_started = log_started
    last_eval: dict[str, float] = {}
    last_eval_step = -1
    last_train_metric: dict[str, Any] = {}
    cumulative_global_tokens = 0
    cumulative_active_seconds = 0.0
    skipped_optimizer_steps = 0
    throughput_warmup_remaining = args.throughput_warmup_steps
    last_observed_loss: Optional[torch.Tensor] = None
    training_flops_per_token = (
        flop_estimate["training_matmul_flops"]
        / (args.micro_batch_size * args.sequence_length)
    )

    while completed_steps < args.max_steps:
        if prefetched_batch is not None:
            batch = prefetched_batch
            prefetched_batch = None
        else:
            try:
                batch = next(train_iterator)
            except StopIteration:
                data_epoch += 1
                batches_in_epoch = 0
                set_dataloader_epoch(train_loader, data_epoch)
                train_iterator = iter(train_loader)
                try:
                    batch = next(train_iterator)
                except StopIteration as exc:
                    raise RuntimeError(
                        "The training dataset produced no full packed batch. "
                        "Add data, lower --sequence-length/--micro-batch-size, or "
                        "disable --drop-last in a custom loader."
                    ) from exc
        batches_in_epoch += 1
        consumed_micro_batches += 1

        with accelerator.accumulate(model):
            output = model(**batch)
            loss = output.loss
            accelerator.backward(loss)
            if accelerator.sync_gradients and args.max_grad_norm > 0:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        # Keep accumulation on device; calling float(loss) would synchronize
        # the GPU on every microbatch and materially damage CUDA throughput.
        running_loss.add_(loss.detach().float())
        last_observed_loss = loss.detach()
        running_micro_batches += 1
        local_micro_tokens += batch["input_ids"].numel()

        if not accelerator.sync_gradients:
            continue
        if accelerator.optimizer_step_was_skipped:
            # FP16 GradScaler overflow: the scheduler also stays put, so this
            # is not a completed optimizer update and must not advance the
            # checkpoint/eval/termination step counter.
            skipped_optimizer_steps += 1
            continue
        completed_steps += 1

        measurement_active = True
        if throughput_warmup_remaining > 0:
            throughput_warmup_remaining -= 1
            if accelerator.device.type == "cuda":
                torch.cuda.synchronize(accelerator.device)
            # Keep the optimizer update, but exclude compile/cache warmup from
            # every throughput and MFU aggregate.
            running_loss = torch.zeros(
                (), device=accelerator.device, dtype=torch.float32
            )
            running_micro_batches = 0
            local_micro_tokens = 0
            log_started = time.perf_counter()
            window_overhead_seconds = 0.0
            measurement_active = False

        if measurement_active and completed_steps % args.log_interval == 0:
            loss_tensor = (
                running_loss / max(running_micro_batches, 1)
            ).reshape(1)
            mean_loss = float(
                accelerator.gather_for_metrics(loss_tensor).float().mean().cpu()
            )
            # The CPU transfer above synchronizes outstanding device work.
            elapsed = max(
                time.perf_counter() - log_started - window_overhead_seconds,
                1e-9,
            )
            global_tokens = local_micro_tokens * accelerator.num_processes
            tokens_per_second = global_tokens / elapsed
            estimated_tflops = (
                tokens_per_second * training_flops_per_token / 1e12
            )
            metric = {
                "step": completed_steps,
                "train_loss": mean_loss,
                "learning_rate": scheduler.get_last_lr()[0],
                "tokens_per_second": tokens_per_second,
                "estimated_training_tflops": estimated_tflops,
                "estimated_dense_bf16_mfu": estimated_dense_bf16_mfu(
                    estimated_tflops,
                    args,
                    hardware,
                    accelerator.num_processes,
                ),
                "estimated_a100_bf16_mfu": estimated_a100_bf16_mfu(
                    estimated_tflops,
                    args,
                    hardware,
                    accelerator.num_processes,
                ),
                "elapsed_seconds": time.perf_counter() - training_started,
                "effective_tokens_per_update": (
                    args.micro_batch_size
                    * args.sequence_length
                    * args.gradient_accumulation_steps
                    * accelerator.num_processes
                ),
            }
            metric.update(
                cuda_memory_metrics(accelerator.device, args, hardware)
            )
            accelerator.print(json.dumps(metric, sort_keys=True))
            if accelerator.is_main_process:
                append_jsonl(metrics_path, metric)
            last_train_metric = metric
            cumulative_global_tokens += global_tokens
            cumulative_active_seconds += elapsed
            running_loss = torch.zeros(
                (), device=accelerator.device, dtype=torch.float32
            )
            running_micro_batches = 0
            local_micro_tokens = 0
            log_started = time.perf_counter()
            window_overhead_seconds = 0.0

        if args.eval_interval and completed_steps % args.eval_interval == 0:
            overhead_started = time.perf_counter()
            last_eval = evaluate(
                model, eval_loader, accelerator, args.eval_batches
            )
            last_eval_step = completed_steps
            metric = {"step": completed_steps, **last_eval}
            accelerator.print(json.dumps(metric, sort_keys=True))
            if accelerator.is_main_process:
                append_jsonl(metrics_path, metric)
            window_overhead_seconds += time.perf_counter() - overhead_started

        if args.save_interval and completed_steps % args.save_interval == 0:
            overhead_started = time.perf_counter()
            checkpoint_dir = output_dir / f"checkpoint-{completed_steps:08d}"
            accelerator.save_state(str(checkpoint_dir))
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                write_json(
                    checkpoint_dir / "training_progress.json",
                    {
                        "completed_steps": completed_steps,
                        "consumed_micro_batches": consumed_micro_batches,
                        "data_epoch": data_epoch,
                        "batches_in_epoch": batches_in_epoch,
                        "gradient_accumulation_steps": (
                            args.gradient_accumulation_steps
                        ),
                        "micro_batch_size": args.micro_batch_size,
                        "effective_tokens_per_update": (
                            args.micro_batch_size
                            * args.sequence_length
                            * args.gradient_accumulation_steps
                            * accelerator.num_processes
                        ),
                        "hardware_profile": args.hardware_profile,
                        "num_processes": accelerator.num_processes,
                        "data_signature": data_signature,
                        "resume_semantics_sha256": resume_semantics_sha256,
                    },
                )
            accelerator.wait_for_everyone()
            window_overhead_seconds += time.perf_counter() - overhead_started

    # Preserve a final partial logging window so short ablations still expose
    # comparable loss/throughput/TFLOPS/MFU in summary.json and the sweep CSV.
    if running_micro_batches:
        loss_tensor = (running_loss / running_micro_batches).reshape(1)
        mean_loss = float(
            accelerator.gather_for_metrics(loss_tensor).float().mean().cpu()
        )
        elapsed = max(
            time.perf_counter() - log_started - window_overhead_seconds,
            1e-9,
        )
        global_tokens = local_micro_tokens * accelerator.num_processes
        tokens_per_second = global_tokens / elapsed
        estimated_tflops = tokens_per_second * training_flops_per_token / 1e12
        last_train_metric = {
            "step": completed_steps,
            "train_loss": mean_loss,
            "learning_rate": scheduler.get_last_lr()[0],
            "tokens_per_second": tokens_per_second,
            "estimated_training_tflops": estimated_tflops,
            "estimated_dense_bf16_mfu": estimated_dense_bf16_mfu(
                estimated_tflops,
                args,
                hardware,
                accelerator.num_processes,
            ),
            "estimated_a100_bf16_mfu": estimated_a100_bf16_mfu(
                estimated_tflops,
                args,
                hardware,
                accelerator.num_processes,
            ),
            "elapsed_seconds": time.perf_counter() - training_started,
            "partial_window": True,
            "effective_tokens_per_update": (
                args.micro_batch_size
                * args.sequence_length
                * args.gradient_accumulation_steps
                * accelerator.num_processes
            ),
        }
        last_train_metric.update(
            cuda_memory_metrics(accelerator.device, args, hardware)
        )
        accelerator.print(json.dumps(last_train_metric, sort_keys=True))
        if accelerator.is_main_process:
            append_jsonl(metrics_path, last_train_metric)
        cumulative_global_tokens += global_tokens
        cumulative_active_seconds += elapsed

    if not last_train_metric and last_observed_loss is not None:
        # A one-step capacity probe may consist entirely of compile warmup. It
        # has no valid throughput window, but its loss should still be useful.
        warmup_loss = float(
            accelerator.gather_for_metrics(last_observed_loss.reshape(1))
            .float()
            .mean()
            .cpu()
        )
        last_train_metric = {
            "step": completed_steps,
            "train_loss": warmup_loss,
            "learning_rate": scheduler.get_last_lr()[0],
            "warmup_only": True,
        }
        accelerator.print(json.dumps(last_train_metric, sort_keys=True))
        if accelerator.is_main_process:
            append_jsonl(metrics_path, last_train_metric)

    accelerator.wait_for_everyone()
    if last_eval_step != completed_steps:
        last_eval = evaluate(model, eval_loader, accelerator, args.eval_batches)
        last_eval_step = completed_steps

    model_dir = output_dir / "model"
    if args.save_model or args.push_to_hub:
        unwrapped = unwrap_compiled_model(model, accelerator)
        unwrapped.save_pretrained(
            model_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            safe_serialization=True,
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(model_dir)

    elapsed_total = time.perf_counter() - training_started
    final_memory = cuda_memory_metrics(accelerator.device, args, hardware)
    aggregate_tflops = (
        cumulative_global_tokens
        / cumulative_active_seconds
        * training_flops_per_token
        / 1e12
        if cumulative_active_seconds > 0
        else math.nan
    )
    summary: dict[str, Any] = {
        **resolved_run_metadata(args, config),
        "status": "completed",
        "steps": completed_steps,
        "resumed_from_step": resumed_from_step,
        "consumed_micro_batches": consumed_micro_batches,
        "data_epoch": data_epoch,
        "batches_in_epoch": batches_in_epoch,
        "num_processes": accelerator.num_processes,
        "skipped_optimizer_steps": skipped_optimizer_steps,
        "elapsed_seconds": elapsed_total,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "hardware_profile": args.hardware_profile,
        "device_name": hardware["device_name"],
        "compute_capability": hardware["compute_capability"],
        "total_memory_gb": hardware["total_memory_gb"],
        "sdpa_backend_policy": hardware["sdpa_backend_policy"],
        "fused_sdpa_required": hardware["fused_sdpa_required"],
        "sdpa_backend_probe": hardware["sdpa_backend_probe"],
        "sdpa_probe_dtype": hardware["sdpa_probe_dtype"],
        "flash_sdpa_required": hardware["flash_sdpa_required"],
        "flash_sdpa_probe": hardware["flash_sdpa_probe"],
        "flash_sdpa_dtype": hardware["flash_sdpa_dtype"],
        "bf16_supported": hardware["bf16_supported"],
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_tokens_per_update": (
            args.micro_batch_size
            * args.sequence_length
            * args.gradient_accumulation_steps
            * accelerator.num_processes
        ),
        "throughput_warmup_steps": args.throughput_warmup_steps,
        "train_loss": last_train_metric.get("train_loss", math.nan),
        "learning_rate": last_train_metric.get("learning_rate", math.nan),
        "tokens_per_second": (
            cumulative_global_tokens / cumulative_active_seconds
            if cumulative_active_seconds > 0
            else math.nan
        ),
        "estimated_training_tflops": aggregate_tflops,
        "estimated_dense_bf16_mfu": estimated_dense_bf16_mfu(
            aggregate_tflops,
            args,
            hardware,
            accelerator.num_processes,
        ),
        "estimated_a100_bf16_mfu": estimated_a100_bf16_mfu(
            aggregate_tflops,
            args,
            hardware,
            accelerator.num_processes,
        ),
        **final_memory,
        **flop_estimate,
        **last_eval,
    }
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if not summary["within_memory_budget"]:
            warnings.warn(
                "Training exceeded the configured HBM budget: "
                f"peak reserved {summary['peak_reserved_gb']:.2f} GiB versus "
                f"budget {summary['memory_budget_gb']:.2f} GiB. Lower "
                "--micro-batch-size before comparing longer ablations.",
                stacklevel=2,
            )
        write_json(output_dir / "summary.json", summary)
        if args.push_to_hub:
            from huggingface_hub import HfApi

            api = HfApi()
            api.create_repo(
                repo_id=args.hub_model_id, repo_type="model", exist_ok=True
            )
            api.upload_folder(
                repo_id=args.hub_model_id,
                repo_type="model",
                folder_path=str(model_dir),
            )
    return summary


# ---------------------------------------------------------------------------
# Synthetic correctness and performance modes
# ---------------------------------------------------------------------------


def smoke_test(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    config = RecurrentAttnResConfig(
        vocab_size=101,
        hidden_size=64,
        num_attention_modules=2,
        num_ffn_modules=2,
        num_iterations=3,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=64,
        router_type="attnres",
        router_heads=1,
        readout_type="attnres",
        input_injection="every",
        share_iteration_weights=True,
        attention_dropout=0.0,
        residual_dropout=0.0,
        embedding_dropout=0.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = RecurrentAttnResForCausalLM(config)
    exact_parameters = count_parameters(model)[0]
    if estimate_parameter_count(config) != exact_parameters:
        raise AssertionError("Analytical parameter count does not match the model.")
    overridden_config = RecurrentAttnResConfig.from_dict(
        config.to_dict(), num_attention_modules=3, num_iterations=5
    )
    if (
        overridden_config.num_modules != 5
        or overridden_config.num_hidden_layers != 15
    ):
        raise AssertionError("Derived config fields became stale after override.")
    model.train()
    input_ids = torch.randint(3, config.vocab_size, (2, 16))
    output = model(input_ids=input_ids, labels=input_ids, output_router_stats=True)
    if output.logits.shape != (2, 16, config.vocab_size):
        raise AssertionError(f"Unexpected logits shape: {output.logits.shape}")
    if output.loss is None or not torch.isfinite(output.loss):
        raise AssertionError("Loss is missing or non-finite.")
    output.loss.backward()
    query_grad = model.shared_cell.mixer.pseudo_queries.grad
    if query_grad is None or not torch.isfinite(query_grad).all():
        raise AssertionError("AttnRes pseudo-query did not receive finite gradients.")

    # The 80 GB profile checkpoints recurrent chunks by default.  Verify that
    # both supported chunk granularities preserve forward values and gradients.
    baseline_grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    for checkpoint_chunk_steps in (1, 2):
        checkpoint_config = RecurrentAttnResConfig.from_dict(
            config.to_dict(), checkpoint_chunk_steps=checkpoint_chunk_steps
        )
        checkpoint_model = RecurrentAttnResForCausalLM(checkpoint_config)
        checkpoint_model.load_state_dict(model.state_dict())
        checkpoint_model.train()
        checkpoint_model.gradient_checkpointing_enable()
        checkpoint_output = checkpoint_model(input_ids=input_ids, labels=input_ids)
        checkpoint_output.loss.backward()
        torch.testing.assert_close(
            checkpoint_output.logits, output.logits, rtol=1e-5, atol=1e-6
        )
        for name, parameter in checkpoint_model.named_parameters():
            if name in baseline_grads:
                torch.testing.assert_close(
                    parameter.grad,
                    baseline_grads[name],
                    rtol=2e-5,
                    atol=2e-6,
                )

    # Causality: modifying only future tokens cannot change earlier logits.
    model.eval()
    probe = torch.randint(3, config.vocab_size, (1, 16))
    altered = probe.clone()
    cutoff = 7
    altered[:, cutoff + 1 :] = torch.randint(
        3, config.vocab_size, altered[:, cutoff + 1 :].shape
    )
    with torch.no_grad():
        unmasked_logits = model(input_ids=probe).logits
        original_logits = unmasked_logits[:, : cutoff + 1]
        altered_logits = model(input_ids=altered).logits[:, : cutoff + 1]
        all_ones_logits = model(
            input_ids=probe, attention_mask=torch.ones_like(probe)
        ).logits
    torch.testing.assert_close(original_logits, altered_logits, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        unmasked_logits,
        all_ones_logits,
        rtol=1e-5,
        atol=1e-6,
    )

    # A heterogeneous left-padding pattern catches accidental [A,B] versus
    # [B,A] mask-folding errors in the packed attention kernel.
    padded_ids = torch.randint(3, config.vocab_size, (2, 12))
    padded_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )
    with torch.no_grad():
        padded_batch_logits = model(
            input_ids=padded_ids, attention_mask=padded_mask
        ).logits
        for sample in range(2):
            single_logits = model(
                input_ids=padded_ids[sample : sample + 1],
                attention_mask=padded_mask[sample : sample + 1],
            ).logits
            valid = padded_mask[sample].bool()
            torch.testing.assert_close(
                padded_batch_logits[sample, valid],
                single_logits[0, valid],
                rtol=1e-5,
                atol=1e-6,
            )

    # Exercise the important structural ablation paths, including zero modules
    # of one type and untied iteration weights.
    variant_specs = [
        (1, 0, "identity", 1, False),
        (0, 2, "static", 1, False),
        (1, 1, "shared_attnres", 4, True),
        (1, 1, "uniform", 1, True),
    ]
    for attn_count, ffn_count, router, router_heads, tied in variant_specs:
        variant_config = RecurrentAttnResConfig(
            vocab_size=101,
            hidden_size=64,
            num_attention_modules=attn_count,
            num_ffn_modules=ffn_count,
            num_iterations=2,
            num_attention_heads=4,
            intermediate_size=128,
            max_position_embeddings=32,
            router_type=router,
            router_heads=router_heads,
            readout_type="attnres",
            share_iteration_weights=tied,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
        variant = RecurrentAttnResForCausalLM(variant_config).eval()
        if count_parameters(variant)[0] != estimate_parameter_count(variant_config):
            raise AssertionError("Variant analytical parameter count is incorrect.")
        with torch.no_grad():
            variant_logits = variant(input_ids=probe[:, :8]).logits
        if variant_logits.shape != (1, 8, 101) or not torch.isfinite(
            variant_logits
        ).all():
            raise AssertionError(
                f"Structural variant failed: A={attn_count}, F={ffn_count}, "
                f"router={router}, tied={tied}."
            )

    exclusion_config = RecurrentAttnResConfig(
        vocab_size=101,
        hidden_size=4,
        num_attention_modules=1,
        num_ffn_modules=1,
        num_iterations=2,
        num_attention_heads=1,
        intermediate_size=8,
        router_type="uniform",
        exclude_self=True,
    )
    exclusion_mixer = CrossModuleMixer(
        exclusion_config,
        num_destinations=2,
        mixer_type="uniform",
        allow_identity=True,
    )
    exclusion_embedding = torch.zeros(1, 1, 4)
    exclusion_states = torch.stack(
        (torch.ones(1, 1, 4), torch.full((1, 1, 4), 3.0))
    )
    exclusion_output, _ = exclusion_mixer(
        exclusion_embedding,
        exclusion_states,
        include_embedding=True,
    )
    torch.testing.assert_close(
        exclusion_output[:, 0, 0, 0], torch.tensor([1.5, 0.5])
    )

    # Tensor-Core-friendly vocabulary padding must never make synthetic rows
    # trainable targets or generation candidates.
    padded_vocab_config = RecurrentAttnResConfig(
        vocab_size=104,
        valid_vocab_size=101,
        hidden_size=64,
        num_attention_modules=1,
        num_ffn_modules=1,
        num_iterations=2,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=32,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    padded_vocab_model = RecurrentAttnResForCausalLM(padded_vocab_config)
    padded_vocab_ids = torch.randint(3, 101, (1, 8))
    padded_vocab_output = padded_vocab_model(
        input_ids=padded_vocab_ids, labels=padded_vocab_ids
    )
    if padded_vocab_output.loss is None or not torch.isfinite(
        padded_vocab_output.loss
    ):
        raise AssertionError("Padded-vocabulary loss is missing or non-finite.")
    padded_vocab_output.loss.backward()
    invalid_logits = padded_vocab_output.logits[..., 101:]
    if not torch.equal(
        invalid_logits,
        torch.full_like(invalid_logits, torch.finfo(invalid_logits.dtype).min),
    ):
        raise AssertionError("Synthetic vocabulary rows were not masked.")
    if int(padded_vocab_output.logits.argmax(dim=-1).max()) >= 101:
        raise AssertionError("A synthetic vocabulary row won argmax.")

    resize_model = RecurrentAttnResForCausalLM(
        RecurrentAttnResConfig(
            vocab_size=101,
            hidden_size=32,
            num_attention_modules=1,
            num_ffn_modules=1,
            num_iterations=1,
            num_attention_heads=4,
            intermediate_size=64,
        )
    )
    resize_model.resize_token_embeddings(
        105, pad_to_multiple_of=8, mean_resizing=False
    )
    if (
        resize_model.config.vocab_size != 112
        or resize_model.config.valid_vocab_size != 105
    ):
        raise AssertionError("Padded vocabulary growth metadata is incorrect.")
    resize_model.resize_token_embeddings(97, mean_resizing=False)
    if (
        resize_model.config.vocab_size != 97
        or resize_model.config.valid_vocab_size != 97
    ):
        raise AssertionError("Vocabulary shrink metadata is incorrect.")

    # Hugging Face save/load round trip.
    with tempfile.TemporaryDirectory(prefix="recurrent_mesh_smoke_") as temp_dir:
        model.save_pretrained(temp_dir, safe_serialization=True)
        reloaded = RecurrentAttnResForCausalLM.from_pretrained(temp_dir)
        reloaded.eval()
        with torch.no_grad():
            reload_logits = reloaded(input_ids=probe).logits
        torch.testing.assert_close(
            model(input_ids=probe).logits, reload_logits, rtol=1e-5, atol=1e-6
        )
        generation = reloaded.generate(
            probe[:, :4],
            max_new_tokens=2,
            do_sample=False,
            use_cache=False,
            pad_token_id=config.pad_token_id,
            return_dict_in_generate=True,
            output_attentions=True,
        )
        generated = generation.sequences
        if generated.shape[1] < 4:
            raise AssertionError("Generation unexpectedly shortened the prompt.")
        fresh_auto = subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                (
                    "import sys; "
                    "from transformers import AutoModelForCausalLM; "
                    "m=AutoModelForCausalLM.from_pretrained("
                    "sys.argv[1], trust_remote_code=True, local_files_only=True); "
                    "assert m.config.model_type == 'recurrent_attnres_mesh'"
                ),
                temp_dir,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if fresh_auto.returncode != 0:
            raise AssertionError(
                "Fresh-process AutoModel load failed:\n"
                + fresh_auto.stdout
                + fresh_auto.stderr
            )

    result = {
        "status": "passed",
        "loss": float(output.loss.detach()),
        "logits_shape": list(output.logits.shape),
        "causality_max_abs_error": float(
            (original_logits - altered_logits).abs().max()
        ),
        "padding_batch_equivalence": "passed",
        "all_ones_mask_fast_path": "passed",
        "checkpoint_equivalence": "passed",
        "analytical_parameter_count": exact_parameters,
        "padded_vocabulary_masking": "passed",
        "vocabulary_resize": "passed",
        "fresh_process_auto_load": "passed",
        "uniform_self_exclusion": "passed",
        "derived_config_overrides": "passed",
        "structural_variants_tested": len(variant_specs),
        "router_entropies": [
            float(value.detach()) for value in (output.router_entropies or ())
        ],
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    configure_torch(args.tf32)
    device = select_execution_device()
    config = build_config(args)
    hardware = validate_execution_hardware(args, device, config=config)
    seed_everything(args.seed)
    model = RecurrentAttnResForCausalLM(config).to(device)
    autocast_dtype: Optional[torch.dtype] = None
    if device.type == "cuda" and args.mixed_precision != "no":
        autocast_dtype = (
            torch.bfloat16
            if args.mixed_precision == "bf16"
            else torch.float16
        )
    scaler: Optional[Any] = None
    if device.type == "cuda" and args.mixed_precision == "fp16":
        scaler = torch.amp.GradScaler("cuda")
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()
    optimizer = make_optimizer(model, args, device)
    if args.compile_model:
        model = torch.compile(
            model, mode=args.compile_mode, dynamic=False, fullgraph=False
        )

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (args.micro_batch_size, args.sequence_length),
        device=device,
    )

    def training_step() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device=device, dtype=torch.float32)
        for _ in range(args.gradient_accumulation_steps):
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=autocast_dtype)
                if autocast_dtype is not None
                else contextlib.nullcontext()
            )
            with autocast_context:
                loss = model(input_ids=input_ids, labels=input_ids).loss
                scaled_loss = loss / args.gradient_accumulation_steps
            loss_sum.add_(loss.detach().float())
            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
        if args.max_grad_norm > 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        return loss_sum / args.gradient_accumulation_steps

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(args.benchmark_warmup):
        training_step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    warmup_memory = cuda_memory_metrics(device, args, hardware)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    final_loss = None
    for _ in range(args.benchmark_steps):
        final_loss = training_step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    tokens = (
        args.benchmark_steps
        * args.micro_batch_size
        * args.sequence_length
        * args.gradient_accumulation_steps
    )
    estimates = estimate_forward_flops(
        config,
        args.micro_batch_size,
        args.sequence_length,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    achieved_tflops = (
        estimates["training_matmul_flops"]
        * args.benchmark_steps
        * args.gradient_accumulation_steps
        / max(elapsed, 1e-9)
        / 1e12
    )
    total, trainable = count_parameters(
        getattr(model, "_orig_mod", model)
    )
    if total != estimate_parameter_count(config):
        raise AssertionError("Analytical parameter count does not match benchmark model.")
    steady_memory = cuda_memory_metrics(device, args, hardware)
    capacity_peak_allocated = max(
        warmup_memory["peak_allocated_gb"], steady_memory["peak_allocated_gb"]
    )
    capacity_peak_reserved = max(
        warmup_memory["peak_reserved_gb"], steady_memory["peak_reserved_gb"]
    )
    memory_budget = float(hardware.get("memory_budget_gb", 0.0))
    result = {
        **resolved_run_metadata(args, config),
        "device": str(device),
        "hardware_profile": args.hardware_profile,
        "device_name": hardware["device_name"],
        "compute_capability": hardware["compute_capability"],
        "total_memory_gb": hardware["total_memory_gb"],
        "sdpa_backend_policy": hardware["sdpa_backend_policy"],
        "fused_sdpa_required": hardware["fused_sdpa_required"],
        "sdpa_backend_probe": hardware["sdpa_backend_probe"],
        "sdpa_probe_dtype": hardware["sdpa_probe_dtype"],
        "flash_sdpa_required": hardware["flash_sdpa_required"],
        "flash_sdpa_probe": hardware["flash_sdpa_probe"],
        "flash_sdpa_dtype": hardware["flash_sdpa_dtype"],
        "bf16_supported": hardware["bf16_supported"],
        "parameter_dtype": str(next(model.parameters()).dtype),
        "autocast_dtype": str(autocast_dtype) if autocast_dtype else "disabled",
        "elapsed_seconds": elapsed,
        "training_tokens_per_second": tokens / max(elapsed, 1e-9),
        "estimated_training_tflops": achieved_tflops,
        "estimated_dense_bf16_mfu": estimated_dense_bf16_mfu(
            achieved_tflops, args, hardware
        ),
        "estimated_a100_bf16_mfu": estimated_a100_bf16_mfu(
            achieved_tflops, args, hardware
        ),
        "micro_batch_size": args.micro_batch_size,
        "sequence_length": args.sequence_length,
        "effective_tokens_per_update": (
            args.micro_batch_size
            * args.sequence_length
            * args.gradient_accumulation_steps
        ),
        "loss": float(final_loss) if final_loss is not None else math.nan,
        "total_parameters": total,
        "trainable_parameters": trainable,
        **steady_memory,
        "steady_peak_allocated_gb": steady_memory["peak_allocated_gb"],
        "steady_peak_reserved_gb": steady_memory["peak_reserved_gb"],
        "warmup_peak_allocated_gb": warmup_memory["peak_allocated_gb"],
        "warmup_peak_reserved_gb": warmup_memory["peak_reserved_gb"],
        "capacity_peak_allocated_gb": capacity_peak_allocated,
        "capacity_peak_reserved_gb": capacity_peak_reserved,
        "peak_allocated_gb": capacity_peak_allocated,
        "peak_reserved_gb": capacity_peak_reserved,
        "peak_memory_gb": capacity_peak_allocated,
        "peak_reserved_fraction": (
            capacity_peak_reserved / max(hardware["total_memory_gb"], 1e-9)
            if device.type == "cuda"
            else 0.0
        ),
        "steady_state_within_memory_budget": steady_memory[
            "within_memory_budget"
        ],
        "within_memory_budget": (
            device.type != "cuda" or capacity_peak_reserved <= memory_budget
        ),
        **estimates,
    }
    if not result["within_memory_budget"]:
        warnings.warn(
            "Benchmark exceeded the configured HBM budget: "
            f"peak reserved {result['capacity_peak_reserved_gb']:.2f} GiB versus "
            f"budget {result['memory_budget_gb']:.2f} GiB. Lower "
            "--micro-batch-size for the ablation sweep.",
            stacklevel=2,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output_dir:
        write_json(Path(args.output_dir) / "benchmark.json", result)
    return result


# ---------------------------------------------------------------------------
# Sweep runner and command-line interface
# ---------------------------------------------------------------------------


def comma_ints(value: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integers.") from exc
    if not result:
        raise argparse.ArgumentTypeError("The list cannot be empty.")
    return result


def comma_strings(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("The list cannot be empty.")
    return result


def comma_tied_flags(value: str) -> list[bool]:
    mapping = {
        "tied": True,
        "true": True,
        "1": True,
        "untied": False,
        "false": False,
        "0": False,
    }
    result: list[bool] = []
    for item in comma_strings(value):
        if item.lower() not in mapping:
            raise argparse.ArgumentTypeError(
                "Use tied/untied (or true/false) for sweep weight sharing."
            )
        result.append(mapping[item.lower()])
    return result


def run_sweep(args: argparse.Namespace) -> list[dict[str, Any]]:
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise RuntimeError(
            "Launch --mode sweep with plain Python, not from inside a distributed job."
        )
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    sweep_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"_{os.getpid()}"
    sweep_root = output_root / f"sweep_{sweep_id}"
    sweep_root.mkdir(parents=True, exist_ok=False)
    combinations = list(
        itertools.product(
            args.sweep_attention_modules,
            args.sweep_ffn_modules,
            args.sweep_iterations,
            args.sweep_router_types,
            args.sweep_router_heads,
            args.sweep_share_iteration_weights,
        )
    )
    combinations = [item for item in combinations if item[0] + item[1] > 0]
    if args.sweep_max_runs > 0:
        combinations = combinations[: args.sweep_max_runs]
    print(f"Prepared {len(combinations)} sweep runs.")

    script_path = Path(__file__).resolve()
    launcher = shlex.split(args.sweep_launcher) if args.sweep_launcher else [
        sys.executable
    ]
    rows: list[dict[str, Any]] = []
    csv_path = sweep_root / "sweep_results.csv"

    for run_index, (attn, ffn, steps, router, router_heads, tied) in enumerate(
        combinations, start=1
    ):
        sharing = "tied" if tied else "untied"
        run_name = (
            f"a{attn}_f{ffn}_s{steps}_{router}_rh{router_heads}_{sharing}"
        )
        # Every invocation gets a fresh sweep root.  This prevents a failed
        # rerun from being misreported using a stale summary/metrics file.
        run_dir = sweep_root / run_name
        run_values = dict(vars(args))
        run_values.update(
            {
                "mode": "train",
                "run_config": None,
                "output_dir": str(run_dir),
                "num_attention_modules": attn,
                "num_ffn_modules": ffn,
                "num_iterations": steps,
                "router_type": router,
                "router_heads": router_heads,
                "share_iteration_weights": tied,
                "save_model": args.sweep_save_models,
                "push_to_hub": False,
            }
        )
        config_path = run_dir / "sweep_run.json"
        write_json(config_path, run_values)
        command = [
            *launcher,
            str(script_path),
            "--mode",
            "train",
            "--run-config",
            str(config_path),
        ]
        print(
            f"[{run_index}/{len(combinations)}] {run_name}\n"
            f"  {shlex.join(command)}"
        )
        if args.sweep_dry_run:
            rows.append(
                {
                    "run": run_name,
                    "status": "dry_run",
                    "hardware_profile": args.hardware_profile,
                    "num_attention_modules": attn,
                    "num_ffn_modules": ffn,
                    "num_iterations": steps,
                    "router_type": router,
                    "router_heads": router_heads,
                    "share_iteration_weights": tied,
                    "micro_batch_size": args.micro_batch_size,
                    "gradient_accumulation_steps": (
                        args.gradient_accumulation_steps
                    ),
                    "effective_tokens_per_update": (
                        args.micro_batch_size
                        * args.sequence_length
                        * args.gradient_accumulation_steps
                    ),
                }
            )
            continue

        completed = subprocess.run(command, check=False)
        summary_path = run_dir / "summary.json"
        failure_path = run_dir / "failure.json"
        if completed.returncode == 0 and summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
        elif failure_path.exists():
            with failure_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            summary.setdefault("return_code", completed.returncode)
        else:
            summary = {
                "status": "failed",
                "return_code": completed.returncode,
                "failure": (
                    "child_failed"
                    if completed.returncode != 0
                    else "missing_summary"
                ),
            }
        row = {
            "run": run_name,
            "num_attention_modules": attn,
            "num_ffn_modules": ffn,
            "num_iterations": steps,
            "router_type": router,
            "router_heads": router_heads,
            "share_iteration_weights": tied,
            **summary,
        }
        rows.append(row)

        # Rewrite after every run so an interrupted sweep retains a complete index.
        fieldnames = sorted({key for item in rows for key in item})
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        child_oom = summary.get("status") == "oom"
        should_stop = (
            child_oom and args.sweep_stop_on_oom
        ) or (
            not child_oom and completed.returncode != 0 and args.sweep_stop_on_error
        )
        if should_stop:
            raise RuntimeError(f"Sweep run failed: {run_name}")

    if args.sweep_dry_run:
        write_json(sweep_root / "sweep_dry_run.json", rows)
        if rows:
            fieldnames = sorted({key for item in rows for key in item})
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
    print(f"Sweep index: {csv_path}")
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train and ablate a recurrent, synchronous all-module "
            "Attention-Residuals mesh."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=("train", "benchmark", "inspect", "smoke", "sweep"),
        default="train",
    )
    parser.add_argument(
        "--run-config",
        type=str,
        default="",
        help="JSON namespace used internally by the isolated sweep subprocesses.",
    )
    parser.add_argument(
        "--hardware-profile",
        choices=tuple(HARDWARE_PROFILES),
        default="a100-80gb",
        help=(
            "Default bundle; explicit CLI and sweep run-config values take "
            "precedence. Strict SKU profiles and generic CUDA/CPU profiles "
            "are available."
        ),
    )

    model = parser.add_argument_group("model")
    model.add_argument("--hidden-size", type=int, default=None)
    model.add_argument("--num-attention-modules", type=int, default=None)
    model.add_argument("--num-ffn-modules", type=int, default=None)
    model.add_argument("--num-iterations", type=int, default=None)
    model.add_argument("--num-attention-heads", type=int, default=None)
    model.add_argument("--intermediate-size", type=int, default=None)
    model.add_argument("--max-position-embeddings", type=int, default=None)
    model.add_argument("--rope-theta", type=float, default=10_000.0)
    model.add_argument(
        "--router-type",
        choices=("attnres", "shared_attnres", "static", "uniform", "identity"),
        default="attnres",
    )
    model.add_argument("--router-heads", type=int, default=1)
    model.add_argument(
        "--readout-type",
        choices=("attnres", "static", "uniform"),
        default="attnres",
    )
    model.add_argument(
        "--input-injection",
        choices=("every", "initial_only"),
        default="every",
    )
    model.add_argument(
        "--share-iteration-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    model.add_argument(
        "--use-step-embeddings",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    model.add_argument(
        "--branch-output", choices=("delta", "residual"), default="delta"
    )
    model.add_argument(
        "--exclude-self",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    model.add_argument("--attention-dropout", type=float, default=0.0)
    model.add_argument("--residual-dropout", type=float, default=0.0)
    model.add_argument("--embedding-dropout", type=float, default=0.0)
    model.add_argument("--initializer-range", type=float, default=0.02)
    model.add_argument("--rms-norm-eps", type=float, default=1e-6)
    model.add_argument(
        "--tie-word-embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    model.add_argument("--vocab-size", type=int, default=0)
    model.add_argument("--pad-vocab-multiple", type=int, default=128)
    model.add_argument("--load-model", type=str, default="")

    data = parser.add_argument_group("Hugging Face data")
    data.add_argument("--dataset", type=str, default="roneneldan/TinyStories")
    data.add_argument("--dataset-config", type=str, default="")
    data.add_argument(
        "--dataset-revision",
        type=str,
        default="",
        help="Optional immutable Hub dataset revision/commit for reproducible runs.",
    )
    data.add_argument("--train-split", type=str, default="train")
    data.add_argument("--eval-split", type=str, default="validation")
    data.add_argument("--text-column", type=str, default="text")
    data.add_argument("--train-file", type=str, default="")
    data.add_argument("--eval-file", type=str, default="")
    data.add_argument(
        "--tokenizer",
        type=str,
        default="",
        help="Tokenizer ID/path; defaults to --load-model, otherwise gpt2.",
    )
    data.add_argument("--tokenizer-revision", type=str, default="")
    data.add_argument(
        "--streaming", action=argparse.BooleanOptionalAction, default=True
    )
    data.add_argument("--shuffle-buffer", type=int, default=10_000)
    data.add_argument("--max-dataset-examples", type=int, default=0)
    data.add_argument("--sequence-length", type=int, default=None)
    data.add_argument("--num-workers", type=int, default=None)
    data.add_argument("--prefetch-factor", type=int, default=None)
    data.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    training = parser.add_argument_group("training / device execution")
    training.add_argument("--output-dir", type=str, default="runs/recurrent_mesh")
    training.add_argument("--max-steps", type=int, default=1000)
    training.add_argument("--micro-batch-size", type=int, default=None)
    training.add_argument(
        "--gradient-accumulation-steps", type=int, default=None
    )
    training.add_argument("--learning-rate", type=float, default=3e-4)
    training.add_argument("--weight-decay", type=float, default=0.1)
    training.add_argument("--adam-beta1", type=float, default=0.9)
    training.add_argument("--adam-beta2", type=float, default=0.95)
    training.add_argument("--adam-epsilon", type=float, default=1e-8)
    training.add_argument("--warmup-steps", type=int, default=100)
    training.add_argument("--max-grad-norm", type=float, default=1.0)
    training.add_argument(
        "--mixed-precision", choices=("no", "fp16", "bf16"), default=None
    )
    training.add_argument(
        "--tf32", action=argparse.BooleanOptionalAction, default=None
    )
    training.add_argument(
        "--compile",
        dest="compile_model",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    training.add_argument(
        "--compile-mode",
        choices=(
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ),
        default=None,
    )
    training.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    training.add_argument("--checkpoint-chunk-steps", type=int, default=None)
    training.add_argument(
        "--sdpa-backend",
        choices=("auto", "fused", "cudnn", "flash"),
        default=None,
        help=(
            "SDPA policy. 'fused' allows cuDNN+Flash only (no math fallback); "
            "'auto' lets PyTorch choose any available backend."
        ),
    )
    training.add_argument(
        "--require-fused-sdpa",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Require a CUDA fused-SDPA path and preflight it."
        ),
    )
    training.add_argument(
        "--require-flash",
        dest="legacy_require_flash",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Legacy compatibility spelling: force flash-only SDPA when true; "
            "prefer --sdpa-backend and --require-fused-sdpa in new commands."
        ),
    )
    training.add_argument(
        "--fused-adamw",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    training.add_argument(
        "--strict-hardware-profile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Fail before model allocation if the profile's device contract fails.",
    )
    training.add_argument(
        "--allocator-expandable-segments",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable expandable CUDA allocator segments before CUDA initializes.",
    )
    training.add_argument(
        "--target-peak-memory-fraction", type=float, default=None
    )
    training.add_argument("--memory-reserve-gb", type=float, default=None)
    training.add_argument("--throughput-warmup-steps", type=int, default=None)
    training.add_argument(
        "--bf16-peak-tflops",
        "--a100-bf16-peak-tflops",
        dest="bf16_peak_tflops",
        type=float,
        default=None,
        help=(
            "Dense BF16 peak used for estimated MFU; zero means unavailable. "
            "The A100-prefixed spelling is a compatibility alias."
        ),
    )
    training.add_argument("--log-interval", type=int, default=10)
    training.add_argument("--eval-interval", type=int, default=100)
    training.add_argument("--eval-batches", type=int, default=20)
    training.add_argument("--save-interval", type=int, default=0)
    training.add_argument(
        "--save-model", action=argparse.BooleanOptionalAction, default=True
    )
    training.add_argument("--resume-state", type=str, default="")
    training.add_argument("--seed", type=int, default=42)
    training.add_argument(
        "--push-to-hub", action=argparse.BooleanOptionalAction, default=False
    )
    training.add_argument("--hub-model-id", type=str, default="")

    bench = parser.add_argument_group("benchmark")
    bench.add_argument("--benchmark-warmup", type=int, default=3)
    bench.add_argument("--benchmark-steps", type=int, default=10)

    sweep = parser.add_argument_group("sweep")
    sweep.add_argument(
        "--sweep-attention-modules", type=comma_ints, default=[1, 2, 4]
    )
    sweep.add_argument("--sweep-ffn-modules", type=comma_ints, default=[1, 2, 4])
    sweep.add_argument(
        "--sweep-iterations", type=comma_ints, default=[1, 2, 4, 6, 8]
    )
    sweep.add_argument(
        "--sweep-router-types",
        type=comma_strings,
        default=["attnres"],
    )
    sweep.add_argument("--sweep-router-heads", type=comma_ints, default=[1])
    sweep.add_argument(
        "--sweep-share-iteration-weights",
        type=comma_tied_flags,
        default=[True],
    )
    sweep.add_argument(
        "--sweep-save-models",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    sweep.add_argument(
        "--sweep-dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    sweep.add_argument(
        "--sweep-stop-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    sweep.add_argument(
        "--sweep-stop-on-oom",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Abort rather than record and continue after a child CUDA OOM.",
    )
    sweep.add_argument("--sweep-max-runs", type=int, default=0)
    sweep.add_argument(
        "--sweep-launcher",
        type=str,
        default="",
        help=(
            "Optional command prefix, e.g. "
            "'accelerate launch --num_processes 1'."
        ),
    )
    return parser


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.run_config:
        dispatch_mode = args.mode
        requested_config_path = Path(args.run_config).resolve()
        with requested_config_path.open("r", encoding="utf-8") as handle:
            values = json.load(handle)
        # Accept run configs emitted by the original A100-only release.
        legacy_peak = values.pop("a100_bf16_peak_tflops", None)
        if legacy_peak is not None and "bf16_peak_tflops" not in values:
            values["bf16_peak_tflops"] = legacy_peak
        legacy_flash = values.pop("require_flash", None)
        if legacy_flash is not None and "legacy_require_flash" not in values:
            values["legacy_require_flash"] = legacy_flash
        unknown = sorted(set(values) - set(vars(args)))
        if unknown:
            raise ValueError(f"Unknown keys in run config: {unknown}")
        for key, value in values.items():
            setattr(args, key, value)
        # The sweep launches train explicitly, while an explicit inspect mode
        # can cheaply verify a child config without allocating its model.
        args.mode = dispatch_mode
        args.run_config = str(requested_config_path)

    # Preserve the original Flash-only switch exactly while keeping it
    # separate from the modern "any fused backend" requirement. An explicit
    # backend always wins over legacy false; legacy true specifically means
    # Flash and therefore rejects a contradictory explicit backend.
    legacy_require_flash = args.legacy_require_flash
    backend_was_explicit = args.sdpa_backend is not None
    fused_requirement_was_explicit = args.require_fused_sdpa is not None
    if legacy_require_flash is True:
        if backend_was_explicit and args.sdpa_backend != "flash":
            parser.error(
                "--require-flash conflicts with explicit "
                f"--sdpa-backend={args.sdpa_backend}; use flash or remove one."
            )
        if fused_requirement_was_explicit and not args.require_fused_sdpa:
            parser.error(
                "--require-flash conflicts with --no-require-fused-sdpa."
            )
        args.sdpa_backend = "flash"
        args.require_fused_sdpa = True
    elif legacy_require_flash is False:
        if not backend_was_explicit:
            # Prevent a strict Flash profile default from undoing the legacy
            # opt-out. Do not weaken an explicitly selected backend.
            args.sdpa_backend = "auto"
        if not fused_requirement_was_explicit:
            args.require_fused_sdpa = False
    delattr(args, "legacy_require_flash")

    resolve_hardware_profile(args)
    # Requiring a fused path with an automatic policy means "any fused CUDA
    # kernel, but never math". The separate legacy --require-flash path above
    # has already resolved to the stricter flash-only policy.
    if args.require_fused_sdpa and args.sdpa_backend == "auto":
        args.sdpa_backend = "fused"
    if args.sequence_length < 2:
        parser.error("--sequence-length must be at least 2.")
    if args.hidden_size < 1:
        parser.error("--hidden-size must be positive.")
    if args.num_attention_heads < 1:
        parser.error("--num-attention-heads must be positive.")
    if args.hidden_size % args.num_attention_heads:
        parser.error("--hidden-size must be divisible by --num-attention-heads.")
    if args.num_attention_modules < 0 or args.num_ffn_modules < 0:
        parser.error("Module counts cannot be negative.")
    if args.num_attention_modules + args.num_ffn_modules < 1:
        parser.error("At least one attention or FFN module is required.")
    if args.num_iterations < 1:
        parser.error("--num-iterations must be positive.")
    if args.micro_batch_size < 1:
        parser.error("--micro-batch-size must be positive.")
    if args.gradient_accumulation_steps < 1:
        parser.error("--gradient-accumulation-steps must be positive.")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative.")
    if args.prefetch_factor < 1:
        parser.error("--prefetch-factor must be positive.")
    if args.checkpoint_chunk_steps < 1:
        parser.error("--checkpoint-chunk-steps must be positive.")
    if not 0.0 < args.target_peak_memory_fraction <= 1.0:
        parser.error("--target-peak-memory-fraction must be in (0, 1].")
    if args.memory_reserve_gb < 0:
        parser.error("--memory-reserve-gb cannot be negative.")
    if args.throughput_warmup_steps < 0:
        parser.error("--throughput-warmup-steps cannot be negative.")
    if args.bf16_peak_tflops < 0:
        parser.error("--bf16-peak-tflops cannot be negative.")
    if args.max_steps < 1:
        parser.error("--max-steps must be positive.")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps cannot be negative.")
    if args.benchmark_warmup < 0:
        parser.error("--benchmark-warmup cannot be negative.")
    if args.benchmark_steps < 1:
        parser.error("--benchmark-steps must be positive.")
    if args.eval_batches < 1:
        parser.error("--eval-batches must be positive.")
    if args.eval_interval < 0 or args.save_interval < 0:
        parser.error("--eval-interval and --save-interval cannot be negative.")
    if args.sweep_max_runs < 0:
        parser.error("--sweep-max-runs cannot be negative.")
    if args.log_interval < 1:
        parser.error("--log-interval must be positive.")
    if args.router_heads < 1:
        parser.error("--router-heads must be positive.")
    if args.push_to_hub and not args.hub_model_id:
        parser.error("--hub-model-id is required with --push-to-hub.")
    return args


def find_cuda_oom(exception: BaseException) -> Optional[BaseException]:
    """Find a typed CUDA OOM, including one wrapped by torch.compile."""
    pending: list[BaseException] = [exception]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        if isinstance(current, torch.OutOfMemoryError):
            return current
        for nested in (current.__cause__, current.__context__):
            if isinstance(nested, BaseException):
                pending.append(nested)
    return None


def record_cuda_oom(args: argparse.Namespace, exception: BaseException) -> None:
    device = select_execution_device()
    failure: dict[str, Any] = {
        "status": "oom",
        "exit_code": CUDA_OOM_EXIT_CODE,
        "error_type": type(exception).__name__,
        "error": str(exception),
        "hardware_profile": args.hardware_profile,
        "sdpa_backend_policy": args.sdpa_backend,
        "fused_sdpa_required": args.require_fused_sdpa,
        "num_attention_modules": args.num_attention_modules,
        "num_ffn_modules": args.num_ffn_modules,
        "num_iterations": args.num_iterations,
        "share_iteration_weights": args.share_iteration_weights,
        "hidden_size": args.hidden_size,
        "intermediate_size": args.intermediate_size,
        "sequence_length": args.sequence_length,
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "recommended_micro_batch_size": max(1, args.micro_batch_size // 2),
    }
    if device.type == "cuda":
        try:
            properties = torch.cuda.get_device_properties(device)
            hardware = {
                "memory_budget_gb": min(
                    properties.total_memory / 2**30
                    * args.target_peak_memory_fraction,
                    max(0.0, properties.total_memory / 2**30 - args.memory_reserve_gb),
                )
            }
            failure["device_name"] = properties.name
            failure.update(cuda_memory_metrics(device, args, hardware))
        except Exception as telemetry_error:
            failure["telemetry_error"] = repr(telemetry_error)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "failure.json", failure)
    print(json.dumps(failure, sort_keys=True), file=sys.stderr)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    configure_runtime_environment(args)
    try:
        if args.mode == "smoke":
            smoke_test(args)
        elif args.mode == "inspect":
            inspect_configuration(args)
        elif args.mode == "benchmark":
            benchmark(args)
        elif args.mode == "sweep":
            run_sweep(args)
        else:
            train(args)
    except BaseException as exc:
        oom = find_cuda_oom(exc)
        if oom is None:
            raise
        record_cuda_oom(args, oom)
        raise SystemExit(CUDA_OOM_EXIT_CODE) from exc


if __name__ == "__main__":
    main()
