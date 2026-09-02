#!/usr/bin/env python3
"""Pre-train matched Hugging Face causal LMs for residual-routing ablations.

The three default experiments differ only in how decoder sublayer outputs are
combined: ordinary additive residuals, Full Attention Residuals with softmax,
and Full Attention Residuals with shifted ReLU-squared normalization.

This is deliberately a single-file experiment so it can be copied to a
training machine without importing nanoGPT's training driver.  It uses a
Hugging Face ``PreTrainedModel`` and datasets/tokenizers, but keeps the model
small by default so that the comparison can be smoke-tested before a long run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

# Keep ``python huggingface_model/attention_residual_pretrain.py`` runnable.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_variations.muon import SingleDeviceMuonWithAuxAdam


class AttentionResidualConfig(PretrainedConfig):
    """Configuration carrying nanoGPT's default architectural choices."""

    model_type = "nanogpt-attention-residual"

    def __init__(
        self,
        vocab_size: int = 50304,
        block_size: int = 512,
        n_layer: int = 6,
        n_head: int = 6,
        n_embd: int = 384,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        rope_base: float = 10000.0,
        qk_norm_scale_init: Optional[float] = None,
        residual_variant: str = "standard",
        residual_activation: str = "softmax",
        relu2_shift: float = 1e-3,
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        if block_size < 2:
            raise ValueError("block_size must be at least 2")
        if n_embd % n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if residual_variant not in {"standard", "full"}:
            raise ValueError("residual_variant must be 'standard' or 'full'")
        if residual_activation not in {"softmax", "relu2max_shift"}:
            raise ValueError("unsupported residual_activation")
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.rms_norm_eps = rms_norm_eps
        self.rope_base = rope_base
        self.qk_norm_scale_init = qk_norm_scale_init
        self.residual_variant = residual_variant
        self.residual_activation = residual_activation
        self.relu2_shift = relu2_shift
        self.use_cache = False


class RMSNorm(nn.Module):
    """RMSNorm with a learned per-channel gain."""

    def __init__(self, width: int, eps: float):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.gain


def apply_rope(x: torch.Tensor, base: float) -> torch.Tensor:
    """Apply RoPE to ``(batch, heads, time, head_dim)`` activations."""

    rotary_dim = x.size(-1) - x.size(-1) % 2
    positions = torch.arange(x.size(-2), device=x.device, dtype=torch.float32)
    inv_freq = base ** (-torch.arange(0, rotary_dim, 2, device=x.device, dtype=torch.float32) / rotary_dim)
    angles = torch.outer(positions, inv_freq)
    cos, sin = angles.cos()[None, None], angles.sin()[None, None]
    first, rest = x[..., :rotary_dim], x[..., rotary_dim:]
    even, odd = first[..., 0::2], first[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)
    return torch.cat((rotated.to(x.dtype), rest), dim=-1)


class CausalSelfAttention(nn.Module):
    """Causal softmax attention with RoPE, QK norm, and learned QK scale."""

    def __init__(self, config: AttentionResidualConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.rope_base = config.rope_base
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        # Match nanoGPT's learned QK-normalization scale initialization.
        initial_scale = config.qk_norm_scale_init
        if initial_scale is None:
            initial_scale = math.log2(config.block_size * config.block_size - config.block_size)
        self.qk_norm_scale = nn.Parameter(torch.tensor(float(initial_scale)))
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, channels = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (batch, time, self.n_head, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        q = F.normalize(apply_rope(q, self.rope_base), dim=-1, eps=1e-6)
        k = F.normalize(apply_rope(k, self.rope_base), dim=-1, eps=1e-6)
        scores = (q @ k.transpose(-2, -1)) * self.qk_norm_scale
        causal = torch.ones(time, time, dtype=torch.bool, device=x.device).tril()
        scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores.float(), dim=-1).to(v.dtype)
        weights = F.dropout(weights, self.dropout, self.training)
        out = (weights @ v).transpose(1, 2).contiguous().view(batch, time, channels)
        return self.proj(out)


class SquaredReLUMLP(nn.Module):
    def __init__(self, config: AttentionResidualConfig):
        super().__init__()
        hidden = config.mlp_ratio * config.n_embd
        self.up = nn.Linear(config.n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.relu(self.up(x)).square())


class DecoderBlock(nn.Module):
    """Sequential pre-norm decoder block; residual combination lives above."""

    def __init__(self, config: AttentionResidualConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd, config.rms_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config.n_embd, config.rms_norm_eps)
        self.mlp = SquaredReLUMLP(config)

    def standard_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        return x + self.mlp(self.mlp_norm(x))


class FullAttentionResidual(nn.Module):
    """Token-local routing over depth, matching nanoGPT's full residual path."""

    def __init__(self, destinations: int, width: int, activation: str, shift: float, eps: float):
        super().__init__()
        self.queries = nn.Parameter(torch.zeros(destinations, width))
        self.activation = activation
        self.shift = shift
        self.eps = eps

    def weights(self, scores: torch.Tensor) -> torch.Tensor:
        if self.activation == "softmax":
            return scores.float().softmax(dim=0).to(scores.dtype)
        # Translation makes every source eligible while the positive shift
        # prevents an exactly-zero denominator at initialization.
        shifted = scores - scores.amin(dim=0, keepdim=True) + self.shift
        numer = F.relu(shifted).square()
        return numer / numer.sum(dim=0, keepdim=True).clamp_min(self.eps)

    def forward(self, sources: list[torch.Tensor], destination: int) -> torch.Tensor:
        if not sources:
            raise ValueError("attention residuals require at least one source")
        values = torch.stack(sources)
        keys = F.rms_norm(values, (values.size(-1),), eps=self.eps)
        scores = torch.einsum("dbtc,c->dbt", keys, self.queries[destination])
        return torch.einsum("dbt,dbtc->btc", self.weights(scores), values)


class AttentionResidualForCausalLM(PreTrainedModel):
    config_class = AttentionResidualConfig
    base_model_prefix = "transformer"
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: AttentionResidualConfig):
        super().__init__(config)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)])
        self.final_norm = RMSNorm(config.n_embd, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.residual_router = None
        if config.residual_variant == "full":
            self.residual_router = FullAttentionResidual(
                2 * config.n_layer + 1,
                config.n_embd,
                config.residual_activation,
                config.relu2_shift,
                config.rms_norm_eps,
            )
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.gain)

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, value):
        self.wte = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def forward(self, input_ids, labels=None, **_) -> CausalLMOutput:
        if input_ids.size(1) > self.config.block_size:
            raise ValueError("input is longer than block_size")
        x = self.wte(input_ids)
        if self.residual_router is None:
            for block in self.blocks:
                x = block.standard_forward(x)
        else:
            sources = [x]
            for layer, block in enumerate(self.blocks):
                attn_input = self.residual_router(sources, 2 * layer)
                sources.append(block.attn(block.attn_norm(attn_input)))
                mlp_input = self.residual_router(sources, 2 * layer + 1)
                sources.append(block.mlp(block.mlp_norm(mlp_input)))
            x = self.residual_router(sources, 2 * self.config.n_layer)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))
        return CausalLMOutput(loss=loss, logits=logits)


def make_muon_optimizer(
    model: nn.Module,
    adamw_lr: float,
    weight_decay: float,
    muon_lr: Optional[float] = None,
):
    """Use Muon only for decoder matrices and AdamW for everything else.

    In particular, token embeddings, the LM head (whether tied or untied),
    RMSNorm gains, QK scales, and residual-routing queries stay on AdamW.
    """

    adam_names = {"wte.weight", "lm_head.weight"}
    muon_lr = adamw_lr if muon_lr is None else muon_lr
    seen: set[int] = set()
    muon, adam = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        is_gain = name.endswith(".gain")
        is_decoder_matrix = name.startswith("blocks.") and name.endswith(".weight") and parameter.ndim == 2
        if is_decoder_matrix and name not in adam_names and not is_gain:
            muon.append(parameter)
        else:
            adam.append(parameter)
    groups = [
        {"params": adam, "use_muon": False, "lr": adamw_lr, "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": weight_decay},
        {"params": muon, "use_muon": True, "lr": muon_lr, "momentum": 0.95, "ns_steps": 5, "nesterov": True, "weight_decay": weight_decay},
    ]
    return SingleDeviceMuonWithAuxAdam(groups)


def _token_blocks(dataset, tokenizer, block_size: int, text_column: str):
    def tokenize(batch):
        return tokenizer(batch[text_column], add_special_tokens=False)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

    def group(batch):
        joined = sum(batch["input_ids"], [])
        usable = len(joined) // block_size * block_size
        blocks = [joined[i : i + block_size] for i in range(0, usable, block_size)]
        return {"input_ids": blocks, "labels": [row[:] for row in blocks]}

    # ``tokenizer`` also returns row-aligned columns such as attention_mask.
    # Grouping changes the number of rows from documents to fixed-size token
    # blocks, so Arrow cannot retain those original columns alongside the new
    # blocks. Recreate the output table from the grouped columns only.
    return tokenized.map(
        group,
        batched=True,
        remove_columns=tokenized.column_names,
        desc="Grouping tokens into fixed-length blocks",
    )


@torch.no_grad()
def evaluate(model, loader, device, max_batches: int) -> float:
    model.eval()
    losses = []
    for step, batch in enumerate(loader):
        if step >= max_batches:
            break
        ids = batch["input_ids"].to(device)
        losses.append(model(input_ids=ids, labels=ids).loss.float())
    return torch.stack(losses).mean().item()


def train_one(name, config, train_loader, val_loader, args, device):
    torch.manual_seed(args.seed)
    if train_loader.generator is not None:
        # Every ablation receives exactly the same shuffled example order.
        train_loader.generator.manual_seed(args.seed)
    model = AttentionResidualForCausalLM(config).to(device)
    optimizer = make_muon_optimizer(
        model,
        adamw_lr=args.learning_rate,
        weight_decay=args.weight_decay,
        muon_lr=args.muon_learning_rate,
    )
    model.train()
    iterator = iter(train_loader)
    history = []
    for step in range(1, args.max_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        ids = batch["input_ids"].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if step % args.eval_interval == 0 or step == args.max_steps:
            val_loss = evaluate(model, val_loader, device, args.eval_batches)
            history.append({"model": name, "step": step, "train_loss": loss.item(), "validation_loss": val_loss})
            print(f"{name}: step {step}: train={loss.item():.4f}, validation={val_loss:.4f}")
            model.train()
    model.save_pretrained(Path(args.output_dir) / name)
    return history


def summarize_history(rows):
    """Return one comparable final/best-validation record per experiment."""

    by_model = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)
    if "standard_residual" not in by_model:
        raise ValueError("summary requires the standard_residual baseline")
    baseline = by_model["standard_residual"][-1]["validation_loss"]
    summary = []
    for model, model_rows in by_model.items():
        final = model_rows[-1]
        final_loss = final["validation_loss"]
        summary.append(
            {
                "model": model,
                "final_step": final["step"],
                "final_validation_loss": final_loss,
                "best_validation_loss": min(row["validation_loss"] for row in model_rows),
                "delta_vs_standard": final_loss - baseline,
                "relative_improvement_vs_standard_pct": 100.0 * (baseline - final_loss) / baseline,
            }
        )
    return sorted(summary, key=lambda row: row["final_validation_loss"])


def write_summary(rows, output: Path):
    summary = summarize_history(rows)
    with (output / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print("\nFinal validation-loss summary (lower is better)")
    print(f"{'model':42} {'final':>10} {'best':>10} {'delta':>10} {'improvement':>12}")
    for row in summary:
        print(
            f"{row['model']:42} {row['final_validation_loss']:10.4f} "
            f"{row['best_validation_loss']:10.4f} {row['delta_vs_standard']:+10.4f} "
            f"{row['relative_improvement_vs_standard_pct']:+11.2f}%"
        )


def parse_args(argv: Optional[Iterable[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="roneneldan/TinyStories")
    parser.add_argument("--dataset-config")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--output-dir", default="results/attention_residual_comparison")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=6)
    parser.add_argument("--n-embd", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--muon-learning-rate", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--relu2-shift", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None):
    from datasets import load_dataset

    args = parse_args(argv)
    random.seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    raw = load_dataset(args.dataset, args.dataset_config)
    validation_key = "validation" if "validation" in raw else "test"
    train_data = _token_blocks(raw["train"], tokenizer, args.block_size, args.text_column)
    val_data = _token_blocks(raw[validation_key], tokenizer, args.block_size, args.text_column)
    train_data.set_format("torch")
    val_data.set_format("torch")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)
    common = dict(vocab_size=len(tokenizer), block_size=args.block_size, n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd)
    experiments = {
        "standard_residual": AttentionResidualConfig(**common, residual_variant="standard"),
        "attention_residual_softmax": AttentionResidualConfig(**common, residual_variant="full", residual_activation="softmax"),
        "attention_residual_relu2max_shift": AttentionResidualConfig(
            **common, residual_variant="full", residual_activation="relu2max_shift", relu2_shift=args.relu2_shift
        ),
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, config in experiments.items():
        rows.extend(train_one(name, config, train_loader, val_loader, args, torch.device(args.device)))
    with (output / "validation_losses.json").open("w") as handle:
        json.dump(rows, handle, indent=2)
    with (output / "validation_losses.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "step", "train_loss", "validation_loss"])
        writer.writeheader()
        writer.writerows(rows)
    write_summary(rows, output)


if __name__ == "__main__":
    main()
