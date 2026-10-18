#!/usr/bin/env python3
"""Read-only checkpoint evaluation for the ComparisonForCausalLM experiment.

Uses the original forward path and reconstructs only sampled attention rows in
FP32. Reconstructed weights are diagnostics, not measurements of kernel error.
No optimizer, training-state pickle, model save, or source modification is used.
"""
import argparse
from collections import defaultdict
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else default


def checkpoint_path(path):
    path = Path(path).resolve()
    if (path / "config.json").is_file():
        return path
    latest = read(path / "latest.json")
    if latest:
        saved = (path / latest["checkpoint"]).resolve()
        if saved.parent != path or not (saved / "config.json").is_file():
            raise ValueError("latest.json does not identify an existing checkpoint in this run")
        return saved
    raise ValueError(f"Expected a checkpoint directory or a run with latest.json: {path}")


def position_bins(length):
    ends = sorted({x for x in (64, 128, 256, 512, 1024, 2048, 4096, length) if x <= length})
    start = 0
    for end in ends:
        yield start, end
        start = end


def weight_statistics(q, k, scale, config, positions, bias=None):
    """No padding/cache in this probe; positions are zero-based causal queries."""
    if q.shape != k.shape or q.ndim != 4:
        raise ValueError("Probe requires equal-length, uncached full multi-head attention")
    with torch.autocast(device_type=q.device.type, enabled=False):
        picked = q[:, :, positions, :]
        # The SDPA path casts scaled Q before QK, whereas the custom kernels
        # multiply scale into the accumulated QK scores.
        if config.variant == "softmax_sdpa":
            scores = (picked * scale).to(q.dtype).float() @ k.float().transpose(-1, -2)
        else:
            scores = (picked.float() @ k.float().transpose(-1, -2)) * scale.float()
        if bias is not None:
            scores = scores - bias.detach().float()[None, :, None, None]
        cols = torch.arange(k.shape[-2], device=q.device)
        keep = cols[None, :] <= positions[:, None]
        if config.variant == "softmax_sdpa":
            weights = scores.masked_fill(~keep, -torch.inf).softmax(-1)
        else:
            positive = scores.relu()
            if config.variant == "relu2" or config.variant.startswith("relu2_"):
                weights = positive.square()
            else:
                threshold = config.effective_threshold
                clipped = positive.clamp(max=threshold)
                weights = clipped.square() + 2 * threshold * (positive - clipped)
            divisor = config.relu2max_divisor
            if config.relu2max_divide_by_sequence_length:
                divisor *= k.shape[-2]
            weights = weights.masked_fill(~keep, 0) / divisor
            alpha = getattr(config, "causal_length_alpha", 0.0)
            anchor = getattr(config, "causal_length_anchor", 512.0)
            if alpha:
                weights = weights * ((positions + 1).float() / anchor).clamp_min(1).pow(-alpha)[None, None, :, None]
        if not torch.isfinite(weights).all():
            raise FloatingPointError("Nonfinite reconstructed attention weights")
        mass = weights.sum(-1)
        energy = weights.square().sum(-1)
        tiny = torch.finfo(weights.dtype).tiny
        share = weights / mass.clamp_min(tiny).unsqueeze(-1)
        recent = cols[None, :] > positions[:, None] - 512
        stats = {
            "row_mass": mass,
            "weight_energy": energy,
            "effective_key_count": mass.square() / energy.clamp_min(tiny),
            "active_key_count": (weights > 0).sum(-1).float(),
            "largest_weight_share": share.max(-1).values,
            "recent_512_weight_share": (share * recent).sum(-1),
            "score_above_16_fraction": ((scores > 16) & keep).sum(-1).float() / (positions + 1),
        }
        return {name: value.mean(0).cpu() for name, value in stats.items()}


class Probe:
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.attention = defaultdict(lambda: defaultdict(float))
        self.norms = defaultdict(lambda: defaultdict(float))
        self.scales = {}
        self.calls = defaultdict(int)
        for layer_index, block in enumerate(model.model.layers):
            if not hasattr(block.attn, "diagnostic_callback"):
                raise ValueError("This probe supports the full-attention arms, not KDA")
            if block.attn.diagnostic_callback is not None:
                raise ValueError("An attention diagnostic callback is already installed")
            block.attn.diagnostic_callback = self.callback(layer_index)
            for name in ("attn_output_norm", "ffn_output_norm"):
                module = getattr(block, name, None)
                if module is not None:
                    self.handles.append(module.register_forward_hook(self.norm_hook(layer_index, name)))

    def callback(self, layer):
        def record(q, k, scale, past):
            if past:
                raise ValueError("This diagnostic evaluates full sequences without cache")
            indices = sorted({x - 1 for x in (1, 64, 128, 256, 512, 1024, 2048, 4096, q.shape[-2]) if x <= q.shape[-2]})
            positions = torch.tensor(indices, device=q.device)
            scale = torch.as_tensor(scale, device=q.device).detach().float()
            bias = getattr(self.model.model.layers[layer].attn, "subtractive_bias", None)
            stats = weight_statistics(q, k, scale, self.model.config, positions, bias)
            self.scales[layer] = float(scale)
            self.calls[layer] += 1
            for name, values in stats.items():
                for head in range(values.shape[0]):
                    for index, pos in enumerate(indices):
                        self.attention[layer, head, pos + 1][name] += float(values[head, index])
        return record

    def norm_hook(self, layer, name):
        def record(module, args, output):
            x = args[0].detach().float()
            y = output.detach().float()
            r = x.square().mean(-1).sqrt()
            yr = y.square().mean(-1).sqrt()
            for start, end in position_bins(x.shape[1]):
                values = self.norms[layer, name, start + 1, end]
                values["count"] += r[:, start:end].numel()
                values["input_rms_sum"] += float(r[:, start:end].sum())
                values["output_rms_sum"] += float(yr[:, start:end].sum())
                values["above_radius"] += float((r[:, start:end] > 1).sum())
        return record

    def close(self):
        for block in self.model.model.layers:
            block.attn.diagnostic_callback = None
        for handle in self.handles:
            handle.remove()

    def results(self):
        attn = [dict(layer=l, head=h, visible_keys=n, scale=self.scales[l],
                     subtractive_bias=float(self.model.model.layers[l].attn.subtractive_bias[h].detach()) if hasattr(self.model.model.layers[l].attn, "subtractive_bias") else 0.0,
                     length_alpha=getattr(self.model.config, "causal_length_alpha", 0.0),
                     length_anchor=getattr(self.model.config, "causal_length_anchor", 512.0),
                     sampled_sequences=self.calls[l], **{k: v / self.calls[l] for k, v in row.items()})
                for (l, h, n), row in sorted(self.attention.items())]
        norms = [dict(layer=l, branch=name, first_position=a, last_position=b,
                      vectors=int(row["count"]), mean_input_rms=row["input_rms_sum"] / row["count"],
                      mean_output_rms=row["output_rms_sum"] / row["count"],
                      fraction_above_radius=row["above_radius"] / row["count"])
                 for (l, name, a, b), row in sorted(self.norms.items())]
        return dict(attention=attn, branch_norms=norms)


@torch.inference_mode()
def evaluate(model, stream, length, tokens, device, dtype):
    probe = Probe(model)
    loss_sum = torch.zeros(length, device=device, dtype=torch.float64)
    count = tokens // length
    autocast = lambda: torch.autocast("cuda", dtype=torch.bfloat16) if str(device).startswith("cuda") and dtype == "bfloat16" else nullcontext()
    try:
        for index in range(count):
            # Exactly the same token prefix and shifted labels for every length.
            x, y = stream.batch(index * length, 1, length, device)
            with autocast():
                hidden = model.model(x, use_cache=False).last_hidden_state
                for start in range(0, length, 256):
                    end = min(length, start + 256)
                    logits = model.lm_head(hidden[:, start:end]).float()
                    ce = F.cross_entropy(logits.reshape(-1, model.config.vocab_size), y[:, start:end].reshape(-1), reduction="none")
                    loss_sum[start:end] += ce.double()
        if not torch.isfinite(loss_sum).all():
            raise FloatingPointError("Nonfinite validation loss")
        loss_bins = [dict(first_position=a + 1, last_position=b, targets=count * (b - a),
                          loss=float(loss_sum[a:b].sum() / (count * (b - a))))
                     for a, b in position_bins(length)]
        return dict(context=length, evaluated_tokens=tokens, loss=float(loss_sum.sum() / tokens),
                    loss_by_position=loss_bins, **probe.results())
    finally:
        probe.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd(), help="Root containing hf_model/ and experiment/")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Saved checkpoint, final/, or a run with latest.json")
    parser.add_argument("--data", type=Path, required=True, help="Prepared data directory with manifest.json and validation.bin")
    parser.add_argument("--contexts", nargs="+", type=int, default=[256, 512, 1024, 2048])
    parser.add_argument("--tokens", type=int, default=32768, help="Evaluation targets, not training tokens")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--sdpa-backend", choices=["flash", "auto", "math"], default="flash")
    parser.add_argument("--output", type=Path, required=True, help="Fresh output directory")
    args = parser.parse_args()
    if args.tokens < 1 or not args.contexts or len(set(args.contexts)) != len(args.contexts) or any(n < 2 or args.tokens % n for n in args.contexts):
        parser.error("Use unique contexts >= 2 that each divide --tokens exactly")
    if args.output.exists():
        parser.error("Choose a fresh --output directory to preserve previous diagnostics")
    project = args.project.resolve()
    if not (project / "hf_model/modeling_comparison.py").is_file():
        parser.error("--project must contain hf_model/modeling_comparison.py")
    sys.path.insert(0, str(project))
    from hf_model.modeling_comparison import ComparisonForCausalLM
    from experiment.common import TokenStream, configure_device
    saved = checkpoint_path(args.checkpoint)
    config = read(saved / "config.json")
    from hf_model.context_attention import CONTEXT_ARMS
    if config.get("variant") not in ("softmax_sdpa", "relu2", "linear16", *CONTEXT_ARMS):
        parser.error("Choose a softmax, ReLU2, context-ablation or linear16 checkpoint; KDA has no attention rows")
    if max(args.contexts) > config.get("max_position_embeddings", 0):
        parser.error("Requested context exceeds checkpoint max_position_embeddings")
    run_info = read(saved.parent / "run.json", {})
    manifest_hash = hashlib.sha256((args.data / "manifest.json").read_bytes()).hexdigest()
    if run_info.get("data_manifest_sha256") and run_info["data_manifest_sha256"] != manifest_hash:
        parser.error("Data manifest does not match this checkpoint's training run")
    stream = TokenStream(args.data, "validation")
    if len(stream.tokens) < args.tokens + 1:
        parser.error("Insufficient validation tokens")
    configure_device(args.device, args.dtype)
    # FP32 parameters plus autocast match the experiment's validation protocol.
    model = ComparisonForCausalLM.from_pretrained(saved).float().to(args.device).eval()
    model.config.sdpa_backend = args.sdpa_backend if args.device.startswith("cuda") else "math"
    model.config.require_fused = args.device.startswith("cuda")
    model.config.relu2max_accelerator = "triton" if args.device.startswith("cuda") else "torch"
    progress = read(saved / "progress.json", {})
    summary = read(saved.parent / "train_summary.json", {})
    result = dict(checkpoint=str(saved), variant=model.config.variant,
                  output_norm=getattr(model.config, "output_norm", "none"), checkpoint_config=config,
                  training_context=run_info.get("config", {}).get("train", {}).get("sequence_length"),
                  training_tokens=progress.get("cursor", summary.get("tokens")),
                  training_step=progress.get("step", summary.get("steps")),
                  data_manifest_sha256=manifest_hash, dtype=args.dtype, device=args.device,
                  torch_version=torch.__version__, sdpa_backend=model.config.sdpa_backend,
                  source_sha256={str(p.relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted((project / "hf_model").glob("*.py"))},
                  note="Read-only validation. Native packing at each length uses identical target tokens. "
                       "Attention weights are FP32 reconstructions of sampled rows, not captured fused weights. "
                       "Effective key count = (sum w)^2 / sum(w^2); zero rows report zero. "
                       "Probing a context beyond training length tests extrapolation. Timings include diagnostics and are not throughput benchmarks.",
                  evaluations=[])
    args.output.mkdir(parents=True)
    for length in sorted(args.contexts):
        started = time.monotonic()
        row = evaluate(model, stream, length, args.tokens, args.device, args.dtype)
        row["probe_seconds"] = time.monotonic() - started
        result["evaluations"].append(row)
        temp = args.output / "diagnostics.json.tmp"
        temp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        temp.replace(args.output / "diagnostics.json")
        print(json.dumps({k: row[k] for k in ("context", "evaluated_tokens", "loss", "probe_seconds")}), flush=True)
    print(f"Saved {args.output / 'diagnostics.json'}", flush=True)


if __name__ == "__main__":
    main()
