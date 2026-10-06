"""Validation, old/new head interventions, token-frequency buckets, and head geometry."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model_utils import autocast, hidden
from prepare_data import TokenBlocks


@torch.no_grad()
def score_states(h, labels, w, b, chunk, rare_mask=None):
    h, labels = h.reshape(-1, w.shape[1]), labels.reshape(-1)
    loss, correct, margin = 0.0, 0, 0.0
    rare_loss, rare_count = 0.0, 0
    for i in range(0, len(labels), chunk):
        y = labels[i:i + chunk]
        logits = F.linear(h[i:i + chunk], w, b).float()
        nll = F.cross_entropy(logits, y, reduction="none")
        loss += nll.double().sum().item()
        correct += (logits.argmax(-1) == y).sum().item()
        truth = logits.gather(1, y[:, None]).squeeze(1)
        logits.scatter_(1, y[:, None], -float("inf"))
        margin += (truth - logits.amax(-1)).double().sum().item()
        if rare_mask is not None:
            mask = rare_mask[y]
            rare_loss += nll[mask].double().sum().item()
            rare_count += mask.sum().item()
    return {"nll_sum": loss, "tokens": len(labels), "correct": correct,
            "margin_sum": margin, "a_common_b_rare_nll_sum": rare_loss,
            "a_common_b_rare_tokens": rare_count}


def summarize(stats):
    n = stats["tokens"]
    loss = stats["nll_sum"] / n
    return {**stats, "loss": loss, "perplexity": math.exp(min(loss, 700)),
            "token_accuracy": stats["correct"] / n, "mean_target_margin": stats["margin_sum"] / n,
            "a_common_b_rare_loss": (stats["a_common_b_rare_nll_sum"] / stats["a_common_b_rare_tokens"]
                                      if stats["a_common_b_rare_tokens"] else None)}


def rare_tokens(root, vocab, device):
    a = np.load(Path(root) / "a_train.counts.npy")
    b = np.load(Path(root) / "b_train.counts.npy")
    out = np.zeros(vocab, dtype=bool)
    size = min(len(a), len(b), vocab)
    out[:size] = (a[:size] >= 20) & ((b[:size] / max(1, b.sum())) < 0.1 * (a[:size] / max(1, a.sum())))
    return torch.from_numpy(out).to(device)


@torch.no_grad()
def evaluate(m, blocks, c, device, rare_mask=None):
    m.eval()
    accum = None
    for start in range(0, len(blocks), c["micro_batch"]):
        batch = blocks.batch(list(range(start, min(start + c["micro_batch"], len(blocks)))), device)
        with autocast(device):
            h = hidden(m, batch)
            head = m.get_output_embeddings()
            result = score_states(h, batch[:, 1:], head.weight, head.bias, c["head_chunk"], rare_mask)
        if accum is None:
            accum = result
        else:
            for k, v in result.items():
                accum[k] += v
    return summarize(accum)


@torch.no_grad()
def make_reference(m, blocks, c, device, path):
    m.eval()
    n = min(c["probe_blocks"], len(blocks))
    states, labels = [], []
    for i in range(n):
        batch = blocks.batch([i], device)
        with autocast(device):
            states.append(hidden(m, batch).cpu())
        labels.append(batch[:, 1:].cpu())
    head = m.get_output_embeddings()
    ref = {"head": head.weight.detach().cpu().clone(),
           "bias": head.bias.detach().cpu().clone() if head.bias is not None else None,
           "states": torch.cat(states), "labels": torch.cat(labels), "probe_blocks": n}
    # Only local self-produced trusted checkpoints are loaded with weights_only=False.
    torch.save(ref, path)


@torch.no_grad()
def head_geometry(m, ref, count, seed):
    w0 = ref["head"]
    g = torch.Generator().manual_seed(seed)
    ids = torch.randperm(len(w0), generator=g)[:min(count, len(w0))]
    wt = m.get_output_embeddings().weight[ids.to(m.device)].detach().float().cpu()
    old = w0[ids].float()
    cosine = F.cosine_similarity(old, wt, dim=-1).clamp(-1, 1)
    angle = torch.rad2deg(torch.acos(cosine))
    ratio = wt.norm(dim=-1) / old.norm(dim=-1).clamp_min(1e-12)
    # Exact equality supersedes acos rounding noise for frozen/unchanged rows.
    unchanged = (old == wt).all(-1)
    angle[unchanged] = 0
    return {"sampled_rows": len(ids), "mean_angle_degrees": angle.mean().item(),
            "p95_angle_degrees": torch.quantile(angle, 0.95).item(),
            "mean_norm_ratio": ratio.mean().item(),
            "relative_frobenius_drift": ((wt - old).norm() / old.norm().clamp_min(1e-12)).item()}


@torch.no_grad()
def head_swap(m, blocks, ref, c, device):
    """One backbone on the GPU. The original head is transferred only for this diagnostic."""
    m.eval()
    w0 = ref["head"].to(device)
    b0 = ref["bias"].to(device) if ref["bias"] is not None else None
    head = m.get_output_embeddings()
    totals = {k: 0.0 for k in ("w0_h0", "wt_h0", "w0_ht", "wt_ht")}
    tokens = 0
    for i in range(ref["probe_blocks"]):
        batch = blocks.batch([i], device)
        h0 = ref["states"][i:i + 1].to(device)
        y = batch[:, 1:]
        if not torch.equal(y.cpu(), ref["labels"][i:i + 1]):
            raise RuntimeError("Probe contexts changed since the anchor was saved")
        with autocast(device):
            ht = hidden(m, batch)
            for key, h, w, b in (("w0_h0", h0, w0, b0), ("wt_h0", h0, head.weight, head.bias),
                                  ("w0_ht", ht, w0, b0), ("wt_ht", ht, head.weight, head.bias)):
                totals[key] += score_states(h, y, w, b, c["head_chunk"])["nll_sum"]
        tokens += y.numel()
    result = {k: v / tokens for k, v in totals.items()}
    result["loss_interaction"] = result["wt_ht"] - result["wt_h0"] - result["w0_ht"] + result["w0_h0"]
    result["tokens"] = tokens
    return result


def data_blocks(c, domain, split):
    return TokenBlocks(Path(c["output"]) / "data" / f"{domain}_{split}.bin", c["seq_len"])
