# train_variations/loss_variants.py
"""Collection of loss functions and scheduling utilities.

This module provides a dictionary mapping loss names to callables. Each
loss takes ``logits`` and ``targets`` tensors and returns a scalar loss.
Optionally the current training iteration ``iter_num`` can be supplied
for schedulers or adaptive losses.

The default loss is standard cross entropy, but additional options are
provided that more strongly encourage correct top-1 predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple

import math
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Individual loss implementations
# ---------------------------------------------------------------------------

def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor, *, iter_num: int | None = None) -> torch.Tensor:
    """Standard cross-entropy loss used by the original codebase."""
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)


def label_smoothing_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    iter_num: int | None = None,
    smoothing: float = 0.1,
) -> torch.Tensor:
    """Cross entropy with label smoothing to prevent overconfidence."""
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=-1,
        label_smoothing=smoothing,
    )


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    iter_num: int | None = None,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal loss from classification literature to focus on hard examples."""
    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = targets.view(-1)
    ce = F.cross_entropy(logits_flat, targets_flat, reduction="none", ignore_index=-1)
    with torch.no_grad():
        pt = torch.exp(-ce)
    loss = ((1 - pt) ** gamma) * ce
    return loss[targets_flat != -1].mean()


def top1_focus_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    iter_num: int | None = None,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Cross entropy with an extra penalty for wrong top-1 predictions."""
    ce = cross_entropy_loss(logits, targets)
    top1 = torch.argmax(logits, dim=-1)
    correct_top1 = (top1 == targets).float()
    penalty = 1.0 - correct_top1
    return ce + alpha * penalty.mean()


def skip_correct_top1_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    iter_num: int | None = None,
) -> torch.Tensor:
    """Ignore examples that are already predicted correctly at top-1."""

    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = targets.view(-1)
    losses = F.cross_entropy(logits_flat, targets_flat, reduction="none", ignore_index=-1)

    with torch.no_grad():
        predictions = torch.argmax(logits_flat, dim=-1)
        mask = (targets_flat != -1) & (predictions != targets_flat)

    if mask.any():
        return losses[mask].mean()

    # If every token is already correct we skip the loss entirely while
    # preserving device/dtype for downstream consumers expecting a tensor.
    return losses.new_full((), 0.0)


def attenuated_correct_top1_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    iter_num: int | None = None,
    attenuation: float = 1.0,
) -> torch.Tensor:
    """Down-weight correctly predicted tokens by a constant factor.

    When ``attenuation`` is ``1.0`` the behaviour exactly matches
    cross-entropy. Setting ``attenuation`` below ``1.0`` decreases the
    contribution from tokens that already have the correct top-1
    prediction while keeping the gradient signal non-zero.
    """

    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = targets.view(-1)
    losses = F.cross_entropy(logits_flat, targets_flat, reduction="none", ignore_index=-1)
    mask = targets_flat != -1

    if not mask.any():
        return losses.new_full((), 0.0)

    with torch.no_grad():
        predictions = torch.argmax(logits_flat, dim=-1)
        correct = (predictions == targets_flat) & mask
        scale = torch.ones_like(losses)
        scale[correct] = attenuation

    scaled = losses * scale
    return scaled[mask].mean()


def distance_attenuated_top1_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    iter_num: int | None = None,
    strength: float = 0.0,
) -> torch.Tensor:
    """Attenuate loss based on how close the target is to the top prediction.

    ``strength`` controls how aggressively the attenuation is applied. A
    value of ``0.0`` makes this identical to cross-entropy. As the
    strength increases, tokens that are already near the top prediction
    receive a reduced loss while badly ranked targets remain close to the
    standard cross-entropy loss.
    """

    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = targets.view(-1)
    losses = F.cross_entropy(logits_flat, targets_flat, reduction="none", ignore_index=-1)
    mask = targets_flat != -1

    if not mask.any():
        return losses.new_full((), 0.0)

    with torch.no_grad():
        logits_sel = logits_flat[mask]
        targets_sel = targets_flat[mask]
        top_logits, _ = logits_sel.max(dim=-1)
        target_logits = logits_sel[torch.arange(logits_sel.size(0)), targets_sel]
        # ``distance`` is zero when the target already matches the top
        # prediction, and grows with the logit gap otherwise.
        distance = torch.clamp(top_logits - target_logits, min=0.0)
        attenuation = 1.0 - strength * torch.exp(-distance)
        attenuation = torch.clamp(attenuation, min=0.0)

    scale = torch.ones_like(losses)
    scale[mask] = attenuation
    scaled = losses * scale
    return scaled[mask].mean()


def top1_margin_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    iter_num: int | None = None,
    margin: float = 0.1,
) -> torch.Tensor:
    """Encourage the target logit to exceed others by a margin."""
    ce = cross_entropy_loss(logits, targets)
    b, t, v = logits.shape
    logits_flat = logits.view(-1, v)
    targets_flat = targets.view(-1)
    target_logits = logits_flat[torch.arange(logits_flat.size(0)), targets_flat]
    others = logits_flat.clone()
    others[torch.arange(logits_flat.size(0)), targets_flat] = float("-inf")
    max_other, _ = others.max(dim=-1)
    margin_loss = torch.clamp(margin - (target_logits - max_other), min=0.0)
    return ce + margin_loss.mean()


def entropy_penalty_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    iter_num: int | None = None,
    beta: float = 0.01,
) -> torch.Tensor:
    """Cross entropy plus penalty on prediction entropy to encourage peaky outputs."""
    ce = cross_entropy_loss(logits, targets)
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    mask = targets != -1
    return ce + beta * entropy[mask].mean()


def top1_ratio_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    iter_num: int | None = None,
    beta: float = 0.5,
) -> torch.Tensor:
    """Novel loss encouraging the target logit to dominate all others exponentially."""
    ce = cross_entropy_loss(logits, targets)
    b, t, v = logits.shape
    logits_flat = logits.view(-1, v)
    targets_flat = targets.view(-1)
    mask = targets_flat != -1
    logits_flat = logits_flat[mask]
    targets_flat = targets_flat[mask]
    target_logits = logits_flat[torch.arange(logits_flat.size(0)), targets_flat]
    others = logits_flat.clone()
    others[torch.arange(logits_flat.size(0)), targets_flat] = float("-inf")
    max_other, _ = others.max(dim=-1)
    ratio_penalty = torch.exp(max_other - target_logits)
    return ce + beta * ratio_penalty.mean()


# def rank_distance_loss(
#     logits: torch.Tensor,
#     targets: torch.Tensor,
#     *,
#     iter_num: int | None = None,
#     gamma: float = 1.0,
#     focal_gamma: float = 2.0,
# ) -> torch.Tensor:
#     """Rank-distance scaled focal loss to emphasize hard, misranked targets."""
#     b, t, v = logits.shape
#     logits_flat = logits.view(-1, v)
#     targets_flat = targets.view(-1)
#     ce = F.cross_entropy(logits_flat, targets_flat, reduction="none", ignore_index=-1)
#     mask = targets_flat != -1
#     with torch.no_grad():
#         logits_sel = logits_flat[mask]
#         targets_sel = targets_flat[mask]
#         target_logits = logits_sel[torch.arange(logits_sel.size(0)), targets_sel]
#         rank = (logits_sel > target_logits.unsqueeze(-1)).sum(dim=-1)
#         rank_scale = (2 - rank) ** focal_gamma
#         pt = torch.exp(-ce[mask])
#     scaled = torch.zeros_like(ce)
#     scaled[mask] = ce[mask] * rank_scale
#     return scaled[mask].mean()

def rank_distance_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    iter_num: int | None = None,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Scale loss by how far the target's rank is from top-1.

    The rank is normalised by the vocabulary size so the scaling factor
    stays within ``[1, 1 + gamma]`` regardless of vocabulary length, which
    prevents overflow when the vocab is large.
    """
    b, t, v = logits.shape
    logits_flat = logits.view(-1, v)
    targets_flat = targets.view(-1)
    loss = F.cross_entropy(logits_flat, targets_flat, reduction="none", ignore_index=-1)
    mask = targets_flat != -1
    with torch.no_grad():
        logits_sel = logits_flat[mask]
        targets_sel = targets_flat[mask]
        target_logits = logits_sel[torch.arange(logits_sel.size(0)), targets_sel]
        rank = (logits_sel > target_logits.unsqueeze(-1)).sum(dim=-1)
        rank_norm = rank.float() / max(v - 1, 1) * 10.0
        scale = 1 + gamma * rank_norm
    scaled = torch.zeros_like(loss)
    # scaled[mask] = loss[mask] * scale
    scaled[mask] = loss[mask] / scale
    return scaled[mask].mean()


def flatness_boost_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    iter_num: int | None = None,
    beta: float = 1.0,
) -> torch.Tensor:
    """Boost loss when the predicted distribution is flat (high entropy)."""
    b, t, v = logits.shape
    logits_flat = logits.view(-1, v)
    targets_flat = targets.view(-1)
    loss = F.cross_entropy(logits_flat, targets_flat, reduction="none", ignore_index=-1)
    mask = targets_flat != -1
    with torch.no_grad():
        probs = torch.softmax(logits_flat[mask], dim=-1)
        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)
        entropy_norm = entropy / math.log(v)
        scale = 1 + beta * entropy_norm
    scaled = torch.zeros_like(loss)
    scaled[mask] = loss[mask] * scale
    return scaled[mask].mean()


def entropy_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    iter_num: int | None = None,
    gamma: float = 2.0,
    beta: float = 0.01,
) -> torch.Tensor:
    """Focal loss with an added entropy penalty to prefer peaky outputs."""
    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = targets.view(-1)
    ce = F.cross_entropy(logits_flat, targets_flat, reduction="none", ignore_index=-1)
    with torch.no_grad():
        pt = torch.exp(-ce)
    focal = ((1 - pt) ** gamma) * ce
    mask = targets_flat != -1
    focal_mean = focal[mask].mean()

    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    entropy_mean = entropy[targets != -1].mean()
    return focal_mean + beta * entropy_mean

# def rank_distance_focal_loss(
#     logits: torch.Tensor,
#     targets: torch.Tensor,
#     *,
#     iter_num: int | None = None,
#     gamma: float = 1.0,
#     focal_gamma: float = 2.0,
# ) -> torch.Tensor:
#     """Rank-distance scaled focal loss to emphasize hard, misranked targets."""
#     b, t, v = logits.shape
#     logits_flat = logits.view(-1, v)
#     targets_flat = targets.view(-1)
#     ce = F.cross_entropy(logits_flat, targets_flat, reduction="none", ignore_index=-1)
#     mask = targets_flat != -1
#     with torch.no_grad():
#         logits_sel = logits_flat[mask]
#         targets_sel = targets_flat[mask]
#         target_logits = logits_sel[torch.arange(logits_sel.size(0)), targets_sel]
#         rank = (logits_sel > target_logits.unsqueeze(-1)).sum(dim=-1)
#         rank_scale = 1 + gamma * (rank.float() / max(v - 1, 1))
#         pt = torch.exp(-ce[mask])
#         focal_scale = (1 - pt) ** focal_gamma
#     scaled = torch.zeros_like(ce)
#     scaled[mask] = ce[mask] * rank_scale * focal_scale
#     return scaled[mask].mean()

def rank_distance_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    iter_num: int | None = None,
    gamma: float = 1.0,
    focal_gamma: float = 2.0,
) -> torch.Tensor:
    """Rank-distance scaled focal loss to emphasize hard, misranked targets."""
    b, t, v = logits.shape
    logits_flat = logits.view(-1, v)
    targets_flat = targets.view(-1)
    ce = F.cross_entropy(logits_flat, targets_flat, reduction="none", ignore_index=-1)
    mask = targets_flat != -1
    with torch.no_grad():
        logits_sel = logits_flat[mask]
        targets_sel = targets_flat[mask]
        target_logits = logits_sel[torch.arange(logits_sel.size(0)), targets_sel]
        rank = (logits_sel > target_logits.unsqueeze(-1)).sum(dim=-1)
        rank_scale = 1 + gamma * (rank.float() / max(v - 1, 1))
        rank_scale = (2 - rank) ** focal_gamma
        pt = torch.exp(-ce[mask])
        focal_scale = (1 - pt) ** focal_gamma
    scaled = torch.zeros_like(ce)
    scaled[mask] = ce[mask] * rank_scale * focal_scale
    return scaled[mask].mean()

def entropy_rank_distance_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    iter_num: int | None = None,
    gamma: float = 1.0,
    focal_gamma: float = 2.0,
    beta: float = 0.01,
) -> torch.Tensor:
    """Combine rank-distance scaling, focal weighting, and entropy penalty."""
    loss = rank_distance_focal_loss(
        logits, targets, iter_num=iter_num, gamma=gamma, focal_gamma=focal_gamma
    )
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    mask = targets != -1
    return loss + beta * entropy[mask].mean()


LOSS_VARIANTS: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "cross_entropy": cross_entropy_loss,
    "label_smoothing": label_smoothing_loss,
    "focal": focal_loss,
    "top1_focus": top1_focus_loss,
    "skip_correct_top1": skip_correct_top1_loss,
    "attenuated_correct_top1": attenuated_correct_top1_loss,
    "distance_attenuated_top1": distance_attenuated_top1_loss,
    "top1_margin": top1_margin_loss,
    "entropy_penalty": entropy_penalty_loss,
    "top1_ratio": top1_ratio_loss,
    "rank_distance": rank_distance_loss,
    "flatness_boost": flatness_boost_loss,
    "entropy_focal": entropy_focal_loss,
    "rank_distance_focal": rank_distance_focal_loss,
    "entropy_rank_distance_focal": entropy_rank_distance_focal_loss,
}


# ---------------------------------------------------------------------------
# Loss scheduling
# ---------------------------------------------------------------------------


@dataclass
class ScheduledValue:
    """Schedule a scalar value over training iterations."""

    schedule: List[Tuple[int, float]]

    def __post_init__(self) -> None:
        self.schedule.sort(key=lambda x: x[0])

    def __call__(self, iter_num: int | None) -> float:
        val = self.schedule[0][1]
        if iter_num is not None:
            for step, candidate in self.schedule:
                if iter_num >= step:
                    val = candidate
                else:
                    break
        return val

@dataclass
class ScheduledLoss:
    """Switch between different loss functions at predefined iterations."""

    schedule: List[Tuple[int, str]]
    loss_dict: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]

    def __post_init__(self) -> None:
        self.schedule.sort(key=lambda x: x[0])

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor, *, iter_num: int | None = None) -> torch.Tensor:
        name = self.schedule[0][1]
        if iter_num is not None:
            for step, candidate in self.schedule:
                if iter_num >= step:
                    name = candidate
                else:
                    break
        return self.loss_dict[name](logits, targets, iter_num=iter_num)


def parse_loss_schedule(schedule_str: str) -> List[Tuple[int, str]]:
    """Parse a schedule string like ``"0:cross_entropy,1000:top1_focus"``."""
    schedule: List[Tuple[int, str]] = []
    for part in schedule_str.split(","):
        step_str, name = part.split(":")
        schedule.append((int(step_str), name.strip()))
    return schedule


def parse_value_schedule(schedule_str: str) -> ScheduledValue:
    """Parse a schedule string like ``"0:1.0,1000:2.0"`` for scalar values."""
    schedule: List[Tuple[int, float]] = []
    for part in schedule_str.split(","):
        step_str, value_str = part.split(":")
        schedule.append((int(step_str), float(value_str)))
    return ScheduledValue(schedule)


def build_loss_function(args) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return the loss function or a scheduler based on ``args``."""
    schedule_str = getattr(args, "loss_schedule", None)

    base = getattr(args, "rank_scale", 1.0)
    scale_sched_str = getattr(args, "rank_scale_schedule", None)
    scaler = parse_value_schedule(scale_sched_str) if scale_sched_str else None

    def rank_gamma(iter_num: int | None) -> float:
        return scaler(iter_num) if scaler else base

    built_losses: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
        "cross_entropy": LOSS_VARIANTS["cross_entropy"],
        "label_smoothing": lambda l, t, *, iter_num=None: LOSS_VARIANTS["label_smoothing"](
            l, t, iter_num=iter_num, smoothing=getattr(args, "label_smoothing", 0.1)
        ),
        "focal": lambda l, t, *, iter_num=None: LOSS_VARIANTS["focal"](
            l, t, iter_num=iter_num, gamma=getattr(args, "focal_gamma", 2.0)
        ),
        "top1_focus": lambda l, t, *, iter_num=None: LOSS_VARIANTS["top1_focus"](
            l, t, iter_num=iter_num, alpha=getattr(args, "top1_focus_alpha", 0.5)
        ),
        "skip_correct_top1": LOSS_VARIANTS["skip_correct_top1"],
        "attenuated_correct_top1": lambda l, t, *, iter_num=None: LOSS_VARIANTS["attenuated_correct_top1"](
            l, t, iter_num=iter_num, attenuation=getattr(args, "correct_top1_attenuation", 1.0)
        ),
        "distance_attenuated_top1": lambda l, t, *, iter_num=None: LOSS_VARIANTS["distance_attenuated_top1"](
            l, t, iter_num=iter_num, strength=getattr(args, "distance_top1_strength", 0.0)
        ),
        "top1_margin": lambda l, t, *, iter_num=None: LOSS_VARIANTS["top1_margin"](
            l, t, iter_num=iter_num, margin=getattr(args, "top1_margin", 0.1)
        ),
        "entropy_penalty": lambda l, t, *, iter_num=None: LOSS_VARIANTS["entropy_penalty"](
            l, t, iter_num=iter_num, beta=getattr(args, "entropy_beta", 0.01)
        ),
        "top1_ratio": lambda l, t, *, iter_num=None: LOSS_VARIANTS["top1_ratio"](
            l, t, iter_num=iter_num, beta=getattr(args, "top1_ratio_beta", 0.5)
        ),
        "rank_distance": lambda l, t, *, iter_num=None: LOSS_VARIANTS["rank_distance"](
            l, t, iter_num=iter_num, gamma=rank_gamma(iter_num)
        ),
        "flatness_boost": lambda l, t, *, iter_num=None: LOSS_VARIANTS["flatness_boost"](
            l, t, iter_num=iter_num, beta=getattr(args, "flatness_beta", 1.0)
        ),
        "entropy_focal": lambda l, t, *, iter_num=None: LOSS_VARIANTS["entropy_focal"](
            l,
            t,
            iter_num=iter_num,
            gamma=getattr(args, "focal_gamma", 2.0),
            beta=getattr(args, "entropy_beta", 0.01),
        ),
        "rank_distance_focal": lambda l, t, *, iter_num=None: LOSS_VARIANTS["rank_distance_focal"](
            l,
            t,
            iter_num=iter_num,
            gamma=rank_gamma(iter_num),
            focal_gamma=getattr(args, "focal_gamma", 2.0),
        ),
        "rank_distance_focal_v2": lambda l, t, *, iter_num=None: LOSS_VARIANTS["rank_distance_focal"](
            l,
            t,
            iter_num=iter_num,
            gamma=rank_gamma(iter_num),
            focal_gamma=getattr(args, "focal_gamma", 2.0),
        ),
        "entropy_rank_distance_focal": lambda l, t, *, iter_num=None: LOSS_VARIANTS["entropy_rank_distance_focal"](
            l,
            t,
            iter_num=iter_num,
            gamma=rank_gamma(iter_num),
            focal_gamma=getattr(args, "focal_gamma", 2.0),
            beta=getattr(args, "entropy_beta", 0.01),
        ),
    }

    if schedule_str:
        schedule = parse_loss_schedule(schedule_str)
        return ScheduledLoss(schedule, built_losses)

    loss_name = getattr(args, "loss_fn", "cross_entropy")
    return built_losses.get(loss_name, LOSS_VARIANTS["cross_entropy"])

