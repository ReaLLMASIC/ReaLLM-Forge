"""Formatting helpers for Rich training progress-bar task fields."""

from typing import Any


def _as_float(value: Any, default: float = float("nan")) -> float:
    if value is None:
        return default
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def format_progress_metrics(metrics: dict[str, Any]) -> dict[str, str]:
    """Convert raw Trainer progress metrics into Rich task field strings."""
    time_remaining_ms = _as_float(metrics.get("time_remaining_ms"), 0.0)
    total_time_est_ms = _as_float(metrics.get("total_time_est_ms"), 0.0)
    peak_torch_allocated = _as_float(metrics.get("peak_torch_allocated"), 0.0)

    return {
        "eta": str(metrics.get("eta", "waiting for calculation")),
        "total_hour": f"{int(total_time_est_ms // 3_600_000)}",
        "total_min": f"{int((total_time_est_ms // 60_000) % 60):02d}",
        "hour": f"{int((time_remaining_ms // 3_600_000) % 24):02d}",
        "min": f"{int((time_remaining_ms // 60_000) % 60):02d}",
        "best_val_loss": f"{_as_float(metrics.get('best_val_loss')):.3f}",
        "best_iter": f"{_as_int(metrics.get('best_iter'))}",
        "best_tokens": f"{_as_int(metrics.get('best_tokens'))}",
        "iter_latency": f"{_as_float(metrics.get('iter_latency_avg'), 0.0):.1f}",
        "peak_gpu_mb": f"{peak_torch_allocated / (1024 ** 2):.1f}",
        "t1p": f"{_as_float(metrics.get('latest_top1_prob')):.6f}",
        "t1c": f"{_as_float(metrics.get('latest_top1_correct')):.6f}",
        "tr": f"{_as_float(metrics.get('latest_target_rank')):.2f}",
        "tp": f"{_as_float(metrics.get('latest_target_prob')):.6f}",
        "tlp": f"{_as_float(metrics.get('latest_target_left_prob')):.6f}",
        "r95": f"{_as_float(metrics.get('latest_rank_95')):.2f}",
        "p95": f"{_as_float(metrics.get('latest_left_prob_95')):.6f}",
        "lnf_cos": f"{_as_float(metrics.get('latest_ln_f_cosine')):.6f}",
        "lnf_cos95": f"{_as_float(metrics.get('latest_ln_f_cosine_95')):.6f}",
    }
