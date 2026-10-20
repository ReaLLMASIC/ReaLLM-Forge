"""Deterministic bounded mixtures. Seeds alter amplitudes/phases, never time labels."""
from __future__ import annotations
import numpy as np


def sine_series(seed: int, frames: int, channels: int, start: int = 0) -> np.ndarray:
    if not 1 <= channels <= 32 or frames < 1 or start < 0:
        raise ValueError("channels must be 1..32; frames positive; start nonnegative")
    t = np.arange(start, start + frames, dtype=np.float64)
    result = []
    for c in range(channels):
        # Separate lane RNG means lane 0 stays identical when adding more lanes.
        rng = np.random.default_rng(np.random.SeedSequence([seed, c]))
        amplitude = np.array([0.52, 0.29, 0.19]) * rng.uniform(0.85, 1.15, 3)
        amplitude *= 0.90 / amplitude.sum()
        phase = rng.uniform(-np.pi, np.pi, 3)
        frequency = np.array([1 / 128, 1 / 47, 1 / 23]) * (1 + 0.07 * c)
        result.append((amplitude[:, None] * np.sin(
            2 * np.pi * frequency[:, None] * t + phase[:, None])).sum(axis=0))
    return np.stack(result, axis=-1).astype(np.float32)


def sequence_bank(seed: int, count: int, frames: int, channels: int) -> np.ndarray:
    return np.stack([sine_series(seed + i, frames, channels) for i in range(count)])
