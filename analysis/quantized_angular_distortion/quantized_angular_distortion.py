#!/usr/bin/env python3
"""
Angle-dependent distortion from low-bit integer quantization and custom fake-FP formats.

This is a full, standalone implementation of the original script with the following
FP-related fixes and extensions:

  * Fixes mantissa-rounding overflow at the largest finite exponent. Values that
    round past the top finite bin now saturate to the true max finite value rather
    than wrapping/collapsing to a smaller value.
  * Rejects IEEE-like formats with fewer than 2 exponent bits. In this mode,
    exponent=0 is subnormal/zero and exponent=all-ones is reserved for specials,
    so E1M* has no normal finite exponent codes.
  * Tightens E/M parsing. A prefix such as fp8:e4m3 is checked against
    1 + E + M, so fp6:e4m3 is rejected instead of silently becoming FP8 E4M3.
  * Keeps IEEE-like custom formats as the default, with E1M* removed from the
    default list.
  * Adds optional all-finite and FN-style fake-FP modes for explicit experiments
    with ML-style minifloats such as finite:e2m1 and e4m3fn.
  * Adds an optional --fp-split-by-total-bits mode to write one FP PDF/CSV per
    total bit-width, e.g. FP8, FP6, FP5, FP4.

The integer quantization transfer-theory, integer Monte Carlo, dimension sweep,
arcsine-law plot, CSV outputs, and FP theory/empirical plotting features from the
original script are retained.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

# Use a non-interactive backend so this runs cleanly on headless machines.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import ndtr
from scipy.stats import norm


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class TheoryPoint:
    angle_deg: float
    bits: int
    rho_true: float
    rho_quant_theory: float
    quant_angle_deg_theory: float
    distortion_deg_theory: float
    std_distortion_deg_delta: float


@dataclass(frozen=True)
class EmpiricalPoint:
    angle_deg: float
    bits: int
    mean_distortion_deg: float
    std_distortion_deg: float
    valid_trials: int


@dataclass(frozen=True)
class FPFormat:
    """Custom fake floating-point format.

    Parameters
    ----------
    exp_bits:
        Number of explicit exponent bits, E.
    mant_bits:
        Number of explicit fraction/mantissa bits, M.
    name:
        Human-readable label. If omitted, one is generated from total bits,
        E, M, and mode.
    mode:
        "ieee"   -> IEEE-like: exponent=0 is subnormal/zero and exponent=all-ones
                    is reserved for specials. This requires exp_bits >= 2.
        "finite" -> all exponent codes are finite; exponent=0 still provides
                    zero/subnormals, but exponent=all-ones is finite.
        "fn"     -> finite-number style, useful for E4M3FN-like experiments:
                    exponent=all-ones is finite except that the largest mantissa
                    pattern at the largest exponent is not used as a finite value.
                    This gives E4M3FN max finite 448 instead of 480.
    exp_bias:
        Optional explicit exponent bias. By default this is 2**(E-1)-1.
    """

    exp_bits: int
    mant_bits: int
    name: str = ""
    mode: str = "ieee"
    exp_bias: Optional[int] = None

    def __post_init__(self) -> None:
        mode = self.mode.lower().replace("-", "_")
        aliases = {
            "ieee_like": "ieee",
            "ieeelike": "ieee",
            "reserved": "ieee",
            "allfinite": "finite",
            "all_finite": "finite",
            "finite_only": "finite",
            "floatonly": "finite",
            "float_only": "finite",
            "e4m3fn": "fn",
            "finite_number": "fn",
        }
        mode = aliases.get(mode, mode)
        object.__setattr__(self, "mode", mode)
        self.validate()

        if not self.name:
            object.__setattr__(self, "name", self.default_name())

    @property
    def total_bits(self) -> int:
        return 1 + self.exp_bits + self.mant_bits

    @property
    def bias(self) -> int:
        if self.exp_bias is not None:
            return int(self.exp_bias)
        return (2 ** (self.exp_bits - 1)) - 1

    @property
    def emin(self) -> int:
        # Minimum normal exponent. Exponent code 0 is zero/subnormal in all modes.
        return 1 - self.bias

    @property
    def emax(self) -> int:
        if self.mode == "ieee":
            # All-ones exponent is reserved for inf/NaN-like specials.
            return (2 ** self.exp_bits - 2) - self.bias
        # All-ones exponent is finite in finite and FN modes.
        return (2 ** self.exp_bits - 1) - self.bias

    @property
    def mantissa_levels(self) -> int:
        return 2 ** self.mant_bits

    @property
    def min_normal(self) -> float:
        return float(2.0 ** self.emin)

    @property
    def min_positive_quantum(self) -> float:
        # With M=0, this is the spacing between zero and min-normal under
        # nearest-value rounding. There are no non-zero subnormal mantissa codes,
        # but this value is still useful for the rounding threshold logic.
        return float((2.0 ** self.emin) / self.mantissa_levels)

    @property
    def top_mantissa_index(self) -> int:
        levels = self.mantissa_levels
        if self.mode == "fn":
            # E4M3FN-style: reserve the largest mantissa at the largest exponent
            # as a non-finite/NaN-like pattern. For M=3 this caps top exponent at
            # mantissa index 6, yielding 1.75 * 2**8 = 448.
            return max(0, levels - 2)
        return levels - 1

    @property
    def max_finite(self) -> float:
        return float((1.0 + self.top_mantissa_index / self.mantissa_levels) * (2.0 ** self.emax))

    def validate(self) -> None:
        if self.exp_bits < 1:
            raise ValueError("exp_bits must be >= 1.")
        if self.mant_bits < 0:
            raise ValueError("mant_bits must be >= 0.")
        if self.mode not in {"ieee", "finite", "fn"}:
            raise ValueError("FP mode must be one of: ieee, finite, fn.")
        if self.mode == "ieee" and self.exp_bits < 2:
            raise ValueError(
                "IEEE-like fake FP reserves exponent=0 for subnormals and "
                "exponent=all-ones for specials, so exp_bits must be >= 2. "
                "Use finite:e1m* for explicit all-finite E1M* experiments."
            )
        if self.mode == "fn" and self.mant_bits < 1:
            raise ValueError("FN mode requires mant_bits >= 1 so a top NaN-like mantissa pattern can be reserved.")

    def default_name(self) -> str:
        if self.mode == "ieee":
            suffix = "IEEE-like"
            return f"FP{self.total_bits} E{self.exp_bits}M{self.mant_bits} {suffix}"
        if self.mode == "finite":
            return f"FP{self.total_bits} E{self.exp_bits}M{self.mant_bits} finite"
        # FN-style is conventionally written as E4M3FN when used for FP8.
        return f"FP{self.total_bits} E{self.exp_bits}M{self.mant_bits}FN"

    def summary_dict(self) -> Mapping[str, object]:
        return {
            "name": self.name,
            "mode": self.mode,
            "total_bits": self.total_bits,
            "exp_bits": self.exp_bits,
            "mant_bits": self.mant_bits,
            "bias": self.bias,
            "emin": self.emin,
            "emax": self.emax,
            "min_normal": self.min_normal,
            "min_positive_quantum": self.min_positive_quantum,
            "max_finite": self.max_finite,
        }


# -----------------------------------------------------------------------------
# Integer quantization path
# -----------------------------------------------------------------------------


def qmax_for_bits(bits: int) -> int:
    if bits < 2:
        raise ValueError("Use bits >= 2.")
    return (2 ** (bits - 1)) - 1


def quantize_codes(values: np.ndarray, bits: int, scale: float) -> np.ndarray:
    qmax = qmax_for_bits(bits)
    return np.clip(np.rint(values / scale), -qmax, qmax).astype(np.float64)


def choose_scale(v: np.ndarray, bits: int, scale_mode: str, clip_sigma: float) -> float:
    qmax = qmax_for_bits(bits)
    d = v.size

    if scale_mode == "fixed":
        full_scale = clip_sigma / math.sqrt(d)
    elif scale_mode == "std":
        full_scale = clip_sigma * float(np.std(v))
    elif scale_mode == "maxabs":
        full_scale = float(np.max(np.abs(v)))
    else:
        raise ValueError(f"Unknown scale_mode={scale_mode}")

    return max(full_scale / qmax, np.finfo(float).tiny)


def quantize_vector(v: np.ndarray, bits: int, scale_mode: str, clip_sigma: float) -> np.ndarray:
    return quantize_codes(v, bits, choose_scale(v, bits, scale_mode, clip_sigma))


def quantized_angle_deg(x: np.ndarray, y: np.ndarray, bits: int, scale_mode: str, clip_sigma: float) -> float:
    qx = quantize_vector(x, bits, scale_mode, clip_sigma)
    qy = quantize_vector(y, bits, scale_mode, clip_sigma)

    nx = float(np.linalg.norm(qx))
    ny = float(np.linalg.norm(qy))

    if nx == 0.0 or ny == 0.0:
        return float("nan")

    c = float(np.dot(qx, qy) / (nx * ny))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def random_unit_pair_at_angle(dim: int, angle_deg: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    theta = math.radians(angle_deg)
    rho = math.cos(theta)

    x = rng.normal(size=dim)
    x_norm = np.linalg.norm(x)
    if x_norm == 0.0:
        raise RuntimeError("RNG returned a zero vector; try a different seed.")
    x /= x_norm

    # Rejection loop is essentially never needed in floating point, but it makes
    # the function well-defined for very small dimensions and pathological draws.
    for _ in range(32):
        z = rng.normal(size=dim)
        z -= np.dot(z, x) * x
        z_norm = np.linalg.norm(z)
        if z_norm > 0.0:
            z /= z_norm
            break
    else:
        raise RuntimeError("Could not sample an orthogonal direction; try a different seed or dimension.")

    y = rho * x + math.sqrt(max(0.0, 1.0 - rho * rho)) * z
    y /= np.linalg.norm(y)

    return x, y


# -----------------------------------------------------------------------------
# Integer Gaussian transfer theory
# -----------------------------------------------------------------------------


def effective_tau(scale_mode: str, clip_sigma: float, dim: int) -> float:
    if scale_mode in {"fixed", "std"}:
        return float(clip_sigma)

    if scale_mode == "maxabs":
        # Approximate expected max-absolute threshold for d standard-normal
        # coordinates. The empirical path still uses the actual vector maxabs.
        return float(norm.ppf(1.0 - 1.0 / (2.0 * dim)))

    raise ValueError(f"Unknown scale_mode={scale_mode}")


def normal_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def quantizer_bins(qmax: int, delta: float) -> Iterable[Tuple[int, float, float]]:
    for k in range(-qmax, qmax + 1):
        if k == -qmax:
            yield k, -math.inf, (k + 0.5) * delta
        elif k == qmax:
            yield k, (k - 0.5) * delta, math.inf
        else:
            yield k, (k - 0.5) * delta, (k + 0.5) * delta


def code_from_standard_x(x: np.ndarray, qmax: int, delta: float) -> np.ndarray:
    return np.clip(np.rint(x / delta), -qmax, qmax).astype(np.float64)


def conditional_code_moments(
    mean: np.ndarray,
    sd: float,
    qmax: int,
    delta: float,
    max_power: int = 4,
) -> np.ndarray:
    mean = np.asarray(mean, dtype=np.float64)
    out = np.zeros((max_power + 1, mean.size), dtype=np.float64)
    out[0, :] = 1.0

    if sd < 1e-12:
        q = code_from_standard_x(mean, qmax, delta)
        for p in range(1, max_power + 1):
            out[p, :] = q ** p
        return out

    for k in range(-qmax, qmax + 1):
        if k == -qmax:
            lo, hi = -np.inf, (k + 0.5) * delta
        elif k == qmax:
            lo, hi = (k - 0.5) * delta, np.inf
        else:
            lo, hi = (k - 0.5) * delta, (k + 0.5) * delta

        pbin = ndtr((hi - mean) / sd) - ndtr((lo - mean) / sd)

        for p in range(1, max_power + 1):
            out[p, :] += (float(k) ** p) * pbin

    return out


def quantized_bivariate_code_moments(
    rho: float,
    qmax: int,
    delta: float,
    nodes_per_bin: int = 96,
    tail_bound: float = 12.0,
) -> np.ndarray:
    rho = max(-1.0, min(1.0, float(rho)))
    sd = math.sqrt(max(0.0, 1.0 - rho * rho))
    lg_x, lg_w = leggauss(nodes_per_bin)
    moments = np.zeros((5, 5), dtype=np.float64)

    for k, lo, hi in quantizer_bins(qmax, delta):
        a = max(lo, -tail_bound)
        b = min(hi, tail_bound)

        if not (a < b):
            continue

        mid = 0.5 * (a + b)
        half = 0.5 * (b - a)

        x = mid + half * lg_x
        wx = half * lg_w * normal_pdf(x)

        y_mom = conditional_code_moments(rho * x, sd, qmax, delta, 4)
        integrated = np.sum(wx[None, :] * y_mom, axis=1)

        kpowers = np.array([float(k) ** p for p in range(5)], dtype=np.float64)
        moments += kpowers[:, None] * integrated[None, :]

    return moments


def theory_for_angle(
    angle_deg: float,
    bits: int,
    dim: int,
    scale_mode: str,
    clip_sigma: float,
    quad_nodes: int = 96,
) -> TheoryPoint:
    theta = math.radians(angle_deg)
    rho = max(-1.0, min(1.0, math.cos(theta)))

    qmax = qmax_for_bits(bits)
    tau = effective_tau(scale_mode, clip_sigma, dim)
    delta = tau / qmax

    M = quantized_bivariate_code_moments(
        rho,
        qmax,
        delta,
        nodes_per_bin=quad_nodes,
        tail_bound=max(12.0, tau + 8.0),
    )

    m_a = float(M[1, 1])
    m2 = float(M[2, 0])
    m4 = float(M[4, 0])
    e22 = float(M[2, 2])
    e31 = float(M[3, 1])
    e13 = float(M[1, 3])

    if m2 <= 0.0:
        rho_q = float("nan")
        theta_q = float("nan")
        std_delta = float("nan")
    else:
        rho_q = max(-1.0, min(1.0, m_a / m2))
        theta_q = math.acos(rho_q)

        var_a = e22 - m_a * m_a
        var_b = m4 - m2 * m2
        cov_ab = e31 - m_a * m2
        cov_ac = e13 - m_a * m2
        cov_bc = e22 - m2 * m2

        cov = np.array(
            [
                [var_a, cov_ab, cov_ac],
                [cov_ab, var_b, cov_bc],
                [cov_ac, cov_bc, var_b],
            ],
            dtype=np.float64,
        )

        grad = np.array(
            [
                1.0 / m2,
                -m_a / (2.0 * m2 * m2),
                -m_a / (2.0 * m2 * m2),
            ],
            dtype=np.float64,
        )

        var_rho_q = float(grad @ cov @ grad) / float(dim)
        var_theta_q = max(0.0, var_rho_q / max(1e-15, 1.0 - rho_q * rho_q))
        std_delta = math.degrees(math.sqrt(var_theta_q))

    return TheoryPoint(
        angle_deg=angle_deg,
        bits=bits,
        rho_true=rho,
        rho_quant_theory=rho_q,
        quant_angle_deg_theory=math.degrees(theta_q) if math.isfinite(theta_q) else float("nan"),
        distortion_deg_theory=math.degrees(theta_q - theta) if math.isfinite(theta_q) else float("nan"),
        std_distortion_deg_delta=std_delta,
    )


def empirical_for_angle(
    angle_deg: float,
    bits: int,
    dim: int,
    trials: int,
    scale_mode: str,
    clip_sigma: float,
    rng: np.random.Generator,
) -> EmpiricalPoint:
    vals: List[float] = []

    for _ in range(trials):
        x, y = random_unit_pair_at_angle(dim, angle_deg, rng)
        qa = quantized_angle_deg(x, y, bits, scale_mode, clip_sigma)

        if np.isfinite(qa):
            vals.append(qa - angle_deg)

    arr = np.asarray(vals, dtype=np.float64)

    return EmpiricalPoint(
        angle_deg=angle_deg,
        bits=bits,
        mean_distortion_deg=float(np.mean(arr)) if arr.size else float("nan"),
        std_distortion_deg=float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        valid_trials=int(arr.size),
    )


# -----------------------------------------------------------------------------
# Fake-FP quantization path
# -----------------------------------------------------------------------------


def fake_fp_quantize(x: np.ndarray, fmt: FPFormat) -> np.ndarray:
    """Quantize to a custom fake-FP format and return float64 dequantized values.

    The implementation rounds to nearest-even via numpy.rint, supports subnormal
    values, preserves NaNs, and saturates infinities/out-of-range finite values to
    the format's largest finite value. The key debugged behavior is that mantissa
    rounding overflow at emax saturates instead of carrying into a clipped exponent.
    """

    fmt.validate()

    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x, dtype=np.float64)

    nan = np.isnan(x)
    out[nan] = np.nan

    ax_all = np.abs(x)
    nonzero = (ax_all > 0.0) & ~nan

    if not np.any(nonzero):
        return out

    ax = ax_all[nonzero]
    sign = np.sign(x[nonzero])

    emin = fmt.emin
    emax = fmt.emax
    levels = fmt.mantissa_levels
    max_val = fmt.max_finite
    min_normal = fmt.min_normal

    q = np.zeros_like(ax, dtype=np.float64)

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        e = np.floor(np.log2(ax))

    normal = e >= emin

    if np.any(normal):
        e_norm = np.clip(e[normal], emin, emax)
        scale = 2.0 ** e_norm

        with np.errstate(over="ignore", invalid="ignore"):
            frac = ax[normal] / scale - 1.0
            frac_i = np.rint(frac * levels)

        # Carry mantissa overflow only when exponent headroom exists. At the top
        # finite exponent, overflow must saturate to max finite; otherwise values
        # above range can collapse to 1.0 * 2**emax, which is the original bug.
        overflow = frac_i >= levels
        can_carry = overflow & (e_norm < emax)
        if np.any(can_carry):
            e_norm[can_carry] += 1.0
            frac_i[can_carry] = 0.0

        must_saturate = overflow & ~can_carry
        if np.any(must_saturate):
            frac_i[must_saturate] = fmt.top_mantissa_index

        # In FN-style formats, the largest mantissa index at the largest exponent
        # is not a finite value. Cap that top-exponent mantissa accordingly.
        at_top_exp = e_norm >= emax
        too_large_at_top = at_top_exp & (frac_i > fmt.top_mantissa_index)
        if np.any(too_large_at_top):
            frac_i[too_large_at_top] = fmt.top_mantissa_index

        frac_i = np.clip(frac_i, 0.0, levels - 1.0)
        q[normal] = (1.0 + frac_i / levels) * (2.0 ** e_norm)

    if np.any(~normal):
        # Subnormal grid spacing. With M=0 this gives nearest rounding between
        # zero and min-normal, because there are no non-zero subnormal codes.
        sub_step = min_normal / levels
        q_sub = np.rint(ax[~normal] / sub_step) * sub_step
        q[~normal] = np.minimum(q_sub, min_normal)

    q = np.minimum(q, max_val)
    out[nonzero] = sign * q
    return out


def fp_theory_mc(angle_deg: float, fmt: FPFormat, n: int, seed: int) -> float:
    rng = np.random.default_rng(seed)

    theta = math.radians(angle_deg)
    rho = math.cos(theta)

    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y = rho * x + math.sqrt(max(0.0, 1.0 - rho * rho)) * z

    qx = fake_fp_quantize(x, fmt)
    qy = fake_fp_quantize(y, fmt)

    denom = math.sqrt(float(np.mean(qx * qx)) * float(np.mean(qy * qy)))

    if denom == 0.0 or not math.isfinite(denom):
        return float("nan")

    cq = float(np.mean(qx * qy) / denom)
    cq = max(-1.0, min(1.0, cq))

    return math.degrees(math.acos(cq) - theta)


def empirical_fp_for_angle(
    angle_deg: float,
    fmt: FPFormat,
    dim: int,
    trials: int,
    rng: np.random.Generator,
) -> Tuple[float, float, int]:
    vals: List[float] = []

    for _ in range(trials):
        x, y = random_unit_pair_at_angle(dim, angle_deg, rng)

        qx = fake_fp_quantize(np.sqrt(dim) * x, fmt)
        qy = fake_fp_quantize(np.sqrt(dim) * y, fmt)

        nx = float(np.linalg.norm(qx))
        ny = float(np.linalg.norm(qy))

        if nx == 0.0 or ny == 0.0:
            continue

        c = float(np.dot(qx, qy) / (nx * ny))
        c = max(-1.0, min(1.0, c))

        vals.append(math.degrees(math.acos(c)) - angle_deg)

    arr = np.asarray(vals, dtype=np.float64)

    return (
        float(np.mean(arr)) if arr.size else float("nan"),
        float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        int(arr.size),
    )


# -----------------------------------------------------------------------------
# FP format presets and parsing
# -----------------------------------------------------------------------------


_MODE_ALIASES: Mapping[str, str] = {
    "ieee": "ieee",
    "ieee_like": "ieee",
    "ieeelike": "ieee",
    "reserved": "ieee",
    "finite": "finite",
    "allfinite": "finite",
    "all_finite": "finite",
    "fn": "fn",
    "e4m3fn": "fn",
    "finite_number": "fn",
}


_FORMAT_BODY_RE = re.compile(
    r"^(?:(?:fp|float)?(?P<total>\d+))?e(?P<e>\d+)m(?P<m>\d+)(?P<suffix>fn|finite|ieee)?$",
    flags=re.IGNORECASE,
)


def normalize_fp_mode(mode: str) -> str:
    key = mode.strip().lower().replace("-", "_")
    if key not in _MODE_ALIASES:
        raise ValueError(f"Unknown FP mode {mode!r}. Use ieee, finite, or fn.")
    return _MODE_ALIASES[key]


def make_fp_format(exp_bits: int, mant_bits: int, mode: str = "ieee", name: str = "", exp_bias: Optional[int] = None) -> FPFormat:
    return FPFormat(exp_bits=exp_bits, mant_bits=mant_bits, name=name, mode=normalize_fp_mode(mode), exp_bias=exp_bias)


def default_fp_formats(preset: str = "debugged-ieee") -> List[FPFormat]:
    preset_key = preset.strip().lower().replace("-", "_")

    if preset_key in {"debugged_ieee", "ieee", "default"}:
        # E1M* intentionally omitted because IEEE-like mode reserves both exponent
        # zero and exponent all-ones, leaving no normal finite E1 codes.
        return [
            make_fp_format(5, 2, "ieee", "FP8 E5M2 IEEE-like"),
            make_fp_format(4, 3, "ieee", "FP8 E4M3 IEEE-like"),
            make_fp_format(3, 4, "ieee", "FP8 E3M4 IEEE-like"),
            make_fp_format(2, 3, "ieee", "FP6 E2M3 IEEE-like"),
            make_fp_format(2, 2, "ieee", "FP5 E2M2 IEEE-like"),
            make_fp_format(2, 1, "ieee", "FP4 E2M1 IEEE-like"),
        ]

    if preset_key in {"ml", "ml_float", "named_ml"}:
        # A compact ML-oriented preset: E5M2 as IEEE-like, E4M3FN as FN-style,
        # E3M4 as IEEE-like, and FLOAT4E2M1 as all-finite.
        return [
            make_fp_format(5, 2, "ieee", "FP8 E5M2 IEEE-like"),
            make_fp_format(4, 3, "fn", "FP8 E4M3FN"),
            make_fp_format(3, 4, "ieee", "FP8 E3M4 IEEE-like"),
            make_fp_format(2, 1, "finite", "FP4 E2M1 finite"),
        ]

    if preset_key in {"legacy_finite", "finite"}:
        # The legacy list is allowed only in all-finite mode; it includes the E1M*
        # experiments that are invalid in IEEE-like mode.
        return [
            make_fp_format(5, 2, "finite"),
            make_fp_format(4, 3, "finite"),
            make_fp_format(3, 4, "finite"),
            make_fp_format(2, 3, "finite"),
            make_fp_format(1, 4, "finite"),
            make_fp_format(2, 2, "finite"),
            make_fp_format(1, 3, "finite"),
            make_fp_format(2, 1, "finite"),
            make_fp_format(1, 2, "finite"),
        ]

    raise ValueError("Unknown --fp-preset. Use debugged-ieee, ml, or legacy-finite.")


def parse_fp_format(spec: str, default_mode: str = "ieee") -> FPFormat:
    """Parse a fake-FP spec.

    Accepted examples:
        e4m3
        E5M2
        fp8:e4m3
        ieee:e4m3
        finite:e1m4
        fn:e4m3
        e4m3fn
        fp8:e4m3fn
        float4e2m1

    Any explicit total-bit prefix is checked against 1 + E + M.
    """

    raw = spec.strip()
    if not raw:
        raise ValueError("Empty FP format spec.")

    mode = normalize_fp_mode(default_mode)
    claimed_total: Optional[int] = None

    parts = [p.strip().lower().replace("-", "_") for p in raw.split(":") if p.strip()]
    if not parts:
        raise ValueError("Empty FP format spec.")

    body = parts[-1]
    prefixes = parts[:-1]

    for prefix in prefixes:
        if prefix in _MODE_ALIASES:
            mode = normalize_fp_mode(prefix)
            continue

        total_match = re.fullmatch(r"(?:fp|float)?(\d+)", prefix)
        if total_match:
            claimed_total = int(total_match.group(1))
            continue

        raise ValueError(
            f"Bad FP format prefix {prefix!r} in {spec!r}. "
            "Use prefixes such as fp8:, ieee:, finite:, or fn:."
        )

    body_match = _FORMAT_BODY_RE.fullmatch(body)
    if not body_match:
        raise ValueError(
            f"Bad FP format spec: {spec}. Use e4m3, fp8:e4m3, finite:e1m4, or e4m3fn."
        )

    if body_match.group("total") is not None:
        body_total = int(body_match.group("total"))
        if claimed_total is not None and claimed_total != body_total:
            raise ValueError(f"Conflicting total-bit prefixes in {spec!r}.")
        claimed_total = body_total

    suffix = body_match.group("suffix")
    if suffix:
        mode = normalize_fp_mode(suffix)

    e = int(body_match.group("e"))
    m = int(body_match.group("m"))
    total = 1 + e + m

    if claimed_total is not None and claimed_total != total:
        raise ValueError(
            f"Bad FP format spec {spec!r}: prefix says FP{claimed_total}, "
            f"but E{e}M{m} implies FP{total}."
        )

    return make_fp_format(e, m, mode=mode)


def parse_fp_formats(specs: Optional[Sequence[str]], preset: str, default_mode: str) -> List[FPFormat]:
    if specs:
        return [parse_fp_format(s, default_mode=default_mode) for s in specs]
    return default_fp_formats(preset)


# -----------------------------------------------------------------------------
# CSV and plotting helpers
# -----------------------------------------------------------------------------


def parse_angles(args: argparse.Namespace) -> List[float]:
    if args.angles:
        return [float(a) for a in args.angles]

    if args.angles_step <= 0.0:
        raise ValueError("--angles-step must be positive.")
    if args.angles_stop < args.angles_start:
        raise ValueError("--angles-stop must be >= --angles-start.")

    return [
        float(a)
        for a in np.arange(
            args.angles_start,
            args.angles_stop + 0.5 * args.angles_step,
            args.angles_step,
        )
    ]


def ensure_parent(path: Path) -> None:
    parent = path.parent
    if parent and str(parent) != ".":
        parent.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, theory: Sequence[TheoryPoint], empirical: Sequence[EmpiricalPoint], args: argparse.Namespace) -> None:
    ensure_parent(path)
    emp_map = {(e.bits, e.angle_deg): e for e in empirical}

    with Path(path).open("w", newline="") as f:
        w = csv.writer(f)

        w.writerow(
            [
                "bits",
                "angle_deg",
                "rho_true",
                "rho_quant_theory",
                "quant_angle_deg_theory",
                "distortion_deg_theory",
                "std_distortion_deg_delta_method",
                "mean_distortion_deg_empirical",
                "std_distortion_deg_empirical",
                "valid_trials",
                "dim",
                "scale_mode",
                "clip_sigma",
            ]
        )

        for t in theory:
            e = emp_map.get((t.bits, t.angle_deg))

            w.writerow(
                [
                    t.bits,
                    t.angle_deg,
                    t.rho_true,
                    t.rho_quant_theory,
                    t.quant_angle_deg_theory,
                    t.distortion_deg_theory,
                    t.std_distortion_deg_delta,
                    e.mean_distortion_deg if e else "",
                    e.std_distortion_deg if e else "",
                    e.valid_trials if e else "",
                    args.dim,
                    args.scale_mode,
                    args.clip_sigma,
                ]
            )


def write_fp_csv(path: Path, rows: Sequence[Sequence[object]]) -> None:
    ensure_parent(path)
    with Path(path).open("w", newline="") as f:
        w = csv.writer(f)

        w.writerow(
            [
                "format",
                "mode",
                "total_bits",
                "exp_bits",
                "mant_bits",
                "bias",
                "emin",
                "emax",
                "min_normal",
                "min_positive_quantum",
                "max_finite",
                "angle_deg",
                "distortion_deg_theory_mc",
                "mean_distortion_deg_empirical",
                "std_distortion_deg_empirical",
                "valid_trials",
                "dim",
            ]
        )

        for row in rows:
            w.writerow(row)


def setup_pub_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "lines.linewidth": 2.2,
        }
    )


def plot_results(
    path: Path,
    bits_list: Sequence[int],
    angles: Sequence[float],
    theory: Sequence[TheoryPoint],
    empirical: Sequence[EmpiricalPoint],
    args: argparse.Namespace,
) -> None:
    setup_pub_style()
    ensure_parent(path)

    t_by_bits = {b: [p for p in theory if p.bits == b] for b in bits_list}
    e_by_bits = {b: [p for p in empirical if p.bits == b] for b in bits_list}

    colors = {
        2: "#d62728",
        3: "#1f77b4",
        4: "#2ca02c",
        5: "#9467bd",
        6: "#8c564b",
        7: "#e377c2",
        8: "#7f7f7f",
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    ax.axhline(0.0, linewidth=1, linestyle="--", color="black", alpha=0.5)
    ax.axvspan(20, 30, color="gray", alpha=0.12, label=r"$20^\circ$--$30^\circ$ regime")

    for bits in bits_list:
        ts = sorted(t_by_bits[bits], key=lambda p: p.angle_deg)
        es = sorted(e_by_bits[bits], key=lambda p: p.angle_deg)
        color = colors.get(bits)

        ax.plot(
            [p.angle_deg for p in ts],
            [p.distortion_deg_theory for p in ts],
            color=color,
            label=fr"INT{bits} transfer",
        )

        if es:
            ax.errorbar(
                [p.angle_deg for p in es],
                [p.mean_distortion_deg for p in es],
                yerr=[p.std_distortion_deg for p in es],
                fmt="o",
                color=color,
                markersize=2.4,
                capsize=1.6,
                elinewidth=0.75,
                alpha=0.45,
                label=fr"INT{bits} MC",
            )

    ax.set_xlim(min(0.0, min(angles)), max(90.0, max(angles)))
    ax.set_xticks(np.arange(0, 91, 10))
    ax.set_xlabel(r"Original angle $\theta$ (degrees)")
    ax.set_ylabel(r"Angular distortion $\arccos(C_b(\cos\theta))-\theta$ (degrees)")
    ax.set_title("Angle-dependent distortion from low-bit symmetric quantization")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=True, ncol=2)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_dimension_sweep_by_bit(
    path: Path,
    bit: int,
    dim_to_theory: Mapping[int, Sequence[TheoryPoint]],
    dim_to_empirical: Mapping[int, Sequence[EmpiricalPoint]],
) -> None:
    setup_pub_style()
    ensure_parent(path)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    ax.axhline(0.0, linewidth=1, linestyle="--", color="black", alpha=0.5)
    ax.axvspan(20, 30, color="gray", alpha=0.12, label=r"$20^\circ$--$30^\circ$ regime")

    cmap = plt.get_cmap("viridis")
    dims = sorted(dim_to_theory.keys())

    for idx, dim in enumerate(dims):
        color = cmap(idx / max(1, len(dims) - 1))

        ts = sorted([p for p in dim_to_theory[dim] if p.bits == bit], key=lambda p: p.angle_deg)
        es = sorted([p for p in dim_to_empirical.get(dim, []) if p.bits == bit], key=lambda p: p.angle_deg)

        ax.plot(
            [p.angle_deg for p in ts],
            [p.distortion_deg_theory for p in ts],
            color=color,
            label=fr"$d={dim}$ transfer",
        )

        if es:
            ax.errorbar(
                [p.angle_deg for p in es],
                [p.mean_distortion_deg for p in es],
                yerr=[p.std_distortion_deg for p in es],
                fmt="o",
                color=color,
                markersize=2.0,
                capsize=1.3,
                elinewidth=0.65,
                alpha=0.35,
            )

    ax.set_xlim(0, 90)
    ax.set_xticks(np.arange(0, 91, 10))
    ax.set_xlabel(r"Original angle $\theta$ (degrees)")
    ax.set_ylabel(r"Angular distortion $\arccos(C_b(\cos\theta))-\theta$ (degrees)")
    ax.set_title(fr"Dimension sweep for INT{bit} quantization")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=True, ncol=2)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_arcsine_alignment(output: Path, n: int = 2_000_000, seed: int = 0) -> None:
    setup_pub_style()
    ensure_parent(output)

    rng = np.random.default_rng(seed)
    rhos = np.linspace(-0.99, 0.99, 101)
    mc: List[float] = []

    for rho in rhos:
        x = rng.normal(size=n)
        z = rng.normal(size=n)
        y = rho * x + np.sqrt(1.0 - rho ** 2) * z
        mc.append(float(np.mean(np.sign(x) * np.sign(y))))

    theory = (2.0 / np.pi) * np.arcsin(rhos)

    fig, ax = plt.subplots(figsize=(6.4, 4.3))

    ax.plot(rhos, theory, label=r"$\frac{2}{\pi}\arcsin(\rho)$")
    ax.scatter(rhos, mc, s=10, alpha=0.55, label="Gaussian MC")

    ax.set_xlabel(r"True Gaussian correlation $\rho$")
    ax.set_ylabel("Sign-quantized correlation")
    ax.set_title("Arcsine-law alignment")
    ax.grid(alpha=0.22)
    ax.legend(frameon=True)

    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_fp_formats(
    path: Path,
    csv_path: Path,
    dim: int,
    trials: int,
    angles: Sequence[float],
    theory_samples: int,
    seed: int,
    formats: Sequence[FPFormat],
    run_empirical: bool = True,
) -> None:
    setup_pub_style()
    ensure_parent(path)

    fig, ax = plt.subplots(figsize=(7.8, 5.2))

    ax.axhline(0.0, linewidth=1, linestyle="--", color="black", alpha=0.5)
    ax.axvspan(20, 30, color="gray", alpha=0.12, label=r"$20^\circ$--$30^\circ$ regime")

    cmap = plt.get_cmap("tab10")
    rows: List[List[object]] = []

    marker_angles = list(angles)

    for idx, fmt in enumerate(formats):
        color = cmap(idx % 10)

        y_theory: List[float] = []
        for a in angles:
            val = fp_theory_mc(
                a,
                fmt,
                n=theory_samples,
                seed=seed + idx * 100_000 + int(round(a * 10)),
            )
            y_theory.append(val)

        ax.plot(
            angles,
            y_theory,
            color=color,
            linewidth=2.1,
            label=fmt.name,
        )

        mc_by_angle: Dict[float, Tuple[object, object, object]] = {}
        if run_empirical:
            rng = np.random.default_rng(seed + idx * 777)
            for a in marker_angles:
                m, s, n_valid = empirical_fp_for_angle(a, fmt, dim, trials, rng)
                mc_by_angle[a] = (m, s, n_valid)

            ax.errorbar(
                marker_angles,
                [mc_by_angle[a][0] for a in marker_angles],
                yerr=[mc_by_angle[a][1] for a in marker_angles],
                fmt="o",
                color=color,
                markersize=2.5,
                capsize=1.5,
                elinewidth=0.7,
                alpha=0.45,
            )

        for a, y_t in zip(angles, y_theory):
            m, s, n_valid = mc_by_angle.get(a, ("", "", ""))
            rows.append(
                [
                    fmt.name,
                    fmt.mode,
                    fmt.total_bits,
                    fmt.exp_bits,
                    fmt.mant_bits,
                    fmt.bias,
                    fmt.emin,
                    fmt.emax,
                    fmt.min_normal,
                    fmt.min_positive_quantum,
                    fmt.max_finite,
                    a,
                    y_t,
                    m,
                    s,
                    n_valid,
                    dim,
                ]
            )

    ax.set_xlim(min(0.0, min(angles)), max(90.0, max(angles)))
    ax.set_xticks(np.arange(0, 91, 10))
    ax.set_xlabel(r"Original angle $\theta$ (degrees)")
    ax.set_ylabel(r"Angular distortion $\arccos(C_{\rm fp}(\cos\theta))-\theta$ (degrees)")
    ax.set_title("Floating-point fake-quant angular distortion")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=True, ncol=2)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

    write_fp_csv(csv_path, rows)


def path_with_fp_suffix(path: Path, total_bits: int) -> Path:
    return path.with_name(f"{path.stem}_fp{total_bits}{path.suffix}")


def grouped_by_total_bits(formats: Sequence[FPFormat]) -> Dict[int, List[FPFormat]]:
    groups: Dict[int, List[FPFormat]] = {}
    for fmt in formats:
        groups.setdefault(fmt.total_bits, []).append(fmt)
    return dict(sorted(groups.items()))


def path_with_bit_suffix(path: Path, bits: int) -> Path:
    return path.with_name(f"{path.stem}_bit{bits}{path.suffix}")


def write_int_fp_compare_csv(path: Path, rows: Sequence[Sequence[object]]) -> None:
    ensure_parent(path)
    with Path(path).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "family",
                "format",
                "total_bits",
                "mode",
                "exp_bits",
                "mant_bits",
                "bias",
                "emin",
                "emax",
                "min_normal",
                "min_positive_quantum",
                "max_finite",
                "angle_deg",
                "distortion_deg_theory",
                "mean_distortion_deg_empirical",
                "std_distortion_deg_empirical",
                "valid_trials",
                "dim",
                "scale_mode",
                "clip_sigma",
            ]
        )
        for row in rows:
            w.writerow(row)


def plot_int_fp_comparison_by_total_bits(
    output: Path,
    csv_path: Path,
    bits_list: Sequence[int],
    fp_formats: Sequence[FPFormat],
    dim: int,
    trials: int,
    angles: Sequence[float],
    fp_theory_samples: int,
    seed: int,
    scale_mode: str,
    clip_sigma: float,
    quad_nodes: int,
    run_int_empirical: bool = True,
    run_fp_empirical: bool = True,
) -> None:
    """Write one overlay plot per total bit-width: INTB versus all FPB layouts.

    The INT curve uses the analytic Gaussian-transfer calculation. The FP curves
    use the same fake-FP theory-Monte-Carlo path as plot_fp_formats(). Empirical
    markers are optional and controlled separately for INT and FP.
    """

    setup_pub_style()
    formats_by_bits = grouped_by_total_bits(fp_formats)
    requested_bits = list(dict.fromkeys(int(b) for b in bits_list))

    wrote_any = False
    for bits in requested_bits:
        group = formats_by_bits.get(bits, [])
        if not group:
            print(f"Skipping INT{bits} vs FP{bits}: no FP{bits} formats were selected.")
            continue

        out_path = path_with_bit_suffix(output, bits)
        out_csv = path_with_bit_suffix(csv_path, bits)
        ensure_parent(out_path)

        print(f"Writing INT{bits} vs FP{bits} comparison plot: {out_path}")
        print(f"Writing INT{bits} vs FP{bits} comparison CSV:  {out_csv}")

        fig, ax = plt.subplots(figsize=(8.2, 5.3))
        ax.axhline(0.0, linewidth=1, linestyle="--", color="black", alpha=0.5)
        ax.axvspan(20, 30, color="gray", alpha=0.12, label=r"$20^\circ$--$30^\circ$ regime")

        rows: List[List[object]] = []
        cmap = plt.get_cmap("tab10")

        # Integer theory and empirical markers.
        int_theory: List[TheoryPoint] = []
        for a in angles:
            int_theory.append(
                theory_for_angle(
                    a,
                    bits,
                    dim,
                    scale_mode,
                    clip_sigma,
                    quad_nodes,
                )
            )

        ax.plot(
            [p.angle_deg for p in int_theory],
            [p.distortion_deg_theory for p in int_theory],
            color=cmap(0),
            linewidth=2.4,
            linestyle="-",
            label=f"INT{bits} transfer",
        )

        int_emp_by_angle: Dict[float, Tuple[object, object, object]] = {}
        if run_int_empirical:
            rng_int = np.random.default_rng(seed + bits * 10_003)
            int_emp: List[EmpiricalPoint] = []
            for a in angles:
                int_emp.append(empirical_for_angle(a, bits, dim, trials, scale_mode, clip_sigma, rng_int))
            int_emp_by_angle = {p.angle_deg: (p.mean_distortion_deg, p.std_distortion_deg, p.valid_trials) for p in int_emp}

            ax.errorbar(
                [p.angle_deg for p in int_emp],
                [p.mean_distortion_deg for p in int_emp],
                yerr=[p.std_distortion_deg for p in int_emp],
                fmt="o",
                color=cmap(0),
                markersize=2.5,
                capsize=1.5,
                elinewidth=0.7,
                alpha=0.45,
                label=f"INT{bits} MC",
            )

        for pnt in int_theory:
            m, s, n_valid = int_emp_by_angle.get(pnt.angle_deg, ("", "", ""))
            rows.append(
                [
                    "INT",
                    f"INT{bits}",
                    bits,
                    "symmetric-code",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    pnt.angle_deg,
                    pnt.distortion_deg_theory,
                    m,
                    s,
                    n_valid,
                    dim,
                    scale_mode,
                    clip_sigma,
                ]
            )

        # FP theory and empirical markers.
        for idx, fmt in enumerate(group, start=1):
            color = cmap(idx % 10)
            y_theory: List[float] = []
            for a in angles:
                y_theory.append(
                    fp_theory_mc(
                        a,
                        fmt,
                        n=fp_theory_samples,
                        seed=seed + bits * 1_000_000 + idx * 100_000 + int(round(a * 10)),
                    )
                )

            ax.plot(
                angles,
                y_theory,
                color=color,
                linewidth=2.1,
                linestyle="-",
                label=fmt.name,
            )

            fp_emp_by_angle: Dict[float, Tuple[object, object, object]] = {}
            if run_fp_empirical:
                rng_fp = np.random.default_rng(seed + bits * 20_003 + idx * 777)
                for a in angles:
                    fp_emp_by_angle[a] = empirical_fp_for_angle(a, fmt, dim, trials, rng_fp)

                ax.errorbar(
                    angles,
                    [fp_emp_by_angle[a][0] for a in angles],
                    yerr=[fp_emp_by_angle[a][1] for a in angles],
                    fmt="o",
                    color=color,
                    markersize=2.5,
                    capsize=1.5,
                    elinewidth=0.7,
                    alpha=0.45,
                )

            for a, y_t in zip(angles, y_theory):
                m, s, n_valid = fp_emp_by_angle.get(a, ("", "", ""))
                rows.append(
                    [
                        "FP",
                        fmt.name,
                        fmt.total_bits,
                        fmt.mode,
                        fmt.exp_bits,
                        fmt.mant_bits,
                        fmt.bias,
                        fmt.emin,
                        fmt.emax,
                        fmt.min_normal,
                        fmt.min_positive_quantum,
                        fmt.max_finite,
                        a,
                        y_t,
                        m,
                        s,
                        n_valid,
                        dim,
                        scale_mode,
                        clip_sigma,
                    ]
                )

        ax.set_xlim(min(0.0, min(angles)), max(90.0, max(angles)))
        ax.set_xticks(np.arange(0, 91, 10))
        ax.set_xlabel(r"Original angle $\theta$ (degrees)")
        ax.set_ylabel(r"Angular distortion $\hat\theta-\theta$ (degrees)")
        ax.set_title(f"INT{bits} versus FP{bits} angular distortion")
        ax.grid(True, alpha=0.22)
        ax.legend(frameon=True, ncol=2)

        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

        write_int_fp_compare_csv(out_csv, rows)
        wrote_any = True

    if not wrote_any:
        raise ValueError(
            "No INT-vs-FP comparison plots were written. Make sure --compare-bits/--bits "
            "matches at least one selected FP total bit-width."
        )


def print_fp_format_table(formats: Sequence[FPFormat]) -> None:
    headers = ["format", "mode", "bits", "E", "M", "bias", "emin", "emax", "min_normal", "quantum", "max_finite"]
    rows = []
    for fmt in formats:
        rows.append(
            [
                fmt.name,
                fmt.mode,
                str(fmt.total_bits),
                str(fmt.exp_bits),
                str(fmt.mant_bits),
                str(fmt.bias),
                str(fmt.emin),
                str(fmt.emax),
                f"{fmt.min_normal:g}",
                f"{fmt.min_positive_quantum:g}",
                f"{fmt.max_finite:g}",
            ]
        )

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))


# -----------------------------------------------------------------------------
# Self-test helpers
# -----------------------------------------------------------------------------


def run_self_tests() -> None:
    """Cheap correctness checks for the debugged fake-FP path and parser."""

    e4m3 = make_fp_format(4, 3, "ieee")
    e5m2 = make_fp_format(5, 2, "ieee")
    e2m1_ieee = make_fp_format(2, 1, "ieee")
    e2m1_finite = make_fp_format(2, 1, "finite")
    e4m3fn = make_fp_format(4, 3, "fn")

    checks = [
        (fake_fp_quantize(np.array([1e30]), e4m3)[0], 240.0, "E4M3 IEEE-like max finite"),
        (fake_fp_quantize(np.array([1e30]), e5m2)[0], 57344.0, "E5M2 IEEE-like max finite"),
        (fake_fp_quantize(np.array([1e30]), e2m1_ieee)[0], 3.0, "E2M1 IEEE-like max finite"),
        (fake_fp_quantize(np.array([1e30]), e2m1_finite)[0], 6.0, "E2M1 finite max finite"),
        (fake_fp_quantize(np.array([1e30]), e4m3fn)[0], 448.0, "E4M3FN max finite"),
        (fake_fp_quantize(np.array([255.0]), e4m3)[0], 240.0, "E4M3 IEEE top overflow saturates"),
        (fake_fp_quantize(np.array([float("inf")]), e4m3)[0], 240.0, "infinity saturates"),
    ]

    for got, expected, label in checks:
        if not np.isclose(got, expected, rtol=0.0, atol=0.0):
            raise AssertionError(f"{label}: got {got}, expected {expected}")

    nan_val = fake_fp_quantize(np.array([float("nan")]), e4m3)[0]
    if not np.isnan(nan_val):
        raise AssertionError("NaN should be preserved")

    parsed = parse_fp_format("fp8:e4m3")
    if not (parsed.exp_bits == 4 and parsed.mant_bits == 3 and parsed.total_bits == 8):
        raise AssertionError("parse fp8:e4m3 failed")

    parsed_fn = parse_fp_format("fp8:e4m3fn")
    if parsed_fn.mode != "fn" or not np.isclose(parsed_fn.max_finite, 448.0):
        raise AssertionError("parse fp8:e4m3fn failed")

    parsed_f4 = parse_fp_format("float4e2m1", default_mode="finite")
    if parsed_f4.mode != "finite" or parsed_f4.total_bits != 4 or not np.isclose(parsed_f4.max_finite, 6.0):
        raise AssertionError("parse float4e2m1 failed")

    try:
        parse_fp_format("fp6:e4m3")
    except ValueError:
        pass
    else:
        raise AssertionError("fp6:e4m3 should reject inconsistent prefix")

    try:
        parse_fp_format("e1m4")
    except ValueError:
        pass
    else:
        raise AssertionError("e1m4 should reject in default IEEE-like mode")

    # Explicit all-finite E1M4 should be valid.
    finite_e1m4 = parse_fp_format("finite:e1m4")
    if finite_e1m4.mode != "finite" or finite_e1m4.exp_bits != 1:
        raise AssertionError("finite:e1m4 should parse")

    print("Self-tests passed.")


# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Integer and fake-FP angular distortion experiments with debugged custom FP formats."
    )

    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--dims", type=int, nargs="+", default=None)
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--bits", type=int, nargs="+", default=[3, 4, 5])

    parser.add_argument("--angles", type=float, nargs="*", default=None)
    parser.add_argument("--angles-start", type=float, default=0.0)
    parser.add_argument("--angles-stop", type=float, default=90.0)
    parser.add_argument("--angles-step", type=float, default=1.0)

    parser.add_argument("--scale-mode", choices=["fixed", "std", "maxabs"], default="fixed")
    parser.add_argument("--clip-sigma", type=float, default=3.0)
    parser.add_argument("--quad-nodes", type=int, default=96)
    parser.add_argument("--seed", type=int, default=12345)

    parser.add_argument("--output", type=Path, default=Path("gaussian_transfer_distortion.pdf"))
    parser.add_argument("--csv", type=Path, default=Path("gaussian_transfer_distortion.csv"))

    parser.add_argument("--no-monte-carlo", action="store_true")
    parser.add_argument("--no-arcsine-plot", action="store_true")
    parser.add_argument("--no-dim-sweep-plots", action="store_true")
    parser.add_argument("--arcsine-samples", type=int, default=2_000_000)

    parser.add_argument("--plot-fp", action="store_true")
    parser.add_argument("--fp-only", action="store_true", help="Run only the FP experiment; implies --plot-fp and skips integer/arcsine outputs.")
    parser.add_argument("--fp-output", type=Path, default=Path("fp_format_angular_distortion.pdf"))
    parser.add_argument("--fp-csv", type=Path, default=Path("fp_format_angular_distortion.csv"))
    parser.add_argument("--fp-theory-samples", type=int, default=500_000)
    parser.add_argument("--fp-formats", type=str, nargs="*", default=None)
    parser.add_argument(
        "--fp-preset",
        choices=["debugged-ieee", "ml", "legacy-finite"],
        default="debugged-ieee",
        help="Default FP format set used when --fp-formats is omitted.",
    )
    parser.add_argument(
        "--fp-default-mode",
        choices=["ieee", "finite", "fn"],
        default="ieee",
        help="Mode for custom --fp-formats specs that do not specify a mode.",
    )
    parser.add_argument(
        "--fp-split-by-total-bits",
        action="store_true",
        help="Write one FP PDF/CSV per total bit-width, e.g. *_fp8.pdf and *_fp4.pdf.",
    )
    parser.add_argument("--no-fp-monte-carlo", action="store_true")
    parser.add_argument(
        "--compare-int-fp-by-total-bits",
        action="store_true",
        help="Write one overlay PDF/CSV per bit-width, e.g. INT6 versus all selected FP6 formats.",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Only write the INT-vs-FP comparison outputs; skip the standalone INT, FP, and arcsine plots.",
    )
    parser.add_argument("--compare-output", type=Path, default=Path("int_vs_fp_angular_distortion.pdf"))
    parser.add_argument("--compare-csv", type=Path, default=Path("int_vs_fp_angular_distortion.csv"))
    parser.add_argument(
        "--compare-bits",
        type=int,
        nargs="*",
        default=None,
        help="Bit-widths to compare. Defaults to --bits, so use --bits 6 8 for INT6/FP6 and INT8/FP8.",
    )
    parser.add_argument("--list-fp-formats", action="store_true", help="Print parsed/default FP format settings and exit.")

    parser.add_argument("--self-test", action="store_true", help="Run cheap internal checks and exit.")

    return parser


def validate_args(args: argparse.Namespace) -> None:
    dims = args.dims if args.dims is not None else [args.dim]

    if any(d < 3 for d in dims):
        raise ValueError("all dimensions must be at least 3")
    if args.trials < 0:
        raise ValueError("--trials must be non-negative")
    if args.fp_theory_samples <= 0:
        raise ValueError("--fp-theory-samples must be positive")
    if args.quad_nodes <= 0:
        raise ValueError("--quad-nodes must be positive")
    if args.clip_sigma <= 0:
        raise ValueError("--clip-sigma must be positive")
    if any(b < 2 for b in args.bits):
        raise ValueError("all --bits entries must be >= 2")
    if args.compare_bits is not None and any(b < 2 for b in args.compare_bits):
        raise ValueError("all --compare-bits entries must be >= 2")
    if args.compare_int_fp_by_total_bits and args.fp_only:
        raise ValueError("--compare-int-fp-by-total-bits needs integer curves, so omit --fp-only.")
    if args.arcsine_samples <= 0:
        raise ValueError("--arcsine-samples must be positive")


def run_integer_experiments(args: argparse.Namespace, dims: Sequence[int], angles: Sequence[float]) -> Tuple[Dict[int, List[TheoryPoint]], Dict[int, List[EmpiricalPoint]]]:
    all_theory: Dict[int, List[TheoryPoint]] = {}
    all_empirical: Dict[int, List[EmpiricalPoint]] = {}

    for dim in dims:
        print(f"\n=== Running integer quantization dimension d={dim} ===")

        args.dim = dim
        rng = np.random.default_rng(args.seed)

        theory: List[TheoryPoint] = []
        empirical: List[EmpiricalPoint] = []

        print("Computing integer Gaussian transfer theory...")
        for bits in args.bits:
            for angle in angles:
                theory.append(
                    theory_for_angle(
                        angle,
                        bits,
                        dim,
                        args.scale_mode,
                        args.clip_sigma,
                        args.quad_nodes,
                    )
                )

        if not args.no_monte_carlo:
            print("Running integer Monte Carlo...")
            for bits in args.bits:
                for angle in angles:
                    empirical.append(
                        empirical_for_angle(
                            angle,
                            bits,
                            dim,
                            args.trials,
                            args.scale_mode,
                            args.clip_sigma,
                            rng,
                        )
                    )

        all_theory[dim] = theory
        all_empirical[dim] = empirical

        out_path = args.output.with_name(f"{args.output.stem}_d{dim}{args.output.suffix}")
        csv_path = args.csv.with_name(f"{args.csv.stem}_d{dim}{args.csv.suffix}")

        print(f"Writing CSV: {csv_path}")
        write_csv(csv_path, theory, empirical, args)

        print(f"Writing plot: {out_path}")
        plot_results(out_path, args.bits, angles, theory, empirical, args)

    if len(dims) > 1 and not args.no_dim_sweep_plots:
        for bits in args.bits:
            sweep_path = args.output.with_name(f"{args.output.stem}_INT{bits}_dimension_sweep{args.output.suffix}")
            print(f"Writing dimension sweep plot: {sweep_path}")
            plot_dimension_sweep_by_bit(sweep_path, bits, all_theory, all_empirical)

    return all_theory, all_empirical


def resolve_output_under_base(base_output: Path, requested: Path) -> Path:
    if requested.is_absolute():
        return requested
    return base_output.parent / requested


def run_fp_experiments(args: argparse.Namespace, dims: Sequence[int], angles: Sequence[float], fp_formats: Sequence[FPFormat]) -> None:
    fp_dim = max(dims)
    fp_path = resolve_output_under_base(args.output, args.fp_output)
    fp_csv_path = resolve_output_under_base(args.output, args.fp_csv)

    if args.fp_split_by_total_bits:
        for total_bits, group in grouped_by_total_bits(fp_formats).items():
            group_path = path_with_fp_suffix(fp_path, total_bits)
            group_csv_path = path_with_fp_suffix(fp_csv_path, total_bits)
            print(f"Writing FP{total_bits} fake-quant plot: {group_path}")
            print(f"Writing FP{total_bits} fake-quant CSV: {group_csv_path}")
            plot_fp_formats(
                group_path,
                group_csv_path,
                dim=fp_dim,
                trials=args.trials,
                angles=angles,
                theory_samples=args.fp_theory_samples,
                seed=args.seed,
                formats=group,
                run_empirical=not args.no_fp_monte_carlo,
            )
        return

    print(f"Writing floating-point fake-quant plot: {fp_path}")
    print(f"Writing floating-point fake-quant CSV: {fp_csv_path}")
    plot_fp_formats(
        fp_path,
        fp_csv_path,
        dim=fp_dim,
        trials=args.trials,
        angles=angles,
        theory_samples=args.fp_theory_samples,
        seed=args.seed,
        formats=fp_formats,
        run_empirical=not args.no_fp_monte_carlo,
    )


def run_int_fp_comparison(args: argparse.Namespace, dims: Sequence[int], angles: Sequence[float], fp_formats: Sequence[FPFormat]) -> None:
    compare_dim = max(dims)
    compare_bits = args.compare_bits if args.compare_bits is not None and len(args.compare_bits) > 0 else args.bits
    compare_path = resolve_output_under_base(args.output, args.compare_output)
    compare_csv_path = resolve_output_under_base(args.output, args.compare_csv)

    plot_int_fp_comparison_by_total_bits(
        output=compare_path,
        csv_path=compare_csv_path,
        bits_list=compare_bits,
        fp_formats=fp_formats,
        dim=compare_dim,
        trials=args.trials,
        angles=angles,
        fp_theory_samples=args.fp_theory_samples,
        seed=args.seed,
        scale_mode=args.scale_mode,
        clip_sigma=args.clip_sigma,
        quad_nodes=args.quad_nodes,
        run_int_empirical=not args.no_monte_carlo,
        run_fp_empirical=not args.no_fp_monte_carlo,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.self_test:
        run_self_tests()
        return 0

    if args.compare_only:
        args.compare_int_fp_by_total_bits = True

    if args.fp_only:
        args.plot_fp = True

    if args.compare_int_fp_by_total_bits:
        args.plot_fp = True

    validate_args(args)

    dims = args.dims if args.dims is not None else [args.dim]
    angles = parse_angles(args)

    fp_formats = parse_fp_formats(args.fp_formats, preset=args.fp_preset, default_mode=args.fp_default_mode)

    if args.list_fp_formats:
        print_fp_format_table(fp_formats)
        return 0

    if not args.fp_only and not args.compare_only:
        run_integer_experiments(args, dims, angles)

    if args.plot_fp and not args.compare_only:
        run_fp_experiments(args, dims, angles, fp_formats)

    if args.compare_int_fp_by_total_bits:
        run_int_fp_comparison(args, dims, angles, fp_formats)

    if not args.no_arcsine_plot and not args.fp_only and not args.compare_only:
        arcsine_path = args.output.parent / "arcsine_alignment.pdf"
        print(f"Writing arcsine-law plot: {arcsine_path}")
        plot_arcsine_alignment(arcsine_path, n=args.arcsine_samples)

    print("Done.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
