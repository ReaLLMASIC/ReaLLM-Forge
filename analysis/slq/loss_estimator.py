#!/usr/bin/env python3
"""Predict validation loss from angular distortion for vector quantization.

Default model (two-sided random perturbation on both vectors):
    L(theta) = L0 + (2*K/D) * sin(theta)^2

Alternative model (one-sided perturbation on only one vector):
    L(theta) = L0 + (4*K/D) * sin(theta/2)^2

The default angle table is an approximate read-off from the user's
"Mean angle vs. bit-width" plot for VECTOR quantization:
    int8 -> 0.45 deg
    int7 -> 0.90 deg
    int6 -> 1.80 deg
    int5 -> 3.70 deg
    int4 -> 7.80 deg
    int3 -> 17.50 deg

Replace these with your measured angles via --angles if you have exact values.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List

DEFAULT_VECTOR_ANGLES: Dict[int, float] = {
    8: 0.45,
    7: 0.90,
    6: 1.80,
    5: 3.70,
    4: 7.80,
    3: 17.50,
}
DEFAULT_BITS: List[int] = [8, 7, 6, 5, 4, 3]


@dataclass(frozen=True)
class PredictionRow:
    bits: int
    angle_deg: float
    angle_rad: float
    penalty_factor: float
    delta_loss: float
    predicted_loss: float


def positive_float(value: str) -> float:
    """Argparse type enforcing a positive float."""
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid float: {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"Expected a positive value, got {parsed}")
    return parsed


def non_negative_float(value: str) -> float:
    """Argparse type enforcing a non-negative float."""
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid float: {value!r}") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"Expected a non-negative value, got {parsed}")
    return parsed


def penalty_factor(angle_rad: float, mode: str) -> float:
    """Return the angular penalty factor.

    two-sided: 2 * sin(theta)^2
    one-sided: 4 * sin(theta/2)^2
    """
    if mode == "two-sided":
        return 2.0 * math.sin(angle_rad) ** 2
    if mode == "one-sided":
        return 4.0 * math.sin(angle_rad / 2.0) ** 2
    raise ValueError(f"Unknown mode: {mode}")


def predict_loss(L0: float, K: float, D: float, angle_deg: float, mode: str) -> PredictionRow:
    """Compute predicted loss for one angle in degrees."""
    theta = math.radians(angle_deg)
    factor = penalty_factor(theta, mode)
    delta = (K / D) * factor
    return PredictionRow(
        bits=-1,
        angle_deg=angle_deg,
        angle_rad=theta,
        penalty_factor=factor,
        delta_loss=delta,
        predicted_loss=L0 + delta,
    )


def build_angle_table(bits: Iterable[int], custom_angles: List[float] | None) -> Dict[int, float]:
    """Build a mapping from bit-width to angle in degrees."""
    bits_list = list(bits)
    if custom_angles is None:
        return {b: DEFAULT_VECTOR_ANGLES[b] for b in bits_list}

    if len(custom_angles) != len(bits_list):
        raise ValueError(
            f"Expected {len(bits_list)} custom angles for bits {bits_list}, "
            f"but received {len(custom_angles)}"
        )
    return dict(zip(bits_list, custom_angles))


def format_table(rows: List[PredictionRow], mode: str, L0: float, K: float, D: float) -> str:
    """Render a plain-text table."""
    if mode == "two-sided":
        formula = "L(theta) = L0 + (2*K/D) * sin(theta)^2"
    else:
        formula = "L(theta) = L0 + (4*K/D) * sin(theta/2)^2"

    lines = [
        formula,
        f"Inputs: L0={L0:.6g}, K={K:.6g}, D={D:.6g}, mode={mode}",
        "",
        f"{'Bits':>4}  {'Angle(deg)':>10}  {'PenaltyFactor':>13}  {'DeltaLoss':>10}  {'PredLoss':>10}",
        f"{'-' * 4}  {'-' * 10}  {'-' * 13}  {'-' * 10}  {'-' * 10}",
    ]
    for row in rows:
        lines.append(
            f"{row.bits:>4d}  {row.angle_deg:>10.4f}  {row.penalty_factor:>13.6f}  "
            f"{row.delta_loss:>10.6f}  {row.predicted_loss:>10.6f}"
        )
    return "\n".join(lines)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict validation loss at vector-quantization bit-widths using an angular "
            "distortion model. Default angles are approximate values read from the plot "
            "for vector quantization at int8, int7, int6, int5, int4, and int3."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--L0", type=non_negative_float, required=True,
                        help="Baseline validation loss with zero distortion.")
    parser.add_argument("--K", type=non_negative_float, required=True,
                        help="Task/model sensitivity constant in the K/D term.")
    parser.add_argument("--D", type=positive_float, required=True,
                        help="Embedding dimension d.")
    parser.add_argument(
        "--bits",
        type=int,
        nargs="+",
        default=DEFAULT_BITS,
        help="Bit-widths to evaluate. Supported defaults are 8 7 6 5 4 3.",
    )
    parser.add_argument(
        "--angles",
        type=non_negative_float,
        nargs="+",
        default=None,
        help=(
            "Custom angles in degrees, in the same order as --bits. "
            "Example: --bits 8 7 6 5 4 3 --angles 0.45 0.90 1.80 3.70 7.80 17.50"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["two-sided", "one-sided"],
        default="two-sided",
        help=(
            "Use two-sided when both vectors are independently perturbed by theta. "
            "Use one-sided when only one vector is perturbed."
        ),
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    unsupported = [b for b in args.bits if b not in DEFAULT_VECTOR_ANGLES]
    if args.angles is None and unsupported:
        supported = " ".join(str(b) for b in sorted(DEFAULT_VECTOR_ANGLES, reverse=True))
        print(
            f"Error: default angles are only defined for bit-widths {supported}. "
            "Provide --angles to use custom values.",
            file=sys.stderr,
        )
        return 2

    try:
        angle_map = build_angle_table(args.bits, args.angles)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    rows: List[PredictionRow] = []
    for b in args.bits:
        row = predict_loss(args.L0, args.K, args.D, angle_map[b], args.mode)
        rows.append(
            PredictionRow(
                bits=b,
                angle_deg=row.angle_deg,
                angle_rad=row.angle_rad,
                penalty_factor=row.penalty_factor,
                delta_loss=row.delta_loss,
                predicted_loss=row.predicted_loss,
            )
        )

    rows.sort(key=lambda r: r.bits, reverse=True)
    print(format_table(rows, args.mode, args.L0, args.K, args.D))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

