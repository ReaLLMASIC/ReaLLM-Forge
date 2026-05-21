#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from token_quant_angle_comparison import run_analysis


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Month quantization angle comparison")
    p.add_argument("--model", default="google/gemma-3-270m")
    p.add_argument("--embedding-source", choices=["input", "lm_head"], default="input")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-dir", default="./gemma_month_quant_angles")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    run_analysis("months", a.model, a.embedding_source, a.device, Path(a.output_dir))


if __name__ == "__main__":
    main()
