#!/usr/bin/env python3
"""Utility to estimate epoch-level token and iteration counts."""

import argparse
import math
import os
import pickle
from typing import Tuple

_DTYPE_SIZES = {
    'uint16': 2,
    'uint32': 4,
}


def _infer_dtype(meta_path: str) -> str:
    if not os.path.exists(meta_path):
        return 'uint16'
    with open(meta_path, 'rb') as meta_file:
        meta = pickle.load(meta_file)
    vocab_size = meta.get('vocab_size')
    if vocab_size == 100277:
        return 'uint32'
    return 'uint16'


def _tokens_in_dataset(data_dir: str, dataset: str) -> int:
    train_path = os.path.join(data_dir, dataset, 'train.bin')
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Could not find train.bin for dataset '{dataset}' in {train_path}")
    meta_path = os.path.join(data_dir, dataset, 'meta.pkl')
    dtype = _infer_dtype(meta_path)
    dtype_size = _DTYPE_SIZES[dtype]
    total_bytes = os.path.getsize(train_path)
    if total_bytes % dtype_size != 0:
        raise ValueError(
            f"Dataset {dataset} size {total_bytes} is not divisible by dtype size {dtype_size}."
        )
    return total_bytes // dtype_size


def _format_iterations(tokens_per_epoch: int, tokens_per_iteration: int) -> Tuple[float, int]:
    if tokens_per_iteration <= 0:
        raise ValueError('Tokens per iteration must be positive.')
    iterations = tokens_per_epoch / tokens_per_iteration
    return iterations, int(math.ceil(iterations))


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Compute tokens-per-epoch and iteration estimates for one or more datasets.',
    )
    parser.add_argument('datasets', nargs='+', help='Dataset names relative to the --data-dir directory.')
    parser.add_argument('--data-dir', default='data', help='Root directory that contains dataset folders (default: data).')
    parser.add_argument('--batch-size', type=int, required=True, help='Batch size per gradient step.')
    parser.add_argument('--block-size', type=int, required=True, help='Block size / context length.')
    parser.add_argument(
        '--gradient-accumulation-steps',
        type=int,
        default=1,
        help='Number of gradient accumulation micro-steps (default: 1).',
    )
    parser.add_argument(
        '--world-size',
        type=int,
        default=1,
        help='Number of distributed workers contributing tokens (default: 1).',
    )

    args = parser.parse_args()

    tokens_per_iteration = (
        args.batch_size * args.block_size * args.gradient_accumulation_steps * args.world_size
    )
    if tokens_per_iteration <= 0:
        raise SystemExit('Tokens per iteration must be positive. Check the provided arguments.')

    print(f"Tokens per iteration: {tokens_per_iteration}")

    for dataset in args.datasets:
        tokens_per_epoch = _tokens_in_dataset(args.data_dir, dataset)
        fractional_iters, ceil_iters = _format_iterations(tokens_per_epoch, tokens_per_iteration)
        print('\nDataset:', dataset)
        print(f'  Tokens per epoch: {tokens_per_epoch:,}')
        print(f'  Iterations per epoch (exact): {fractional_iters:.4f}')
        print(f'  Iterations per epoch (ceiling): {ceil_iters}')


if __name__ == '__main__':
    main()
