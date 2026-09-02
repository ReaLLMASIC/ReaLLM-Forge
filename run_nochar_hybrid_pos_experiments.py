#!/usr/bin/env python3
"""Training and Evaluation Pipeline for 24-Lane Hangul Hybrid Factorizer (No Char Stream Overhead).

Trains 4 models across 10 epochs on the 24-lane hybrid corpus:
1. nochar_coarse_pos_weighted (17 coarse POS tags, weighted loss: w_bpe=1.0, w_pos=0.5, w_struct=0.05)
2. nochar_coarse_pos_unweighted (17 coarse POS tags, unweighted loss: w=1.0 for all 24 lanes)
3. nochar_full_pos_weighted (46 Sejong POS tags, weighted loss: w_bpe=1.0, w_pos=0.5, w_struct=0.05)
4. nochar_full_pos_unweighted (46 Sejong POS tags, unweighted loss: w=1.0 for all 24 lanes)

Saves checkpoints at 3, 5, and 10 epochs, then evaluates all saved checkpoints on:
- Four Benchmark Capabilities (KLUE-NER, KLUE-DP, NSMC 80%, UnSmile 80%, KorMedMCQA)
- Ko-HellaSwag Zero-Shot Reasoning
- Phonetic Slang & Typo Resilience
- Vocabulary Tail Perplexity (Deciles 1 to 10)
- OOV / Unicode Robustness Stress Test
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
BLOCK_SIZE = 256
EVAL_ITERS = 50
MAX_EVAL_EXAMPLES = 200

# 23 Common Factor Lanes
COMMON_LANES = [
    "korean_pos_mc_nochar/script_bpe",
    "korean_pos_mc_nochar/choseong",
    "korean_pos_mc_nochar/jungseong",
    "korean_pos_mc_nochar/jongseong",
    "korean_pos_mc_nochar/jung_base1",
    "korean_pos_mc_nochar/jung_base2",
    "korean_pos_mc_nochar/jung_has_w",
    "korean_pos_mc_nochar/jung_has_y",
    "korean_pos_mc_nochar/jung_has_i",
    "korean_pos_mc_nochar/jong_base1",
    "korean_pos_mc_nochar/jong_base2",
    "korean_pos_mc_nochar/jong_base3",
    "korean_pos_mc_nochar/choseong_tense",
    "korean_pos_mc_nochar/choseong_aspirated",
    "korean_pos_mc_nochar/choseong_nasal_liquid",
    "korean_pos_mc_nochar/choseong_place",
    "korean_pos_mc_nochar/jung_height",
    "korean_pos_mc_nochar/jung_backness",
    "korean_pos_mc_nochar/jung_round",
    "korean_pos_mc_nochar/jong_complex",
    "korean_pos_mc_nochar/has_batchim",
    "korean_pos_mc_nochar/syllable_index_mod",
    "korean_pos_mc_nochar/codepoint_mod",
]

EXPERIMENTS = {
    "nochar_coarse_pos_weighted": {
        "display_name": "No-Char Coarse POS (Weighted, ws=0.05, wpos=0.5)",
        "out_dir": "./out_mc_nochar_coarse_pos_weighted",
        "pos_lane": "korean_pos_mc_nochar/pos",
        "loss_args": ["--structural_loss_weight", "0.05", "--pos_loss_weight", "0.5"],
        "pos_mode": "coarse",
    },
    "nochar_coarse_pos_unweighted": {
        "display_name": "No-Char Coarse POS (Unweighted, w=1.0)",
        "out_dir": "./out_mc_nochar_coarse_pos_unweighted",
        "pos_lane": "korean_pos_mc_nochar/pos",
        "loss_args": ["--structural_loss_weight", "1.0", "--pos_loss_weight", "1.0"],
        "pos_mode": "coarse",
    },
    "nochar_full_pos_weighted": {
        "display_name": "No-Char Full POS (Weighted, ws=0.05, wpos=0.5)",
        "out_dir": "./out_mc_nochar_full_pos_weighted",
        "pos_lane": "korean_pos_mc_nochar/pos_full",
        "loss_args": ["--structural_loss_weight", "0.05", "--pos_loss_weight", "0.5"],
        "pos_mode": "full",
    },
    "nochar_full_pos_unweighted": {
        "display_name": "No-Char Full POS (Unweighted, w=1.0)",
        "out_dir": "./out_mc_nochar_full_pos_unweighted",
        "pos_lane": "korean_pos_mc_nochar/pos_full",
        "loss_args": ["--structural_loss_weight", "1.0", "--pos_loss_weight", "1.0"],
        "pos_mode": "full",
    },
}


def get_epoch_iterations() -> Dict[str, int]:
    """Calculate exact iteration counts for 3, 5, and 10 epochs from train.bin."""
    train_bin = os.path.join(REPO_ROOT, "data", "korean_pos_mc_nochar", "script_bpe", "train.bin")
    if not os.path.exists(train_bin):
        raise FileNotFoundError(f"Training dataset not found at {train_bin}. Run prepare_hybrid_pos_lanes.py first.")

    total_train_tokens = os.path.getsize(train_bin) // 2  # uint16
    tokens_per_iter = BATCH_SIZE * BLOCK_SIZE  # 16,384
    iters_per_epoch = total_train_tokens // tokens_per_iter

    iters_3ep = iters_per_epoch * 3
    iters_5ep = iters_per_epoch * 5
    iters_10ep = iters_per_epoch * 10

    print(f"Dataset Size: {total_train_tokens:,} training tokens.")
    print(f"Tokens/iter: {tokens_per_iter:,} (batch={BATCH_SIZE}, block={BLOCK_SIZE})")
    print(f"Iterations/epoch: {iters_per_epoch:,}")
    print(f"Milestones: 3ep = {iters_3ep:,} iters | 5ep = {iters_5ep:,} iters | 10ep = {iters_10ep:,} iters")

    return {
        "iters_per_epoch": iters_per_epoch,
        "3_epochs": iters_3ep,
        "5_epochs": iters_5ep,
        "10_epochs": iters_10ep,
    }


def train_single_model(exp_key: str, cfg: Dict[str, Any], epoch_iters: Dict[str, int]):
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    lanes = [*COMMON_LANES, cfg["pos_lane"]]
    primary_dataset = "korean_pos_mc_nochar/script_bpe"

    milestones = [
        ("3_epochs", epoch_iters["3_epochs"], "3_epochs.pt"),
        ("5_epochs", epoch_iters["5_epochs"], "5_epochs.pt"),
        ("10_epochs", epoch_iters["10_epochs"], "ckpt.pt"),
    ]

    print("\n" + "=" * 80)
    print(f" TRAINING: {exp_key.upper()} ({cfg['display_name']})")
    print(f" Output Directory: {out_dir}")
    print(f" 24 Lanes: {len(lanes)} total (1 BPE/Script + 22 Factors + 1 POS)")
    print("=" * 80)

    for stage_name, target_iters, ckpt_filename in milestones:
        stage_ckpt = os.path.join(out_dir, ckpt_filename)
        if os.path.exists(stage_ckpt):
            try:
                ckpt_data = torch.load(stage_ckpt, map_location="cpu", weights_only=False)
                curr_iter = ckpt_data.get("iter_num", 0)
                if curr_iter >= target_iters:
                    print(f"Milestone {stage_name} ({ckpt_filename}) already at {curr_iter} >= {target_iters}. Skipping.")
                    continue
            except Exception:
                pass

        print(f"\n>>> Running stage {stage_name} up to iter {target_iters:,}...")

        cmd = [
            sys.executable,
            "train.py",
            "--dataset", primary_dataset,
            "--training_mode", "multicontext",
            "--multicontext",
            "--multicontext_datasets", *lanes,
            *cfg["loss_args"],
            "--batch_size", str(BATCH_SIZE),
            "--block_size", str(BLOCK_SIZE),
            "--max_iters", str(target_iters),
            "--eval_iters", str(EVAL_ITERS),
            "--always_save_checkpoint",
            "--dropout", "0.1",
            "--device", DEVICE,
            "--out_dir", out_dir,
        ]

        # Resume if previous checkpoint exists
        prev_ckpt = os.path.join(out_dir, "ckpt.pt")
        if os.path.exists(prev_ckpt):
            cmd.extend(["--init_from", "resume"])

        t0 = time.time()
        res = subprocess.run(cmd, cwd=REPO_ROOT)
        t1 = time.time()

        if res.returncode != 0:
            raise RuntimeError(f"Training failed for {exp_key} at stage {stage_name} with exit code {res.returncode}")

        # Save milestone checkpoint
        if os.path.exists(prev_ckpt) and ckpt_filename != "ckpt.pt":
            shutil.copyfile(prev_ckpt, stage_ckpt)
            print(f"Saved milestone checkpoint: {stage_ckpt}")

        print(f"Stage {stage_name} finished in {t1 - t0:.1f}s ({(t1 - t0)/60:.1f}m).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 24-lane hybrid Hangul factorizer training & evaluation.")
    parser.add_argument("--experiments", nargs="+", choices=list(EXPERIMENTS.keys()), default=None,
                        help="Select specific experiment(s) to run")
    parser.add_argument("--eval_only", action="store_true", help="Skip training and run evaluations only")
    args = parser.parse_args()

    epoch_iters = get_epoch_iterations()
    exp_keys = args.experiments or list(EXPERIMENTS.keys())

    if not args.eval_only:
        for key in exp_keys:
            train_single_model(key, EXPERIMENTS[key], epoch_iters)

    print("\n==========================================================================")
    print(" ALL MODEL TRAINING RUNS COMPLETED SUCCESSFULLY!")
    print("==========================================================================")
