#!/usr/bin/env python3
"""Run 10-Epoch Training of Full POS and Coarse POS models with Weighted & Unweighted Loss
using 256-Byte Fallback on Companion Stream, and Evaluate Downstream Tasks across Checkpoints.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda:0"
MAX_ITERS = 36330  # 10 epochs on 59M tokens (3,633 iters/epoch)
EVAL_ITERS = 50
MAX_EXAMPLES = 200

# 23 common factor lanes
COMMON_LANES = [
    "korean_pos_mc/script", "korean_pos_mc/choseong", "korean_pos_mc/jungseong", "korean_pos_mc/jongseong",
    "korean_pos_mc/jung_base1", "korean_pos_mc/jung_base2", "korean_pos_mc/jung_has_w", "korean_pos_mc/jung_has_y",
    "korean_pos_mc/jung_has_i", "korean_pos_mc/jong_base1", "korean_pos_mc/jong_base2", "korean_pos_mc/jong_base3",
    "korean_pos_mc/choseong_tense", "korean_pos_mc/choseong_aspirated", "korean_pos_mc/choseong_nasal_liquid",
    "korean_pos_mc/choseong_place", "korean_pos_mc/jung_height", "korean_pos_mc/jung_backness", "korean_pos_mc/jung_round",
    "korean_pos_mc/jong_complex", "korean_pos_mc/has_batchim", "korean_pos_mc/syllable_index_mod", "korean_pos_mc/codepoint_mod",
]

# 4 Experiment configurations
EXPERIMENTS = {
    "coarse_pos_weighted": {
        "description": "Coarse POS (17 tags) + Weighted Loss (w_char=1.0, w_pos=0.5, w_struct=0.05)",
        "out_dir": "./out_mc_coarse_pos_weighted_byte",
        "pos_lane": "korean_pos_mc/pos",
        "char_lane": "korean_pos_mc/char_byte",
        "loss_args": ["--structural_loss_weight", "0.05", "--pos_loss_weight", "0.5"],
    },
    "coarse_pos_unweighted": {
        "description": "Coarse POS (17 tags) + Unweighted Loss (w=1.0 for all lanes)",
        "out_dir": "./out_mc_coarse_pos_unweighted_byte",
        "pos_lane": "korean_pos_mc/pos",
        "char_lane": "korean_pos_mc/char_byte",
        "loss_args": ["--structural_loss_weight", "1.0", "--pos_loss_weight", "1.0"],
    },
    "full_pos_weighted": {
        "description": "Full POS (46 Sejong tags) + Weighted Loss (w_char=1.0, w_pos=0.5, w_struct=0.05)",
        "out_dir": "./out_mc_full_pos_weighted_byte",
        "pos_lane": "korean_pos_mc/pos_full",
        "char_lane": "korean_pos_mc/char_byte",
        "loss_args": ["--structural_loss_weight", "0.05", "--pos_loss_weight", "0.5"],
    },
    "full_pos_unweighted": {
        "description": "Full POS (46 Sejong tags) + Unweighted Loss (w=1.0 for all lanes)",
        "out_dir": "./out_mc_full_pos_unweighted_byte",
        "pos_lane": "korean_pos_mc/pos_full",
        "char_lane": "korean_pos_mc/char_byte",
        "loss_args": ["--structural_loss_weight", "1.0", "--pos_loss_weight", "1.0"],
    },
}

CHECKPOINT_EPOCHS = {
    "3_epochs": "10899.pt",
    "5_epochs": "18165.pt",
    "10_epochs": "ckpt.pt",
}


def train_single_experiment(exp_key: str, cfg: dict):
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    final_ckpt = os.path.join(out_dir, "ckpt.pt")

    print(f"\n==========================================================================")
    print(f" TRAINING: {exp_key.upper()}")
    print(f" Description: {cfg['description']}")
    print(f" Output Dir: {out_dir}")
    print(f" Iterations: {MAX_ITERS}")
    print(f" Milestones: 3ep (10899.pt), 5ep (18165.pt), 10ep (ckpt.pt)")
    print(f"==========================================================================")

    if os.path.exists(final_ckpt):
        print(f"Final checkpoint {final_ckpt} already exists. Skipping training.")
        return

    lanes = [*COMMON_LANES, cfg["pos_lane"], cfg["char_lane"]]

    cmd = [
        sys.executable, "train.py",
        "--dataset", cfg["char_lane"],
        "--training_mode", "multicontext",
        "--multicontext",
        "--multicontext_datasets", *lanes,
        *cfg["loss_args"],
        "--max_iters", str(MAX_ITERS),
        "--eval_iters", str(EVAL_ITERS),
        "--always_save_checkpoint",
        "--dropout", "0.1",
        "--device", DEVICE,
        "--out_dir", out_dir,
    ]

    t0 = time.time()
    res = subprocess.run(cmd, cwd=REPO_ROOT)
    t1 = time.time()
    if res.returncode != 0:
        raise RuntimeError(f"Training for {exp_key} failed with exit code {res.returncode}")
    print(f"Training for {exp_key} completed in {t1 - t0:.1f}s ({(t1 - t0)/60:.1f}m).")


def evaluate_single_experiment(exp_key: str, cfg: dict, max_examples: int = MAX_EXAMPLES) -> dict:
    out_dir = cfg["out_dir"]
    print(f"\n==========================================================================")
    print(f" EVALUATING: {exp_key.upper()}")
    print(f"==========================================================================")

    exp_results = {}

    for epoch_label, ckpt_name in CHECKPOINT_EPOCHS.items():
        ckpt_path = os.path.join(out_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"Warning: Checkpoint {ckpt_path} not found! Skipping {epoch_label}.")
            continue

        print(f"\n--- Evaluating {exp_key} @ {epoch_label} ({ckpt_name}) ---")
        epoch_res = {}

        # 1. Four Capabilities Benchmark (KLUE-NER, KLUE-DP, NSMC 80%, UnSmile 80%, KorMedMCQA)
        temp_4cap_json = f"temp_4cap_{exp_key}_{epoch_label}.json"
        cmd_4cap = [
            sys.executable, "benchmarks/run_four_capability_evals.py",
            "--ckpts", f"{epoch_label}:{ckpt_path}",
            "--device", DEVICE,
            "--max_examples", str(max_examples),
            "--output_json", temp_4cap_json,
        ]
        subprocess.run(cmd_4cap, cwd=REPO_ROOT, check=True)
        if os.path.exists(temp_4cap_json):
            with open(temp_4cap_json, "r", encoding="utf-8") as f:
                data_4cap = json.load(f)
            os.remove(temp_4cap_json)
            epoch_res.update(data_4cap.get(epoch_label, {}))

        # 2. Ko-HellaSwag Zero-Shot Reasoning
        temp_hs_json = f"temp_hs_{exp_key}_{epoch_label}.json"
        cmd_hs = [
            sys.executable, "benchmarks/run_ko_hellaswag.py",
            "--ckpt_path", ckpt_path,
            "--device", DEVICE,
            "--eval_all_norms",
            "--max_examples", str(max_examples),
            "--output_json", temp_hs_json,
        ]
        subprocess.run(cmd_hs, cwd=REPO_ROOT, check=True)
        if os.path.exists(temp_hs_json):
            with open(temp_hs_json, "r", encoding="utf-8") as f:
                data_hs = json.load(f)
            os.remove(temp_hs_json)
            epoch_res["ko_hellaswag"] = data_hs.get("accuracies", {})

        exp_results[epoch_label] = epoch_res

    return exp_results


def run_all(experiments_to_run=None, max_examples=MAX_EXAMPLES, output_json="pos_byte_experiments_results.json"):
    exp_keys = experiments_to_run or list(EXPERIMENTS.keys())
    all_results = {}

    # Phase 1: Train all models
    for key in exp_keys:
        train_single_experiment(key, EXPERIMENTS[key])

    # Phase 2: Evaluate all models across checkpoints
    for key in exp_keys:
        all_results[key] = {
            "config": EXPERIMENTS[key],
            "checkpoints": evaluate_single_experiment(key, EXPERIMENTS[key], max_examples=max_examples),
        }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nAll experimental results saved to {output_json}")
    return all_results


def main():
    p = argparse.ArgumentParser(description="Train & Evaluate Full/Coarse POS models with 256-Byte Fallback.")
    p.add_argument("--experiments", nargs="+", choices=list(EXPERIMENTS.keys()), default=None,
                   help="Specific experiment(s) to run (default: run all 4)")
    p.add_argument("--eval_only", action="store_true", help="Skip training and run evaluations only")
    p.add_argument("--max_examples", type=int, default=MAX_EXAMPLES, help="Max examples per evaluation dataset")
    p.add_argument("--output_json", type=str, default="pos_byte_experiments_results.json", help="Output JSON results file")
    args = p.parse_args()

    exp_keys = args.experiments or list(EXPERIMENTS.keys())

    if not args.eval_only:
        for key in exp_keys:
            train_single_experiment(key, EXPERIMENTS[key])

    all_results = {}
    for key in exp_keys:
        all_results[key] = {
            "config": EXPERIMENTS[key],
            "checkpoints": evaluate_single_experiment(key, EXPERIMENTS[key], max_examples=args.max_examples),
        }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nCompleted! Saved results to {args.output_json}")


if __name__ == "__main__":
    main()
