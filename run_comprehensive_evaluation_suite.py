#!/usr/bin/env python3
"""Comprehensive Evaluation Suite across all 24-Lane Hybrid Hangul Factorizer Checkpoints and Baselines.

Evaluates 12 checkpoint variants:
- No-Char Coarse POS Weighted (3ep, 5ep, 10ep)
- No-Char Coarse POS Unweighted (3ep, 5ep, 10ep)
- No-Char Full POS Weighted (3ep, 5ep, 10ep)
- No-Char Full POS Unweighted (3ep, 5ep, 10ep)

And all 10 baseline models:
- Original Character Baseline (V=4,555)
- Byte-Fallback BPE Baseline (V=4,924)
- Pure Byte Baseline (V=256)
- Standard Full POS Weighted (char stream)
- Standard Full POS Unweighted (char stream)
- Standard Coarse POS Weighted (char stream)
- Byte Fallback Full POS Weighted (byte companion)
- Byte Fallback Full POS Unweighted (byte companion)
- Byte Fallback Coarse POS Weighted (byte companion)
- Byte Fallback Coarse POS Unweighted (byte companion)

Downstream Benchmarks:
1. Four Capabilities: KLUE-NER, KLUE-DP, NSMC (80% noise), UnSmile (80% noise), KorMedMCQA
2. Ko-HellaSwag Zero-Shot Reasoning
3. Phonetic Slang & Typo Resilience (Clean vs 80% Corrupted)
4. Vocabulary Tail Perplexity (Deciles 1 to 10)
5. OOV & Unicode Stress-Test
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List

import torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

EVAL_TARGETS = [
    # 24-Lane Hybrid Models (No Char Stream)
    {"id": "nochar_coarse_pos_weighted_3ep", "name": "No-Char Coarse POS Weighted (3ep)", "ckpt": "out_mc_nochar_coarse_pos_weighted/3_epochs.pt", "type": "hybrid"},
    {"id": "nochar_coarse_pos_weighted_5ep", "name": "No-Char Coarse POS Weighted (5ep)", "ckpt": "out_mc_nochar_coarse_pos_weighted/5_epochs.pt", "type": "hybrid"},
    {"id": "nochar_coarse_pos_weighted_10ep", "name": "No-Char Coarse POS Weighted (10ep)", "ckpt": "out_mc_nochar_coarse_pos_weighted/ckpt.pt", "type": "hybrid"},

    {"id": "nochar_coarse_pos_unweighted_3ep", "name": "No-Char Coarse POS Unweighted (3ep)", "ckpt": "out_mc_nochar_coarse_pos_unweighted/3_epochs.pt", "type": "hybrid"},
    {"id": "nochar_coarse_pos_unweighted_5ep", "name": "No-Char Coarse POS Unweighted (5ep)", "ckpt": "out_mc_nochar_coarse_pos_unweighted/5_epochs.pt", "type": "hybrid"},
    {"id": "nochar_coarse_pos_unweighted_10ep", "name": "No-Char Coarse POS Unweighted (10ep)", "ckpt": "out_mc_nochar_coarse_pos_unweighted/ckpt.pt", "type": "hybrid"},

    {"id": "nochar_full_pos_weighted_3ep", "name": "No-Char Full POS Weighted (3ep)", "ckpt": "out_mc_nochar_full_pos_weighted/3_epochs.pt", "type": "hybrid"},
    {"id": "nochar_full_pos_weighted_5ep", "name": "No-Char Full POS Weighted (5ep)", "ckpt": "out_mc_nochar_full_pos_weighted/5_epochs.pt", "type": "hybrid"},
    {"id": "nochar_full_pos_weighted_10ep", "name": "No-Char Full POS Weighted (10ep)", "ckpt": "out_mc_nochar_full_pos_weighted/ckpt.pt", "type": "hybrid"},

    {"id": "nochar_full_pos_unweighted_3ep", "name": "No-Char Full POS Unweighted (3ep)", "ckpt": "out_mc_nochar_full_pos_unweighted/3_epochs.pt", "type": "hybrid"},
    {"id": "nochar_full_pos_unweighted_5ep", "name": "No-Char Full POS Unweighted (5ep)", "ckpt": "out_mc_nochar_full_pos_unweighted/5_epochs.pt", "type": "hybrid"},
    {"id": "nochar_full_pos_unweighted_10ep", "name": "No-Char Full POS Unweighted (10ep)", "ckpt": "out_mc_nochar_full_pos_unweighted/ckpt.pt", "type": "hybrid"},

    # Baselines (10 Epochs)
    {"id": "baseline_character", "name": "Original Character Baseline (V=4,555)", "ckpt": "out_baseline_korean_pos_full1m/ckpt.pt", "type": "baseline"},
    {"id": "baseline_char_bpe", "name": "Byte-Fallback BPE Baseline (V=4,924)", "ckpt": "out_baseline_char_bpe/ckpt.pt", "type": "baseline"},
    {"id": "baseline_pure_byte", "name": "Pure Byte Baseline (V=256)", "ckpt": "out_baseline_pure_byte/ckpt.pt", "type": "baseline"},

    {"id": "std_full_pos_weighted", "name": "Standard Full POS Weighted (Char Stream)", "ckpt": "out_mc_korean_pos_weighted/ckpt.pt", "type": "std_pos"},
    {"id": "std_full_pos_unweighted", "name": "Standard Full POS Unweighted (Char Stream)", "ckpt": "out_mc_korean_pos_unweighted/ckpt.pt", "type": "std_pos"},
    {"id": "std_coarse_pos_weighted", "name": "Standard Coarse POS Weighted (Char Stream)", "ckpt": "out_opt1_sw005_10ep/ckpt.pt", "type": "std_pos"},

    {"id": "byte_fallback_full_pos_weighted", "name": "Byte Fallback Full POS Weighted", "ckpt": "out_mc_full_pos_weighted_byte/ckpt.pt", "type": "byte_pos"},
    {"id": "byte_fallback_full_pos_unweighted", "name": "Byte Fallback Full POS Unweighted", "ckpt": "out_mc_full_pos_unweighted_byte/ckpt.pt", "type": "byte_pos"},
    {"id": "byte_fallback_coarse_pos_weighted", "name": "Byte Fallback Coarse POS Weighted", "ckpt": "out_mc_coarse_pos_weighted_byte/ckpt.pt", "type": "byte_pos"},
    {"id": "byte_fallback_coarse_pos_unweighted", "name": "Byte Fallback Coarse POS Unweighted", "ckpt": "out_mc_coarse_pos_unweighted_byte/ckpt.pt", "type": "byte_pos"},
]


def run_benchmark_evaluations(max_examples: int = 100, output_file: str = "eval_results_comprehensive.json"):
    results = {}

    print("==========================================================================")
    print(" STARTING COMPREHENSIVE DOWNSTREAM EVALUATION SUITE")
    print(f" Targets: {len(EVAL_TARGETS)} models/checkpoints")
    print(f" Max Examples per benchmark: {max_examples}")
    print(f" Device: {DEVICE}")
    print("==========================================================================")

    for idx, target in enumerate(EVAL_TARGETS, 1):
        target_id = target["id"]
        target_name = target["name"]
        ckpt_path = target["ckpt"]

        if not os.path.exists(ckpt_path):
            print(f"[{idx}/{len(EVAL_TARGETS)}] Checkpoint not found: {ckpt_path}. Skipping.")
            continue

        print(f"\n[{idx}/{len(EVAL_TARGETS)}] Evaluating: {target_name} ({ckpt_path})")
        target_res = {
            "name": target_name,
            "ckpt": ckpt_path,
            "type": target["type"],
        }

        # 1. Four Capability Evaluations (KLUE-NER, KLUE-DP, NSMC 80%, UnSmile 80%, KorMedMCQA)
        print("  -> Running Four Capabilities...")
        temp_4cap = f"/tmp/eval_4cap_{target_id}.json"
        cmd_4cap = [
            sys.executable, "benchmarks/run_four_capability_evals.py",
            "--ckpts", f"{target_id}:{ckpt_path}",
            "--device", DEVICE,
            "--max_examples", str(max_examples),
            "--output_json", temp_4cap,
        ]
        try:
            subprocess.run(cmd_4cap, cwd=REPO_ROOT, check=True)
            if os.path.exists(temp_4cap):
                with open(temp_4cap, "r", encoding="utf-8") as f:
                    data = json.load(f)
                target_res["four_capabilities"] = data.get(target_id, {})
                os.remove(temp_4cap)
        except Exception as e:
            print(f"     Failed: {e}")

        # 2. Ko-HellaSwag Zero-Shot Reasoning
        print("  -> Running Ko-HellaSwag Zero-Shot Reasoning...")
        temp_hs = f"/tmp/eval_hs_{target_id}.json"
        cmd_hs = [
            sys.executable, "benchmarks/run_ko_hellaswag.py",
            "--ckpt_path", ckpt_path,
            "--device", DEVICE,
            "--eval_all_norms",
            "--max_examples", str(max_examples),
            "--output_json", temp_hs,
        ]
        try:
            subprocess.run(cmd_hs, cwd=REPO_ROOT, check=True)
            if os.path.exists(temp_hs):
                with open(temp_hs, "r", encoding="utf-8") as f:
                    data = json.load(f)
                target_res["ko_hellaswag"] = data.get("accuracies", {})
                os.remove(temp_hs)
        except Exception as e:
            print(f"     Failed: {e}")

        # 3. Phonetic Slang & Typo Resilience (Clean vs 80% Corrupted)
        print("  -> Running Phonetic Slang & Typo Resilience...")
        temp_slang = f"/tmp/eval_slang_{target_id}.json"
        is_mc = target["type"] in ["hybrid", "std_pos", "byte_pos"]
        cmd_slang = [
            sys.executable, "benchmarks/run_phonetic_slang_eval.py",
            "--device", DEVICE,
            "--max_examples", str(max_examples),
            "--output_json", temp_slang,
        ]
        if is_mc:
            cmd_slang.extend(["--mc_ckpt", ckpt_path])
        else:
            cmd_slang.extend(["--base_ckpt", ckpt_path])

        try:
            subprocess.run(cmd_slang, cwd=REPO_ROOT, check=True)
            if os.path.exists(temp_slang):
                with open(temp_slang, "r", encoding="utf-8") as f:
                    data = json.load(f)
                target_res["phonetic_slang"] = data
                os.remove(temp_slang)
        except Exception as e:
            print(f"     Failed: {e}")

        # 4. Vocabulary Tail Perplexity
        print("  -> Running Vocabulary Tail Perplexity...")
        temp_tail = f"/tmp/eval_tail_{target_id}.json"
        cmd_tail = [
            sys.executable, "benchmarks/run_vocab_tail_perplexity.py",
            "--device", DEVICE,
            "--max_examples", str(min(max_examples, 50)),
            "--sample_syllables_per_decile", "50",
            "--output_json", temp_tail,
        ]
        if is_mc:
            cmd_tail.extend(["--mc_ckpt", ckpt_path])
        else:
            cmd_tail.extend(["--base_ckpt", ckpt_path])

        try:
            subprocess.run(cmd_tail, cwd=REPO_ROOT, check=True)
            if os.path.exists(temp_tail):
                with open(temp_tail, "r", encoding="utf-8") as f:
                    data = json.load(f)
                target_res["vocab_tail"] = data
                os.remove(temp_tail)
        except Exception as e:
            print(f"     Failed: {e}")

        results[target_id] = target_res

        # Incremental save
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    # 5. OOV & Unicode Stress-Test across all 10-epoch models
    print("\n  -> Running OOV & Unicode Stress-Test...")
    temp_oov = f"/tmp/eval_oov_all.json"
    oov_models = [
        "nochar_coarse_pos_weighted",
        "nochar_coarse_pos_unweighted",
        "nochar_full_pos_weighted",
        "nochar_full_pos_unweighted",
        "baseline_character",
        "byte_fallback_bpe",
        "pure_byte",
        "std_full_pos_weighted",
        "std_coarse_pos_weighted",
        "byte_fallback_full_pos_weighted",
        "byte_fallback_coarse_pos_weighted",
    ]
    cmd_oov = [
        sys.executable, "benchmarks/run_oov_evaluations.py",
        "--models", *oov_models,
        "--device", DEVICE,
        "--max_new_tokens", "25",
        "--output_json", temp_oov,
    ]
    try:
        subprocess.run(cmd_oov, cwd=REPO_ROOT, check=True)
        if os.path.exists(temp_oov):
            with open(temp_oov, "r", encoding="utf-8") as f:
                oov_data = json.load(f)
            results["oov_evaluations"] = oov_data.get("model_evaluations", {})
            os.remove(temp_oov)
    except Exception as e:
        print(f"     Failed OOV evaluation: {e}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n==========================================================================")
    print(f" ALL EVALUATIONS COMPLETED! Results saved to {output_file}")
    print(f"==========================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete downstream evaluation suite.")
    parser.add_argument("--max_examples", type=int, default=100, help="Max examples per benchmark")
    parser.add_argument("--output_file", type=str, default="eval_results_comprehensive.json", help="Output JSON path")
    args = parser.parse_args()

    run_benchmark_evaluations(max_examples=args.max_examples, output_file=args.output_file)
