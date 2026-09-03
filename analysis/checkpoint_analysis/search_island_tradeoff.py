import argparse
import json
import os
import shutil
import subprocess
from typing import Dict, List, Tuple

import torch
import yaml


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Greedy per-tensor threshold search: start each tensor near threshold 0.5, "
            "then step downward (default 0.05) one tensor at a time, accepting the "
            "lowest val-loss increase candidate each round while under loss tolerance."
        )
    )
    p.add_argument("ckpt_dir", type=str, help="Checkpoint directory containing ckpt.pt")
    p.add_argument(
        "--island_json",
        type=str,
        default=None,
        help="Path to islands_detailed.json (default: <ckpt_dir>/island_analysis/islands_detailed.json)",
    )
    p.add_argument("--eval_dataset", type=str, default=None, help="Dataset name for validation loss")
    p.add_argument("--eval_iters", type=int, default=50, help="Evaluation iterations")
    p.add_argument("--device", type=str, default="cpu", help="Device for eval subprocess")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--loss_tolerance_pct", type=float, default=2.0, help="Max allowed val-loss increase (%)")
    p.add_argument("--start_threshold", type=float, default=0.6, help="Start threshold per tensor")
    p.add_argument("--threshold_step", type=float, default=0.05, help="Threshold decrement per step")
    p.add_argument(
        "--pattern",
        type=str,
        default="",
        help="Only mutate tensors containing this substring (simple filter)",
    )
    p.add_argument(
        "--provider_mode",
        choices=["top", "mean"],
        default="top",
        help="Representative to use for row replacement inside an island",
    )
    p.add_argument("--max_rounds", type=int, default=999, help="Max greedy rounds")
    p.add_argument(
        "--export_selected_ckpt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to export selected/ckpt.pt (default: true)",
    )
    p.add_argument(
        "--delete_unselected_ckpts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Delete candidates/*/ckpt.pt that are not selected (default: true). "
            "Useful to reduce disk usage during large searches."
        ),
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory (default: <ckpt_dir>/island_tradeoff_search)",
    )
    return p.parse_args()


def load_ckpt(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    state = ckpt.get("model", ckpt)
    return ckpt, state


def island_rep(weight: torch.Tensor, island: Dict, provider_mode: str) -> torch.Tensor:
    members = island["member_indices"]
    if provider_mode == "mean":
        return weight[members].mean(dim=0)
    providers = island.get("providers", [])
    if providers:
        return weight[int(providers[0]["vector_index"])]
    return weight[members].mean(dim=0)


def eval_metrics_to_latency(metrics: Dict, block_size: int) -> Dict:
    elapsed = float(metrics.get("elapsed_time_s", 0.0) or 0.0)
    num_batches = int(metrics.get("num_batches", 0) or 0)
    iter_ms = (elapsed / max(num_batches, 1)) * 1000.0
    tok_ms = (elapsed / max(num_batches * max(block_size, 1), 1)) * 1000.0
    tok_s = 1000.0 / max(tok_ms, 1e-9)
    return {
        "iter_latency_ms": iter_ms,
        "decode_token_latency_ms": tok_ms,
        "decode_tokens_per_s": tok_s,
    }


def run_eval(candidate_dir: str, eval_dataset: str, eval_iters: int, device: str, dtype: str, block_size: int) -> Dict:
    eval_out = os.path.join(candidate_dir, "eval")
    os.makedirs(eval_out, exist_ok=True)
    cmd = [
        "python3", "sample.py",
        "--init_from", "resume",
        "--out_dir", candidate_dir,
        "--eval_only",
        "--eval_dataset", eval_dataset,
        "--eval_iters", str(eval_iters),
        "--device", device,
        "--dtype", dtype,
        "--no-print_model_info",
        "--eval_output_dir", eval_out,
    ]
    subprocess.run(cmd, check=True)
    with open(os.path.join(eval_out, "eval_loss.txt"), "r", encoding="utf-8") as f:
        metrics = json.load(f)
    metrics.update(eval_metrics_to_latency(metrics, block_size))
    return metrics


def canonical_tensor_candidates(report: Dict) -> List[Dict]:
    entries: List[Dict] = []
    for e in report.get("threshold_tries", []):
        thr = float(e.get("threshold", 0.0))
        entries.append({
            "threshold": thr,
            "num_islands": int(e.get("summary", {}).get("num_islands", 0)),
            "islands": e.get("islands", []),
        })
    if not entries:
        return []

    dedup = {}
    for e in entries:
        dedup[e["threshold"]] = e
    return [dedup[t] for t in sorted(dedup.keys(), reverse=True)]


def initial_candidate_index(cands: List[Dict], start_threshold: float) -> int:
    best_i, best_dist = 0, 1e9
    for i, c in enumerate(cands):
        dist = abs(float(c["threshold"]) - start_threshold)
        if dist < best_dist:
            best_i, best_dist = i, dist
    return best_i


def next_lower_threshold_index(cands: List[Dict], idx: int, step: float) -> int:
    current = float(cands[idx]["threshold"])
    target = current - step
    for j in range(idx + 1, len(cands)):
        if float(cands[j]["threshold"]) <= target + 1e-12:
            return j
    return -1


def build_mutated_state(base_state: Dict, tensor_candidates: Dict[str, List[Dict]], chosen_idx: Dict[str, int], provider_mode: str) -> Tuple[Dict, Dict]:
    state = {k: v.clone() if torch.is_tensor(v) else v for k, v in base_state.items()}
    meta = {"tensors": {}, "total_rows": 0, "total_islands": 0}

    for tname, idx in chosen_idx.items():
        if tname not in state:
            continue
        w = state[tname]
        if not torch.is_tensor(w) or not torch.is_floating_point(w) or w.ndim != 2:
            continue
        cands = tensor_candidates.get(tname, [])
        if not cands or idx < 0 or idx >= len(cands):
            continue

        cand = cands[idx]
        islands = cand.get("islands", [])
        new_w = w.clone()
        for island in islands:
            rep = island_rep(w, island, provider_mode)
            for ridx in island.get("member_indices", []):
                ridx = int(ridx)
                if 0 <= ridx < new_w.shape[0]:
                    new_w[ridx] = rep
        state[tname] = new_w

        meta["tensors"][tname] = {
            "threshold": float(cand["threshold"]),
            "num_islands": int(cand["num_islands"]),
            "covered_rows": int(sum(len(i.get("member_indices", [])) for i in islands)),
            "total_rows": int(w.shape[0]),
        }
        meta["total_rows"] += int(w.shape[0])
        meta["total_islands"] += int(cand["num_islands"])

    meta["decode_reduction_proxy"] = 0.0
    if meta["total_rows"] > 0:
        meta["decode_reduction_proxy"] = 1.0 - (meta["total_islands"] / meta["total_rows"])
    return state, meta


def to_selected_config(tensor_candidates: Dict[str, List[Dict]], chosen_idx: Dict[str, int]) -> Dict[str, Dict]:
    out = {}
    for tname, idx in chosen_idx.items():
        cands = tensor_candidates.get(tname, [])
        if not cands or idx < 0 or idx >= len(cands):
            continue
        c = cands[idx]
        out[tname] = {"idx": idx, "threshold": float(c["threshold"]), "num_islands": int(c["num_islands"])}
    return out


def write_yaml(path: str, payload: Dict):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def remove_file_if_exists(path: str):
    if os.path.exists(path):
        os.remove(path)


def main():
    args = parse_args()
    ckpt_path = os.path.join(args.ckpt_dir, "ckpt.pt")
    island_json = args.island_json or os.path.join(args.ckpt_dir, "island_analysis", "islands_detailed.json")
    out_dir = args.out_dir or os.path.join(args.ckpt_dir, "island_tradeoff_search")
    os.makedirs(out_dir, exist_ok=True)

    base_ckpt, base_state = load_ckpt(ckpt_path)
    block_size = int(base_ckpt.get("config", {}).get("block_size", 1))
    dataset = args.eval_dataset or base_ckpt.get("config", {}).get("dataset")
    if not dataset:
        raise ValueError("--eval_dataset is required when checkpoint config has no dataset")

    with open(island_json, "r", encoding="utf-8") as f:
        island_payload = json.load(f)

    tensor_candidates: Dict[str, List[Dict]] = {}
    for report in island_payload.get("reports", []):
        tname = report.get("tensor")
        if not tname:
            continue
        if args.pattern and args.pattern not in tname:
            continue
        cands = canonical_tensor_candidates(report)
        if cands:
            tensor_candidates[tname] = cands

    if not tensor_candidates:
        raise ValueError("No tensor candidates found. Check --pattern and island analysis JSON.")

    baseline_dir = os.path.join(out_dir, "baseline")
    os.makedirs(baseline_dir, exist_ok=True)
    shutil.copy2(ckpt_path, os.path.join(baseline_dir, "ckpt.pt"))
    meta_path = os.path.join(args.ckpt_dir, "meta.pkl")
    if os.path.exists(meta_path):
        shutil.copy2(meta_path, os.path.join(baseline_dir, "meta.pkl"))

    base_eval = run_eval(baseline_dir, dataset, args.eval_iters, args.device, args.dtype, block_size)
    base_loss = float(base_eval["val"])
    max_allowed = base_loss * (1 + args.loss_tolerance_pct / 100.0)

    chosen_idx = {t: initial_candidate_index(c, args.start_threshold) for t, c in tensor_candidates.items()}

    rounds = []
    current_loss = base_loss
    accepted_rounds = 0

    for rnd in range(1, args.max_rounds + 1):
        tested = []
        best = None

        for tname, idx in chosen_idx.items():
            cands = tensor_candidates[tname]
            nidx = next_lower_threshold_index(cands, idx, args.threshold_step)
            if nidx < 0:
                continue

            trial_idx = dict(chosen_idx)
            trial_idx[tname] = nidx
            trial_state, trial_meta = build_mutated_state(base_state, tensor_candidates, trial_idx, args.provider_mode)

            cand_tag = f"round_{rnd}_{tname.replace('.', '_')}"
            cand_dir = os.path.join(out_dir, "candidates", cand_tag)
            os.makedirs(cand_dir, exist_ok=True)

            cand_ckpt = {k: v for k, v in base_ckpt.items()}
            cand_ckpt["model"] = trial_state
            torch.save(cand_ckpt, os.path.join(cand_dir, "ckpt.pt"))
            if os.path.exists(meta_path):
                shutil.copy2(meta_path, os.path.join(cand_dir, "meta.pkl"))

            em = run_eval(cand_dir, dataset, args.eval_iters, args.device, args.dtype, block_size)
            vloss = float(em["val"])
            rec = {
                "tensor": tname,
                "from_idx": idx,
                "to_idx": nidx,
                "from_threshold": float(cands[idx]["threshold"]),
                "to_threshold": float(cands[nidx]["threshold"]),
                "from_num_islands": int(cands[idx]["num_islands"]),
                "to_num_islands": int(cands[nidx]["num_islands"]),
                "val_loss": vloss,
                "loss_delta_pct": (vloss / base_loss - 1.0) * 100.0,
                "decode_reduction_proxy": trial_meta["decode_reduction_proxy"],
                "decode_token_latency_ms": float(em.get("decode_token_latency_ms", 0.0)),
                "decode_tokens_per_s": float(em.get("decode_tokens_per_s", 0.0)),
                "iter_latency_ms": float(em.get("iter_latency_ms", 0.0)),
                "feasible": vloss <= max_allowed,
                "candidate_dir": cand_dir,
            }
            tested.append(rec)
            if best is None or rec["loss_delta_pct"] < best["loss_delta_pct"]:
                best = rec

        if args.delete_unselected_ckpts:
            best_dir = best["candidate_dir"] if best is not None else None
            for rec in tested:
                cand_ckpt_path = os.path.join(rec["candidate_dir"], "ckpt.pt")
                if rec["candidate_dir"] != best_dir:
                    remove_file_if_exists(cand_ckpt_path)

        round_log = {
            "round": rnd,
            "tested": tested,
            "best_candidate": best,
            "selected": None,
            "stop_reason": None,
            "selected_config": to_selected_config(tensor_candidates, chosen_idx),
            "pre_round_val_loss": current_loss,
            "post_round_val_loss": current_loss,
            "current_decode_token_latency_ms": None,
            "current_decode_tokens_per_s": None,
        }

        if not tested:
            round_log["stop_reason"] = "no_more_threshold_steps"
            round_log["post_round_val_loss"] = current_loss
            rounds.append(round_log)
            break

        if best is None or best["val_loss"] > max_allowed:
            round_log["stop_reason"] = "best_above_loss_threshold"
            round_log["post_round_val_loss"] = current_loss
            rounds.append(round_log)
            break

        chosen_idx[best["tensor"]] = int(best["to_idx"])
        current_loss = float(best["val_loss"])
        accepted_rounds += 1
        round_log["selected"] = best
        round_log["selected_config"] = to_selected_config(tensor_candidates, chosen_idx)
        round_log["post_round_val_loss"] = current_loss
        round_log["current_decode_token_latency_ms"] = float(best["decode_token_latency_ms"])
        round_log["current_decode_tokens_per_s"] = float(best["decode_tokens_per_s"])
        rounds.append(round_log)

    final_state, final_meta = build_mutated_state(base_state, tensor_candidates, chosen_idx, args.provider_mode)

    selected_loss = current_loss
    selected_eval = None
    selected_dir = os.path.join(out_dir, "selected")
    if args.export_selected_ckpt:
        selected_ckpt = {k: v for k, v in base_ckpt.items()}
        selected_ckpt["model"] = final_state
        os.makedirs(selected_dir, exist_ok=True)
        torch.save(selected_ckpt, os.path.join(selected_dir, "ckpt.pt"))
        if os.path.exists(meta_path):
            shutil.copy2(meta_path, os.path.join(selected_dir, "meta.pkl"))
        selected_eval = run_eval(selected_dir, dataset, args.eval_iters, args.device, args.dtype, block_size)
        selected_loss = float(selected_eval["val"])

    baseline_speed = {
        "iter_latency_ms": float(base_eval.get("iter_latency_ms", 0.0)),
        "decode_token_latency_ms": float(base_eval.get("decode_token_latency_ms", 0.0)),
        "decode_tokens_per_s": float(base_eval.get("decode_tokens_per_s", 0.0)),
    }
    selected_speed = baseline_speed.copy()
    if selected_eval is not None:
        selected_speed = {
            "iter_latency_ms": float(selected_eval.get("iter_latency_ms", 0.0)),
            "decode_token_latency_ms": float(selected_eval.get("decode_token_latency_ms", 0.0)),
            "decode_tokens_per_s": float(selected_eval.get("decode_tokens_per_s", 0.0)),
        }

    speed_comparison = {
        "baseline": baseline_speed,
        "selected": selected_speed,
        "iter_speedup": baseline_speed["iter_latency_ms"] / max(selected_speed["iter_latency_ms"], 1e-9),
        "decode_token_latency_speedup": baseline_speed["decode_token_latency_ms"] / max(selected_speed["decode_token_latency_ms"], 1e-9),
        "decode_tokens_per_s_speedup": selected_speed["decode_tokens_per_s"] / max(baseline_speed["decode_tokens_per_s"], 1e-9),
    }

    log_payload = {
        "dataset": dataset,
        "baseline": {
            "val_loss": base_loss,
            "max_allowed_val_loss": max_allowed,
            "loss_tolerance_pct": args.loss_tolerance_pct,
            **baseline_speed,
        },
        "search": {
            "strategy": "greedy_per_tensor_threshold_stepdown",
            "start_threshold": args.start_threshold,
            "threshold_step": args.threshold_step,
            "provider_mode": args.provider_mode,
            "pattern": args.pattern,
            "accepted_rounds": accepted_rounds,
            "max_rounds": args.max_rounds,
            "export_selected_ckpt": args.export_selected_ckpt,
            "delete_unselected_ckpts": args.delete_unselected_ckpts,
        },
        "initial_config": to_selected_config(
            tensor_candidates,
            {t: initial_candidate_index(c, args.start_threshold) for t, c in tensor_candidates.items()},
        ),
        "rounds": rounds,
        "selected": {
            "config": to_selected_config(tensor_candidates, chosen_idx),
            "val_loss": selected_loss,
            "loss_delta_pct": (selected_loss / base_loss - 1.0) * 100.0,
            "decode_reduction_proxy": final_meta["decode_reduction_proxy"],
            "tensor_meta": final_meta["tensors"],
            **selected_speed,
            "exported_ckpt_dir": selected_dir if args.export_selected_ckpt else None,
        },
        "speed_comparison": speed_comparison,
    }

    yaml_path = os.path.join(out_dir, "search_log.yaml")
    write_yaml(yaml_path, log_payload)
    with open(os.path.join(out_dir, "search_results.json"), "w", encoding="utf-8") as f:
        json.dump(log_payload, f, indent=2)

    if args.delete_unselected_ckpts:
        selected_ckpt_real = os.path.realpath(os.path.join(selected_dir, "ckpt.pt")) if args.export_selected_ckpt else None
        for rec in (r for rnd in rounds for r in rnd.get("tested", [])):
            cand_ckpt_path = os.path.join(rec["candidate_dir"], "ckpt.pt")
            if selected_ckpt_real is not None and os.path.exists(cand_ckpt_path):
                if os.path.realpath(cand_ckpt_path) == selected_ckpt_real:
                    continue
            remove_file_if_exists(cand_ckpt_path)

    print(f"Baseline val loss: {base_loss:.6f}")
    print(f"Max allowed val loss: {max_allowed:.6f}")
    print(f"Selected val loss: {selected_loss:.6f}")
    print(f"Accepted rounds: {accepted_rounds}")
    print(f"Decode speedup (tok/s): {speed_comparison['decode_tokens_per_s_speedup']:.4f}x")
    print(f"Wrote {yaml_path}")


if __name__ == "__main__":
    main()
