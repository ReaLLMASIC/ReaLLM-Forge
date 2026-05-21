"""
Greedy iterative hyper-parameter search for **train.py**

Changes vs. the original version
--------------------------------
* `--random_iterations N` — run each candidate N times with **different seeds**.
  - The seed passed to `train.py` is `baseline_seed + run_id`, where
    `baseline_seed` is whatever is in the current baseline config
    (or the default 1337 if absent).
  - The candidate block in *sweep_log.yaml* includes a `seeds:` list
    with per-seed results.
  - Efficiency is computed on the **average objective** across those N runs.
* Supports efficiency normalization by params, torch-allocated VRAM,
  torch-reserved VRAM, process GPU usage, or iteration latency.
* Supports optimization on score, RankMe, or AReQ.
"""

import argparse
import ast
import gc
import math
import os
import re
import subprocess
import sys
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple
import random

import torch
import yaml


TrialMetrics = Tuple[float, float, int, float, float, float, float, float, float]


# ───────────────────────── helpers ──────────────────────────
def dict_to_cli(d: Dict[str, Any]) -> List[str]:
    """
    Convert a config dict to a flat list of CLI args for *train.py*.

    Any key that starts with “_” is considered **private** and is *not*
    forwarded, because *train.py* would reject unknown flags such as
    “--_last_dup_idx”.
    """
    cli: List[str] = []
    for k, v in d.items():
        if str(k).startswith("_"):
            continue

        if isinstance(v, bool):
            if v:
                cli.append(f"--{k}")
        elif isinstance(v, list):
            cli.append(f"--{k}")
            cli.extend(map(str, v))
        else:
            cli.extend([f"--{k}", str(v)])
    return cli


@contextmanager
def patched_argv(argv: List[str]):
    old = sys.argv
    sys.argv = argv
    try:
        yield
    finally:
        sys.argv = old


def _cleanup_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _nanmean(values: List[float]) -> float:
    valid = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    return float(sum(valid) / len(valid)) if valid else float("nan")


def run_trial_inproc(cfg: Dict[str, Any]) -> TrialMetrics:
    """
    Return:
        (
            best_val_loss,
            num_params,
            best_iter,
            torch_alloc_mb,
            torch_resv_mb,
            process_gpu_mb,
            iter_latency_ms,
            rankme,
            areq,
        )
    """
    from train import Trainer
    from train_args import parse_args as parse_train_args

    cli = ["train.py"] + dict_to_cli(cfg)
    with patched_argv(cli):
        args, mg, tg, lg = parse_train_args()
    tr = Trainer(args, mg, tg, lg)
    tr.train()

    loss = float(tr.best_val_loss)
    nparam = float(tr.raw_model.num_param)
    best_iter = int(getattr(tr, "best_iter", 0))
    torch_alloc_mb = float(
        getattr(tr, "peak_torch_allocated", getattr(tr, "peak_gpu_usage", 0.0))
        / (1024 ** 2)
    )
    torch_resv_mb = float(getattr(tr, "peak_torch_reserved", 0.0) / (1024 ** 2))
    process_gpu_mb = float(
        getattr(tr, "peak_process_gpu_usage", 0.0) / (1024 ** 2)
    )
    iter_latency_ms = float(getattr(tr, "iter_latency_avg", 0.0))
    rankme = float(getattr(tr, "latest_rankme", float("nan")))
    areq = float(getattr(tr, "latest_areq", float("nan")))

    del tr
    _cleanup_cuda()
    return (
        loss,
        nparam,
        best_iter,
        torch_alloc_mb,
        torch_resv_mb,
        process_gpu_mb,
        iter_latency_ms,
        rankme,
        areq,
    )


def _parse_best_metrics_file(metrics_path: Path) -> TrialMetrics:
    line = [x.strip() for x in metrics_path.read_text().strip().split(",")]
    if len(line) < 21:
        raise ValueError(
            f"Unexpected metric layout in {metrics_path}: expected >=21 columns, got {len(line)}"
        )

    loss = float(line[0])
    best_iter = int(line[1])
    nparam = float(line[3])
    # Layout
    torch_alloc_mb = float(line[6])
    torch_resv_mb = float(line[7])
    process_gpu_mb = float(line[8])
    iter_latency_ms = float(line[9])

    # Optional Zeus column (zeus_best_train_step_energy_j) may be inserted at index 10.
    rankme_idx = 20 if len(line) > 21 else 19
    areq_idx = rankme_idx + 1
    rankme = float(line[rankme_idx])
    areq = float(line[areq_idx])

    return (
        loss,
        nparam,
        best_iter,
        torch_alloc_mb,
        torch_resv_mb,
        process_gpu_mb,
        iter_latency_ms,
        rankme,
        areq,
    )


def run_trial_subproc(cfg: Dict[str, Any]) -> TrialMetrics:
    script_dir = Path(__file__).parent
    cmd = [sys.executable, str(script_dir / "train.py")] + dict_to_cli(cfg)
    env = {k: v for k, v in os.environ.items() if k not in {"RANK", "WORLD_SIZE"}}
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if p.returncode:
        print(p.stderr)
        raise RuntimeError("train.py failed")

    out_dir = Path(cfg.get("out_dir", "out"))
    metrics = _parse_best_metrics_file(out_dir / "best_val_loss_and_iter.txt")
    _cleanup_cuda()
    return metrics


def load_log(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text())
    return data or {}


def save_log(path: Path, log: Dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(yaml.dump(log, sort_keys=False))
    tmp.replace(path)


# ───────────────────────── search controller ─────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Greedy hyper-param search wrapper")
    ap.add_argument("--orig_settings", required=True)
    ap.add_argument("--param_names", nargs="+", required=True)
    ap.add_argument("--increments", nargs="+", type=float, required=True)
    ap.add_argument(
        "--iterations",
        type=int,
        required=True,
        help="multiples of increment to try per parameter",
    )
    ap.add_argument(
        "--num_iterations", type=int, default=1, help="max outer search iterations"
    )
    ap.add_argument(
        "--random_iterations",
        type=int,
        default=1,
        help="how many different random seeds per candidate",
    )
    ap.add_argument("--results_file", default="sweep_log.yaml")
    ap.add_argument("--spawn_subprocess", action="store_true")
    ap.add_argument(
        "--override_cfg",
        nargs="*",
        metavar="KEY=VALUE",
        default=[],
        help=(
            "Override baseline config settings from orig_settings before starting the "
            "search. Example: --override_cfg max_iters=10000 learning_rate=0.0005 "
            "flag=True name='my_exp' path=data/run"
        ),
    )
    ap.add_argument(
        "--max_iters_increase",
        type=int,
        default=None,
        help=(
            "If set, and no positive-efficiency candidate is found, increase "
            "'max_iters' by this amount."
        ),    )
    ap.add_argument(
        "--nlayer_dup_mode",
        choices=["dup_middle", "dup_each"],
        default="dup_middle",
        help=(
            "Strategy when testing +1 to n_layer:\n"
            "  dup_middle (default) – duplicate the rounded-up middle layer\n"
            "  dup_each             – create one candidate per layer by duplicating it"
        ),
    )
    ap.add_argument(
        "--efficiency_target",
        choices=["params", "vram", "iter", "torch_allocated", "torch_reserved", "process_gpu"],
        default="params",
        help=(
            "Metric to normalize score gain: 'params' (default) for parameter count, "
            "'vram' (legacy alias for torch_allocated), 'torch_allocated', "
            "'torch_reserved', 'process_gpu', or 'iter' for average iteration latency in ms."
        ),
    )
    ap.add_argument(
        "--optimize_target",
        choices=["score", "rankme", "areq"],
        default="score",
        help="Optimization objective: score (1/exp(loss)), rankme, or areq.",
    )
    ap.add_argument(
        "--optimize_mode",
        choices=["max", "min"],
        default="max",
        help="Whether to maximize or minimize the selected optimization target.",
    )
    ap.add_argument(
        "--randomize_seed",
        action="store_true",
        help="Whether to random seed for each separate train.py run, and prevent overfitting via hillclimbing on one section of the target dataset",
    )

    args = ap.parse_args()

    if len(args.increments) == 1:
        args.increments *= len(args.param_names)
    if len(args.increments) != len(args.param_names):
        sys.exit("--increments length mismatch")

    inc_map = dict(zip(args.param_names, args.increments))
    run_fn = run_trial_subproc if args.spawn_subprocess else run_trial_inproc

    baseline_cfg_master = yaml.safe_load(Path(args.orig_settings).read_text())
    log_path = Path(args.results_file)
    log = load_log(log_path)

    def _apply_overrides_to_active_config(
        config_dict: Dict[str, Any], overrides: List[str], context_msg: str
    ) -> None:
        if not overrides:
            return

        print(f"[CONFIG_OVERRIDE] Checking {len(overrides)} overrides for {context_msg}...")
        effective_overrides = 0
        for item in overrides:
            try:
                key, value_str = item.split("=", 1)
            except ValueError:
                sys.exit(f"Error: Invalid override format '{item}'. Expected KEY=VALUE.")

            try:
                value = ast.literal_eval(value_str)
            except (ValueError, SyntaxError):
                value = value_str

            original_value = config_dict.get(key)
            if key not in config_dict or original_value != value:
                old_desc = repr(original_value) if key in config_dict else "N/A (new key)"
                print(
                    f"  Applying to active config: {key} = {repr(value)} "
                    f"(was: {old_desc}, type: {type(value).__name__})"
                )
                config_dict[key] = value
                effective_overrides += 1
            else:
                print(f"  Skipping (no change): {key} = {repr(value)}")
        if effective_overrides > 0:
            print(
                f"[CONFIG_OVERRIDE] Applied {effective_overrides} effective overrides "
                f"to {context_msg}."
            )

    log.setdefault("baseline_config", deepcopy(baseline_cfg_master))
    log.setdefault("iterations", [])

    def _extend_layerlists(cfg: Dict[str, Any], dup_idx: int) -> None:
        """
        Duplicate element *dup_idx* (0-based) in every X_layerlist present in *cfg*.
        Modifies the dict in place.
        """
        for key, val in cfg.items():
            if key.endswith("_layerlist") and isinstance(val, list) and val:
                src = min(dup_idx, len(val) - 1)
                val.insert(src + 1, deepcopy(val[src]))

    if log["iterations"]:
        last = log["iterations"][-1]
        baseline_cfg = deepcopy(last["baseline_config_after"])
        base_loss = last["baseline_metrics"]["loss"]
        base_score = last["baseline_metrics"]["score"]
        base_rankme = last["baseline_metrics"].get("rankme", float("nan"))
        base_areq = last["baseline_metrics"].get("areq", float("nan"))
        base_params = last["baseline_metrics"]["params"]
        base_torch_alloc = last["baseline_metrics"].get(
            "peak_torch_allocated_mb",
            last["baseline_metrics"].get("peak_gpu_mb", 0.0),
        )
        base_torch_reserved = last["baseline_metrics"].get("peak_torch_reserved_mb", 0.0)
        base_process_gpu = last["baseline_metrics"].get("peak_process_gpu_mb", 0.0)
        base_iter_ms = last["baseline_metrics"].get("iter_latency_avg", 0.0)
        cur_iter = last["iter"] + 1
        _apply_overrides_to_active_config(
            baseline_cfg, args.override_cfg, "resumed baseline_cfg"
        )
    else:
        baseline_cfg = deepcopy(log["baseline_config"])
        _apply_overrides_to_active_config(
            baseline_cfg, args.override_cfg, "initial baseline_cfg for new sweep"
        )

        print("[BASELINE] measuring initial config …")
        (
            base_loss,
            base_params,
            base_best_iter,
            base_torch_alloc,
            base_torch_reserved,
            base_process_gpu,
            base_iter_ms,
            base_rankme,
            base_areq,
        ) = run_fn(deepcopy(baseline_cfg))
        base_score = 1 / math.exp(base_loss)
        log["iterations"].append(
            {
                "iter": -1,
                "baseline_metrics": {
                    "loss": base_loss,
                    "score": base_score,
                    "params": base_params,
                    "peak_torch_allocated_mb": base_torch_alloc,
                    "peak_torch_reserved_mb": base_torch_reserved,
                    "peak_process_gpu_mb": base_process_gpu,
                    "iter_latency_avg": base_iter_ms,
                    "best_iter": base_best_iter,
                    "rankme": base_rankme,
                    "areq": base_areq,
                },
                "baseline_config_after": deepcopy(baseline_cfg),
            }
        )
        cur_iter = 0

    log["baseline_config"] = deepcopy(baseline_cfg)
    save_log(log_path, log)

    while cur_iter < args.num_iterations:
        print(f"========== Iteration {cur_iter} ==========")
        candidates: List[Dict[str, Any]] = []
        best_choice: Tuple[float, Dict[str, Any]] | None = None

        for pname in args.param_names:
            if pname not in baseline_cfg:
                print(f"[WARN] parameter '{pname}' not in baseline config – skipping")
                continue

            base_val = baseline_cfg[pname]
            step_spec = inc_map[pname]

            def _numeric_add(x: Any, delta: float) -> Any:
                return int(round(x + delta)) if isinstance(x, int) else float(x + delta)

            def _evaluate(
                cfg_template: Dict[str, Any], label_for_log: str, value_for_log: Any
            ) -> None:
                nonlocal best_choice, candidates

                seed0 = int(cfg_template.get("seed", 1337))
                if args.randomize_seed:
                    seed0 = random.randint(0, 2**31 - 1)
                seed_runs: List[Dict[str, Any]] = []
                scores: List[float] = []

                for r in range(args.random_iterations):
                    cfg_run = deepcopy(cfg_template)
                    cfg_run["seed"] = seed0 + r

                    print(f"[TEST] {label_for_log}={value_for_log}  seed={cfg_run['seed']}")
                    try:
                        (
                            loss,
                            nparam,
                            best_it,
                            torch_alloc_mb,
                            torch_resv_mb,
                            process_gpu_mb,
                            iter_ms,
                            rankme,
                            areq,
                        ) = run_fn(cfg_run)
                    except Exception as exc:
                        print("   ⚠", exc)
                        return

                    score = 1.0 / math.exp(loss)
                    seed_runs.append(
                        {
                            "seed": cfg_run["seed"],
                            "loss": loss,
                            "score": score,
                            "best_iter": best_it,
                            "peak_torch_allocated_mb": torch_alloc_mb,
                            "peak_torch_reserved_mb": torch_resv_mb,
                            "peak_process_gpu_mb": process_gpu_mb,
                            "iter_latency_ms": iter_ms,
                            "rankme": rankme,
                            "areq": areq,
                        }
                    )
                    scores.append(score)

                avg_score = sum(scores) / len(scores)
                avg_torch_alloc = (
                    sum(s["peak_torch_allocated_mb"] for s in seed_runs) / len(seed_runs)
                )
                avg_torch_reserved = (
                    sum(s["peak_torch_reserved_mb"] for s in seed_runs) / len(seed_runs)
                )
                avg_process_gpu = (
                    sum(s["peak_process_gpu_mb"] for s in seed_runs) / len(seed_runs)
                )
                avg_iter = sum(s["iter_latency_ms"] for s in seed_runs) / len(seed_runs)
                avg_rankme = _nanmean([s["rankme"] for s in seed_runs])
                avg_areq = _nanmean([s["areq"] for s in seed_runs])
                avg_loss = -math.log(avg_score)

                d_score = avg_score - base_score
                d_rankme = (
                    avg_rankme - base_rankme
                    if not math.isnan(avg_rankme) and not math.isnan(base_rankme)
                    else float("nan")
                )
                d_areq = (
                    avg_areq - base_areq
                    if not math.isnan(avg_areq) and not math.isnan(base_areq)
                    else float("nan")
                )
                d_param = nparam - base_params
                d_torch_alloc = avg_torch_alloc - base_torch_alloc
                d_torch_reserved = avg_torch_reserved - base_torch_reserved
                d_process_gpu = avg_process_gpu - base_process_gpu
                d_iter = avg_iter - base_iter_ms

                if args.efficiency_target == "params":
                    d_cost = d_param
                elif args.efficiency_target in ("vram", "torch_allocated"):
                    d_cost = d_torch_alloc
                elif args.efficiency_target == "torch_reserved":
                    d_cost = d_torch_reserved
                elif args.efficiency_target == "process_gpu":
                    d_cost = d_process_gpu
                elif args.efficiency_target == "iter":
                    d_cost = d_iter
                else:
                    raise ValueError("Unknown efficiency target")

                if args.optimize_target == "score":
                    objective_value = avg_score
                    baseline_objective = base_score
                elif args.optimize_target == "rankme":
                    objective_value = avg_rankme
                    baseline_objective = base_rankme
                elif args.optimize_target == "areq":
                    objective_value = avg_areq
                    baseline_objective = base_areq
                else:
                    raise ValueError("Unknown optimize target")

                objective_delta = objective_value - baseline_objective
                direction = 1.0 if args.optimize_mode == "max" else -1.0
                objective_improvement = direction * objective_delta

                if math.isnan(objective_improvement):
                    return

                eff = (
                    (objective_improvement / d_cost)
                    if d_cost != 0
                    else (math.inf if objective_improvement > 0 else 0.0)
                )

                cand = {
                    "param": label_for_log,
                    "value": value_for_log,
                    "avg_loss": avg_loss,
                    "avg_score": avg_score,
                    "avg_rankme": avg_rankme,
                    "avg_areq": avg_areq,
                    "best_val_loss": avg_loss,
                    "best_iter": max(s["best_iter"] for s in seed_runs),
                    "num_params": nparam,
                    "peak_torch_allocated_mb": avg_torch_alloc,
                    "peak_torch_reserved_mb": avg_torch_reserved,
                    "peak_process_gpu_mb": avg_process_gpu,
                    "iter_latency_avg": avg_iter,
                    "delta_score": d_score,
                    "delta_rankme": d_rankme,
                    "delta_areq": d_areq,
                    "delta_params": d_param,
                    "delta_torch_allocated_mb": d_torch_alloc,
                    "delta_torch_reserved_mb": d_torch_reserved,
                    "delta_process_gpu_mb": d_process_gpu,
                    "delta_iter_latency": d_iter,
                    "efficiency": eff,
                    "target_metric": args.optimize_target,
                    "target_mode": args.optimize_mode,
                    "target_value": objective_value,
                    "target_delta": objective_delta,
                    "target_improvement": objective_improvement,
                    "seeds": seed_runs,
                }
                candidates.append(cand)

                if eff > 0:
                    if best_choice is None:
                        best_choice = (eff, cand)
                    else:
                        old_eff, old_cand = best_choice
                        if (eff > old_eff) or (
                            math.isinf(eff)
                            and eff == old_eff
                            and cand["target_improvement"] > old_cand["target_improvement"]
                        ):
                            best_choice = (eff, cand)

            if pname == "n_layer":
                old_nlayer = int(baseline_cfg["n_layer"])
                new_nlayer = old_nlayer + 1

                def _nlayer_candidate(dup_idx: int, tag: str) -> None:
                    cfg2 = deepcopy(baseline_cfg)
                    cfg2["n_layer"] = new_nlayer
                    _extend_layerlists(cfg2, dup_idx)
                    cfg2["_last_dup_idx"] = dup_idx
                    _evaluate(
                        cfg2,
                        "n_layer",
                        {"dup": dup_idx, "new_layers": new_nlayer},
                    )

                if args.nlayer_dup_mode == "dup_middle":
                    mid = (old_nlayer - 1) // 2
                    _nlayer_candidate(mid, f"+1_dup_mid{mid}")
                elif args.nlayer_dup_mode == "dup_each":
                    for dup_idx in range(old_nlayer):
                        _nlayer_candidate(dup_idx, f"+1_dup{dup_idx}")
                else:
                    raise ValueError(f"Unknown --nlayer_dup_mode={args.nlayer_dup_mode}")

                continue

            if isinstance(base_val, (int, float)):
                for m in range(1, args.iterations + 1):
                    new_val = _numeric_add(base_val, m * step_spec)
                    cfg_tmpl = deepcopy(baseline_cfg)
                    cfg_tmpl[pname] = new_val
                    _evaluate(cfg_tmpl, pname, new_val)
                continue

            if isinstance(base_val, list):
                if isinstance(step_spec, list):
                    if len(step_spec) != len(base_val):
                        sys.exit(
                            f"--increments for '{pname}' must be 1 value or "
                            f"{len(base_val)} values (got {len(step_spec)})"
                        )
                    per_idx_steps = step_spec
                else:
                    per_idx_steps = [step_spec] * len(base_val)

                for idx, elem in enumerate(base_val):
                    if not isinstance(elem, (int, float)):
                        continue
                    step_here = per_idx_steps[idx]

                    for m in range(1, args.iterations + 1):
                        new_elem = _numeric_add(elem, m * step_here)
                        new_list = deepcopy(base_val)
                        new_list[idx] = new_elem
                        cfg_tmpl = deepcopy(baseline_cfg)
                        cfg_tmpl[pname] = new_list
                        _evaluate(cfg_tmpl, f"{pname}[{idx}]", new_elem)
                continue

            print(f"[SKIP] '{pname}' is neither numeric nor list-numeric – ignored")

        if best_choice is None:
            if args.max_iters_increase is not None and cur_iter < args.num_iterations:
                current_max_iters = baseline_cfg.get("max_iters")
                if current_max_iters is not None:
                    new_max_iters = current_max_iters + args.max_iters_increase
                    print(
                        "[ACTION] No positive-efficiency candidate. Increasing "
                        f"'max_iters' from {current_max_iters} to {new_max_iters}."
                    )
                    baseline_cfg["max_iters"] = new_max_iters
                    log["iterations"].append(
                        {
                            "iter": cur_iter,
                            "baseline_metrics": {
                                "loss": base_loss,
                                "score": base_score,
                                "params": base_params,
                                "peak_torch_allocated_mb": base_torch_alloc,
                                "peak_torch_reserved_mb": base_torch_reserved,
                                "peak_process_gpu_mb": base_process_gpu,
                                "iter_latency_avg": base_iter_ms,
                                "best_iter": log["iterations"][-1]["baseline_metrics"]["best_iter"],
                                "rankme": base_rankme,
                                "areq": base_areq,
                            },
                            "candidates": candidates,
                            "chosen": None,
                            "action": f"max_iters_increased_to_{new_max_iters}",
                            "baseline_config_after": deepcopy(baseline_cfg),
                        }
                    )
                    log["baseline_config"] = deepcopy(baseline_cfg)
                    save_log(log_path, log)
                    cur_iter += 1
                    continue
                else:
                    print(
                        "Warning: --max_iters_increase specified, but 'max_iters' is "
                        "not defined in the baseline config. Stopping."
                    )

            print("No positive-efficiency candidate — stopping.")
            log["stop_reason"] = "no_positive_efficiency"
            log["baseline_config"] = deepcopy(baseline_cfg)
            save_log(log_path, log)
            break

        _, chosen = best_choice
        print(
            f"[CHOSEN] {chosen['param']} → {chosen['value']}  eff={chosen['efficiency']:.3e}"
        )

        if chosen["param"] == "n_layer":
            dup_idx = chosen["value"]["dup"]
            new_layers = chosen["value"]["new_layers"]
            baseline_cfg["n_layer"] = new_layers
            _extend_layerlists(baseline_cfg, dup_idx)
            baseline_cfg["_last_dup_idx"] = dup_idx
        elif (m := re.fullmatch(r"(\w+_layerlist)\[(\d+)\]", chosen["param"])):
            list_key, str_idx = m.groups()
            idx = int(str_idx)

            if list_key not in baseline_cfg or not isinstance(baseline_cfg[list_key], list):
                raise RuntimeError(
                    f"BUG: expected {list_key} to be a list in baseline_cfg"
                )

            while idx >= len(baseline_cfg[list_key]):
                baseline_cfg[list_key].append(deepcopy(baseline_cfg[list_key][-1]))

            baseline_cfg[list_key][idx] = chosen["value"]
        else:
            baseline_cfg[chosen["param"]] = chosen["value"]

        base_loss = chosen["avg_loss"]
        base_score = chosen["avg_score"]
        base_params = chosen["num_params"]
        base_torch_alloc = chosen.get("peak_torch_allocated_mb", base_torch_alloc)
        base_torch_reserved = chosen.get("peak_torch_reserved_mb", base_torch_reserved)
        base_process_gpu = chosen.get("peak_process_gpu_mb", base_process_gpu)
        base_iter_ms = chosen.get("iter_latency_avg", base_iter_ms)
        base_rankme = chosen.get("avg_rankme", base_rankme)
        base_areq = chosen.get("avg_areq", base_areq)

        log["iterations"].append(
            {
                "iter": cur_iter,
                "baseline_metrics": {
                    "loss": base_loss,
                    "score": base_score,
                    "params": base_params,
                    "peak_torch_allocated_mb": base_torch_alloc,
                    "peak_torch_reserved_mb": base_torch_reserved,
                    "peak_process_gpu_mb": base_process_gpu,
                    "iter_latency_avg": base_iter_ms,
                    "best_iter": chosen["best_iter"],
                    "rankme": base_rankme,
                    "areq": base_areq,
                },
                "candidates": candidates,
                "chosen": chosen,
                "baseline_config_after": deepcopy(baseline_cfg),
            }
        )
        log["baseline_config"] = deepcopy(baseline_cfg)
        save_log(log_path, log)
        cur_iter += 1


if __name__ == "__main__":
    main()
