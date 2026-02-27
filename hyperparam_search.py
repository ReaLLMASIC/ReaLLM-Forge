# hyperparam_search.py
"""
Greedy iterative hyper-parameter search for **train.py**

Changes vs. previous version
----------------------------
* `--random_iterations N` — run each candidate N times with **different seeds**.
  - The seed passed to `train.py` is       `baseline_seed + run_id`
    where `baseline_seed` is whatever is in the current baseline config
    (or the default 1337 if absent).
  - The candidate block in *sweep_log.yaml* now includes a `seeds:` list
    with per-seed results.
  - Efficiency is computed on the **average score** across those N runs.
* Everything else (YAML log structure, early-stop rule, resume, etc.) is
  unchanged.
"""

import argparse
import gc
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from contextlib import contextmanager
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import yaml


import ast


# ExecuTorch export settings
@dataclass(slots=True)
class ExecuTorchExportOptions:
    enabled: bool = False
    delegate: str = 'none'
    smoke_test_tokens: int = 0
    smoke_test_prompt: str | None = None
    tokenizer_vocab: Path | None = None
    max_output_tokens: int = 32


def maybe_export_executorch(ckpt_dir: Path, run_label: str, options: ExecuTorchExportOptions) -> None:
    if not options.enabled:
        return

    ckpt_path = ckpt_dir / 'ckpt.pt'
    if not ckpt_path.exists():
        print(f"[WARN] ExecuTorch export skipped (missing {ckpt_path}).")
        return

    try:
        from model_exports.executorch.exporter import ExportConfig, export_checkpoint_to_pte
    except ImportError as exc:
        print(f"[WARN] ExecuTorch export unavailable: {exc}")
        return

    export_dir = ckpt_path.parent / 'executorch'
    export_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r'[^A-Za-z0-9._-]+', '_', run_label).strip('_') or 'model'
    pte_path = export_dir / f"{safe_name}.pte"

    config = ExportConfig(
        delegate=options.delegate,
        generate_etrecord=False,
        smoke_test_tokens=max(0, options.smoke_test_tokens),
        smoke_test_prompt=options.smoke_test_prompt,
        tokenizer_path=options.tokenizer_vocab,
        max_output_tokens=options.max_output_tokens,
        metadata=True,
    )

    try:
        export_checkpoint_to_pte(ckpt_path, pte_path, config)
        print(f"[INFO] ExecuTorch export ready: {pte_path}")
    except ImportError as exc:
        print(f"[WARN] ExecuTorch export failed (missing dependency): {exc}")
    except Exception as exc:
        print(f"[ERROR] ExecuTorch export failed for {run_label}: {exc}")


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
        # Skip internal/meta fields
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


def run_trial_inproc(
    cfg: Dict[str, Any],
    export_options: ExecuTorchExportOptions,
    run_label: str,
) -> Tuple[float, float, int, float, float]:
    """Return (best_val_loss, num_params, best_iter, peak_gpu_mb, iter_latency_ms)."""
    from train import Trainer
    from train_args import parse_args as parse_train_args

    cli = ["train.py"] + dict_to_cli(cfg)
    with patched_argv(cli):
        args, mg, tg, lg = parse_train_args()
    tr = Trainer(args, mg, tg, lg)
    tr.train()
    loss = float(tr.best_val_loss)
    nparam = float(tr.raw_model.num_param)
    best_iter = int(getattr(tr, "iter_num_best_val_loss", 0))
    peak_gpu_mb = float(getattr(tr, "peak_gpu_usage", 0.0) / (1024 ** 2))
    iter_latency_ms = float(getattr(tr, "iter_latency_avg", 0.0))
    maybe_export_executorch(Path(cfg.get("out_dir", "out")), run_label, export_options)
    del tr
    torch.cuda.empty_cache()
    gc.collect()
    return loss, nparam, best_iter, peak_gpu_mb, iter_latency_ms


def run_trial_subproc(
    cfg: Dict[str, Any],
    export_options: ExecuTorchExportOptions,
    run_label: str,
) -> Tuple[float, float, int, float, float]:
    script_dir = Path(__file__).parent
    cmd = [sys.executable, str(script_dir / "train.py")] + dict_to_cli(cfg)
    env = {k: v for k, v in os.environ.items() if k not in {"RANK", "WORLD_SIZE"}}
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if p.returncode:
        print(p.stderr)
        raise RuntimeError("train.py failed")

    out_dir = Path(cfg.get("out_dir", "out"))
    line = (out_dir / "best_val_loss_and_iter.txt").read_text().strip().split(',')
    loss = float(line[0])
    best_iter = int(line[1])
    nparam = float(line[2])
    peak_gpu_mb = float(line[5])
    iter_latency_ms = float(line[6])
    maybe_export_executorch(out_dir, run_label, export_options)
    torch.cuda.empty_cache()
    gc.collect()
    return loss, nparam, best_iter, peak_gpu_mb, iter_latency_ms


def load_log(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text()) if path.exists() else {}


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
            nargs="*", # Allows zero or more KEY=VALUE strings
            metavar="KEY=VALUE",
            default=[],
            help="Override baseline config settings from orig_settings before starting the search. Example: --override_cfg max_iters=10000 learning_rate=0.0005 flag=True name='my_exp' path=data/run")
    ap.add_argument(
        "--max_iters_increase",
        type=int,
        default=None,
        help="If set, and no positive-efficiency candidate is found, increase 'max_iters' by this amount.",
    )
    ap.add_argument(
        "--nlayer_dup_mode",
        choices=["dup_middle", "dup_each"],
        default="dup_middle",
        help="Strategy when testing +1 to n_layer:\n"
             "  dup_middle (default) – duplicate the rounded-up middle layer\n"
             "  dup_each             – create one candidate per layer by duplicating it",
    )
    ap.add_argument(
        "--efficiency_target",
        choices=["params", "vram", "iter"],
        default="params",
        help=(
            "Metric to normalize score gain: 'params' (default) for parameter count, "
            "'vram' for peak GPU memory in MB, or 'iter' for average iteration latency in ms."
        ),
    )
    ap.add_argument(
        "--executorch_export",
        dest="executorch_export",
        action='store_true',
        default=True,
        help="Automatically export ExecuTorch programs for each candidate run.",
    )
    ap.add_argument(
        "--no-executorch-export",
        dest="executorch_export",
        action='store_false',
        help="Disable ExecuTorch exports.",
    )
    ap.add_argument(
        "--executorch_delegate",
        choices=['none', 'xnnpack'],
        default='none',
        help="Delegate to target when exporting to ExecuTorch.",
    )
    ap.add_argument(
        "--executorch_smoke_test_tokens",
        type=int,
        default=0,
        help="If >0, run a random-token smoke test after export.",
    )
    ap.add_argument(
        "--executorch_smoke_test_prompt",
        help="Optional prompt to evaluate with the exported program.",
    )
    ap.add_argument(
        "--executorch_tokenizer_vocab",
        help="Path to a vocab.json for ExecuTorch prompt smoke tests.",
    )
    ap.add_argument(
        "--executorch_max_output_tokens",
        type=int,
        default=32,
        help="Maximum decode tokens when running ExecuTorch smoke tests.",
    )



    args = ap.parse_args()

    # match increments to param list
    if len(args.increments) == 1:
        args.increments *= len(args.param_names)
    if len(args.increments) != len(args.param_names):
        sys.exit("--increments length mismatch")

    inc_map = dict(zip(args.param_names, args.increments))
    export_options = ExecuTorchExportOptions(
        enabled=args.executorch_export,
        delegate=args.executorch_delegate,
        smoke_test_tokens=max(0, args.executorch_smoke_test_tokens),
        smoke_test_prompt=args.executorch_smoke_test_prompt,
        tokenizer_vocab=Path(args.executorch_tokenizer_vocab) if args.executorch_tokenizer_vocab else None,
        max_output_tokens=args.executorch_max_output_tokens,
    )

    if args.spawn_subprocess:
        def run_trial(cfg: Dict[str, Any], label: str) -> Tuple[float, float, int, float, float]:
            return run_trial_subproc(cfg, export_options, label)
    else:
        def run_trial(cfg: Dict[str, Any], label: str) -> Tuple[float, float, int, float, float]:
            return run_trial_inproc(cfg, export_options, label)

    baseline_cfg_master = yaml.safe_load(Path(args.orig_settings).read_text())
    log_path = Path(args.results_file)
    log = load_log(log_path)

    # Helper function to apply overrides to a given config dictionary
    def _apply_overrides_to_active_config(config_dict: Dict[str, Any], overrides: List[str], context_msg: str):
        if not overrides: # No overrides to apply
            return

        print(f"[CONFIG_OVERRIDE] Checking {len(overrides)} overrides for {context_msg}...")
        effective_overrides = 0
        for item in args.override_cfg:
            try:
                key, value_str = item.split("=", 1)
            except ValueError:
                sys.exit(f"Error: Invalid override format '{item}'. Expected KEY=VALUE.")

            try:
                # Safely evaluate the value string as a Python literal.
                # Handles numbers (int, float), booleans (True, False), strings (e.g., 'text', "text"),
                # lists (e.g., "[1, 2]"), dicts (e.g., "{'a': 1}").
                value = ast.literal_eval(value_str)
            except (ValueError, SyntaxError):
                # If ast.literal_eval fails (e.g., for an unquoted string like 'cpu' or 'data/my_path'),
                # treat the value as a plain string.
                value = value_str

            original_value = config_dict.get(key)
            if key not in config_dict or original_value != value:
                print(f"  Applying to active config: {key} = {repr(value)} (was: {repr(original_value) if key in config_dict else 'N/A (new key)'}, type: {type(value).__name__})")
                config_dict[key] = value
                effective_overrides += 1
            else:
                print(f"  Skipping (no change): {key} = {repr(value)}")
        if effective_overrides > 0:
            print(f"[CONFIG_OVERRIDE] Applied {effective_overrides} effective overrides to {context_msg}.")

    # Initialise log structure if new
    log.setdefault("baseline_config", deepcopy(baseline_cfg_master))
    log.setdefault("iterations", [])
    # helper: duplicate a layer in every *_layerlist
    def _extend_layerlists(cfg: Dict[str, Any], dup_idx: int) -> None:
        """
        Duplicate element *dup_idx* (0-based) in every X_layerlist that is
        present in *cfg*.  Modifies the dict in place.
        """
        for key, val in cfg.items():
            if key.endswith("_layerlist") and isinstance(val, list) and val:
                src = min(dup_idx, len(val) - 1)
                val.insert(src + 1, deepcopy(val[src]))


    # ── restore / initialise baseline ─────────────────────────
    if log["iterations"]:
        last = log["iterations"][-1]
        baseline_cfg = deepcopy(last["baseline_config_after"])
        base_loss = last["baseline_metrics"]["loss"]
        base_score = last["baseline_metrics"]["score"]
        base_params = last["baseline_metrics"]["params"]
        base_gpu = last["baseline_metrics"].get("peak_gpu_mb", 0.0)
        base_iter_ms = last["baseline_metrics"].get("iter_latency_avg", 0.0)
        cur_iter = last["iter"] + 1
        # Apply overrides to the resumed configuration for the current session
        _apply_overrides_to_active_config(baseline_cfg, args.override_cfg, "resumed baseline_cfg")
    else:
        # For a new sweep, start with baseline_config (which is a copy of baseline_cfg_master)
        baseline_cfg = deepcopy(log["baseline_config"])
        # Apply overrides to this initial config before the first measurement
        _apply_overrides_to_active_config(baseline_cfg, args.override_cfg, "initial baseline_cfg for new sweep")

        print("[BASELINE] measuring initial config …")
        # run_trial receives a deepcopy of the (potentially overridden) baseline_cfg

        base_loss, base_params, base_best_iter, base_gpu, base_iter_ms = run_trial(deepcopy(baseline_cfg), 'baseline')
        base_score = 1 / math.exp(base_loss)
        log["iterations"].append(
            {
                "iter": -1,
                "baseline_metrics": {
                    "loss": base_loss,
                    "score": base_score,
                    "params": base_params,
                    "peak_gpu_mb": base_gpu,
                    "iter_latency_avg": base_iter_ms,
                    "best_iter": base_best_iter,
                },
                "baseline_config_after": deepcopy(baseline_cfg),
            }
        )
        save_log(log_path, log)
        cur_iter = 0

    # ── outer greedy loop ─────────────────────────────────────
    while cur_iter < args.num_iterations:
        print(f"========== Iteration {cur_iter} ==========")
        candidates: List[Dict[str, Any]] = []
        best_choice: Tuple[float, Dict[str, Any]] | None = None

        for pname in args.param_names:
            if pname not in baseline_cfg:
                print(f"[WARN] parameter '{pname}' not in baseline config – skipping")
                continue

            base_val   = baseline_cfg[pname]
            step_spec  = inc_map[pname]                 # could be scalar *or* list

            # ---------- helpers (shared by ALL param kinds) --------------------
            def _numeric_add(x, delta):
                """Add *delta* while keeping int-vs-float type."""
                return int(round(x + delta)) if isinstance(x, int) else float(x + delta)

            def _evaluate(cfg_template: Dict[str, Any],
                          label_for_log: str,
                          value_for_log: Any) -> None:
                """
                Run one candidate (possibly several seeds) and record its
                performance into the surrounding `candidates` / `best_choice`
                variables (declared in the parent scope).
                """
                nonlocal best_choice, candidates

                seed0     = cfg_template.get("seed", 1337)
                seed_runs = []
                scores    = []

                for r in range(args.random_iterations):
                    cfg_run        = deepcopy(cfg_template)
                    cfg_run["seed"] = seed0 + r

                    print(f"[TEST] {label_for_log}={value_for_log}  seed={cfg_run['seed']}")
                    try:
                        loss, nparam, best_it, peak_mb, iter_ms = run_trial(cfg_run, f"{label_for_log}-seed{cfg_run['seed']}")
                    except Exception as exc:
                        print("   ⚠", exc)
                        return                                      # discard this candidate

                    score = 1.0 / math.exp(loss)
                    seed_runs.append({
                                      "seed": cfg_run["seed"],
                                      "loss": loss,
                                      "score": score,
                                      "best_iter": best_it,
                                      "peak_gpu_mb": peak_mb,
                                      "iter_latency_ms": iter_ms})
                    scores.append(score)

                # ── aggregate across seeds ───────────────────────────────────
                avg_score  = sum(scores) / len(scores)
                avg_peak   = sum(s["peak_gpu_mb"] for s in seed_runs) / len(seed_runs)
                avg_iter   = sum(s["iter_latency_ms"] for s in seed_runs) / len(seed_runs)
                avg_loss   = -math.log(avg_score)
                d_score    = avg_score - base_score
                d_param    = nparam     - base_params
                d_vram     = avg_peak   - base_gpu
                d_iter     = avg_iter   - base_iter_ms

                if args.efficiency_target == "params":
                    d_cost = d_param
                elif args.efficiency_target == "vram":
                    d_cost = d_vram
                elif args.efficiency_target == "iter":
                    d_cost = d_iter
                else:
                    raise ValueError("Unknown efficiency target")

                eff = (d_score / d_cost) if d_cost != 0 else (math.inf if d_score > 0 else 0.0)

                cand = {
                    "param":         label_for_log,
                    "value":         value_for_log,
                    "avg_loss":      avg_loss,
                    "avg_score":     avg_score,
                    "best_val_loss": avg_loss,
                    "best_iter":     max(s["best_iter"] for s in seed_runs),
                    "num_params":    nparam,
                    "peak_gpu_mb":   avg_peak,
                    "iter_latency_avg": avg_iter,
                    "delta_score":   d_score,
                    "delta_params":  d_param,
                    "delta_vram":    d_vram,
                    "delta_iter_latency": d_iter,
                    "efficiency":    eff,
                    "seeds":         seed_runs,
                }
                candidates.append(cand)

                # keep global best
                if eff > 0:
                    if best_choice is None:
                        best_choice = (eff, cand)
                    else:
                        old_eff, old_cand = best_choice
                        if (eff > old_eff) or (math.isinf(eff) and eff == old_eff
                                               and cand["delta_score"] > old_cand["delta_score"]):
                            best_choice = (eff, cand)


            # ------------------------------------------------------------------
            # Special handling for *n_layer* (+1 with layer duplication)
            # ------------------------------------------------------------------
            if pname == "n_layer":
                old_nlayer   = int(baseline_cfg["n_layer"])
                new_nlayer   = old_nlayer + 1

                def _nlayer_candidate(dup_idx: int, tag: str):
                    cfg2              = deepcopy(baseline_cfg)
                    cfg2["n_layer"]   = new_nlayer
                    _extend_layerlists(cfg2, dup_idx)
                    # store which layer got duplicated (for logging/debug)
                    cfg2["_last_dup_idx"] = dup_idx
                    _evaluate(cfg2, "n_layer", {"dup": dup_idx,
                                                "new_layers": new_nlayer})

                if args.nlayer_dup_mode == "dup_middle":
                    mid = (old_nlayer - 1) // 2     # rounded-up middle
                    _nlayer_candidate(mid, f"+1_dup_mid{mid}")

                elif args.nlayer_dup_mode == "dup_each":
                    for dup_idx in range(old_nlayer):
                        _nlayer_candidate(dup_idx, f"+1_dup{dup_idx}")

                else:
                    raise ValueError(f"Unknown --nlayer_dup_mode={args.nlayer_dup_mode}")

                continue         # done with 'n_layer', next pname


            # ---------- scalar hyper-parameters ---------------------------------
            if isinstance(base_val, (int, float)):
                for m in range(1, args.iterations + 1):
                    new_val        = _numeric_add(base_val, m * step_spec)
                    cfg_tmpl       = deepcopy(baseline_cfg)
                    cfg_tmpl[pname] = new_val
                    _evaluate(cfg_tmpl, pname, new_val)
                continue   # next pname

            # ---------- list hyper-parameters (e.g. mlp_size_layerlist) ---------
            if isinstance(base_val, list):
                # allow a scalar step or list-of-steps (one per index)
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
                        continue                            # skip non-numeric slots
                    step_here = per_idx_steps[idx]

                    for m in range(1, args.iterations + 1):
                        new_elem        = _numeric_add(elem, m * step_here)
                        new_list        = deepcopy(base_val)
                        new_list[idx]   = new_elem
                        cfg_tmpl        = deepcopy(baseline_cfg)
                        cfg_tmpl[pname] = new_list
                        _evaluate(cfg_tmpl, f"{pname}[{idx}]", new_elem)
                continue   # next pname

            # ---------- unsupported types ---------------------------------------
            print(f"[SKIP] '{pname}' is neither numeric nor list-numeric – ignored")

        # -- pick or stop ---------------------------------------
        if best_choice is None:
            if args.max_iters_increase is not None and cur_iter < args.num_iterations:
                current_max_iters = baseline_cfg.get("max_iters")
                if current_max_iters is not None:
                    new_max_iters = current_max_iters + args.max_iters_increase
                    print(f"[ACTION] No positive-efficiency candidate. Increasing 'max_iters' from {current_max_iters} to {new_max_iters}.")
                    baseline_cfg["max_iters"] = new_max_iters
                    # Log the change to the baseline config for this iteration
                    log["iterations"].append(
                        {
                            "iter": cur_iter,
                            "baseline_metrics": {
                                "loss": base_loss, # Keep current baseline metrics
                                "score": base_score,
                                "params": base_params,
                                "peak_gpu_mb": base_gpu,
                                "iter_latency_avg": base_iter_ms,
                                "best_iter": log["iterations"][-1]["baseline_metrics"]["best_iter"],
                            },
                            "candidates": candidates, # Log candidates for this unproductive iteration
                            "chosen": None, # Indicate no candidate was chosen for this iteration
                            "action": f"max_iters_increased_to_{new_max_iters}", # Custom action field
                            "baseline_config_after": deepcopy(baseline_cfg),
                        }
                    )
                    save_log(log_path, log)
                    cur_iter += 1 # Advance iteration as an action was taken
                    continue # Continue to the next outer loop iteration
                else:
                    print(f"Warning: --max_iters_increase specified, but 'max_iters' is not defined in the baseline config. Stopping.")

            print("No positive-efficiency candidate — stopping.")
            log["stop_reason"] = "no_positive_efficiency"
            save_log(log_path, log)
            break

        _, chosen = best_choice
        print(
            f"[CHOSEN] {chosen['param']} → {chosen['value']}  eff={chosen['efficiency']:.3e}"
        )

        # ───────────── baseline update ───────────────────────────────
        if chosen["param"] == "n_layer":
            # `chosen["value"]` is the dict we stuffed into the log:
            #   {"dup": dup_idx, "new_layers": new_nlayer}
            dup_idx    = chosen["value"]["dup"]
            new_layers = chosen["value"]["new_layers"]

            # 1) keep n_layer an *integer*
            baseline_cfg["n_layer"] = new_layers

            # 2) replicate the duplicated layer in every *_layerlist
            _extend_layerlists(baseline_cfg, dup_idx)

            # 3) (optional) remember for debugging / inspection
            baseline_cfg["_last_dup_idx"] = dup_idx

        # else:
        #     baseline_cfg[chosen["param"]] = chosen["value"]
        # 2) list element like “mlp_size_layerlist[3]”
        elif (m := re.fullmatch(r"(\w+_layerlist)\[(\d+)]", chosen["param"])) :
            list_key, str_idx = m.groups()
            idx = int(str_idx)

            if list_key not in baseline_cfg or not isinstance(baseline_cfg[list_key], list):
                raise RuntimeError(
                    f"BUG: expected {list_key} to be a list in baseline_cfg")

            # grow the list if the dup-each mode added a new tail element
            while idx >= len(baseline_cfg[list_key]):
                baseline_cfg[list_key].append(
                    deepcopy(baseline_cfg[list_key][-1]))

            baseline_cfg[list_key][idx] = chosen["value"]

        # 3) ordinary scalar / list parameters
        else:
            baseline_cfg[chosen["param"]] = chosen["value"]
        base_loss = chosen["avg_loss"]
        base_score = chosen["avg_score"]
        base_params = chosen["num_params"]
        base_gpu = chosen.get("peak_gpu_mb", base_gpu)
        base_iter_ms = chosen.get("iter_latency_avg", base_iter_ms)

        # log block
        log["iterations"].append(
            {
                "iter": cur_iter,
                "baseline_metrics": {
                    "loss": base_loss,
                    "score": base_score,
                    "params": base_params,
                    "peak_gpu_mb": base_gpu,
                    "iter_latency_avg": base_iter_ms,
                    "best_iter": chosen["best_iter"],
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
