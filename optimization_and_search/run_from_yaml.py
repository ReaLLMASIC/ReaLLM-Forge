import argparse
from pathlib import Path
import json
import subprocess
from pathlib import Path
from datetime import datetime
from itertools import product
import argparse
import os
import sys
from typing import Optional

import yaml
from rich import print
from rich.console import Console
from rich.table import Table

# Optional: import your run_experiment function from your main script
# from your_main_script import run_experiment

# Constants
# LOG_DIR = Path("exploration_logs")
# LOG_DIR.mkdir(exist_ok=True)
METRICS_FILENAME = "best_val_loss_and_iter.txt"
METRIC_KEYS = [
    "best_val_loss",
    "best_val_iter", 
    "best_tokens",
    "num_params",
]


def _parse_override_args(arg_list: list[str] | None) -> dict:
    """Parse --override_args entries like ["batch_size=32", "learning_rate=0.001"].

    Uses yaml.safe_load to infer types (int/float/bool/list/str).
    Invalid items (without '=') are ignored with a warning.
    """
    if not arg_list:
        return {}
    overrides: dict = {}
    for item in arg_list:
        if "=" not in item:
            print(f"[yellow]Ignoring malformed override (missing '='):[/] {item}")
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        try:
            # yaml.safe_load gives us nice typing for numbers, bools, lists, etc.
            parsed_val = yaml.safe_load(value)
        except Exception:
            parsed_val = value  # fallback to raw string
        overrides[key] = parsed_val
    return overrides


def format_run_name(combo: dict, base: str, prefix: str, row_index: int) -> str:
    """Create a unique run name.

    Preferred scheme (stable, short): <base><prefix>-row<row_index>
    Fallback if row_index is None: concatenate parameter values (legacy behavior).
    """
    return f"{prefix}-row{row_index}"


def read_metrics(out_dir: str) -> dict:
    """
    Read best_val_loss_and_iter.txt and parse the first four metrics.

    Returns:
        Dict with keys: best_val_loss, best_val_iter, best_tokens, num_params.
    """
    path = Path(out_dir) / METRICS_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    line = path.read_text().strip()
    parts = [p.strip() for p in line.split(',')]

    # Take only the first 4 values and cast them appropriately
    if len(parts) < len(METRIC_KEYS):
        raise ValueError(f"Expected at least {len(METRIC_KEYS)} metrics, got {len(parts)}")
    
    casts = [float, int, int, int]
    return {k: typ(v) for k, typ, v in zip(METRIC_KEYS, casts, parts[:len(METRIC_KEYS)])}


def completed_runs(log_file: Path) -> set[str]:
    """
    Return set of run names already logged in YAML file.
    """
    if not log_file.exists():
        return set()
    runs = set()
    for doc in yaml.safe_load_all(log_file.open()):
        if doc and 'formatted_name' in doc:
            runs.add(doc['formatted_name'])
    return runs


def append_log(log_file: Path, name: str, combo: dict, metrics: dict) -> None:
    """
    Append a YAML entry with run details and metrics.
    """
    entry = {'formatted_name': name, 'config': combo, **metrics}
    with log_file.open('a') as f:
        yaml.safe_dump(entry, f, explicit_start=True)


def build_command(combo: dict) -> list[str]:
    """
    Construct the command-line invocation for train.py.
    """
    cmd = ['python3', 'train.py', '--compile']
    for k, v in combo.items():
        if k == 'idx':  # skip the 'idx' parameter
            continue
        if isinstance(v, bool):
            cmd.append(f"--{'' if v else 'no-'}{k}")
        elif isinstance(v, list):
            # For list parameters, add each element as a separate argument
            cmd.append(f"--{k}")
            cmd.extend(str(x) for x in v)
        else:
            cmd += [f"--{k}", str(v)]
    return cmd


def run_experiment(
    combo: dict,
    base: str,
    args: argparse.Namespace,
    row_index: Optional[int] = None,
) -> bool:
    """
    Execute one experiment combo: skip if done, run train.py, record metrics.

    Returns:
        True if the subprocess completed successfully (or was skipped/dry-run),
        False if the subprocess returned a non-zero exit code.
    """
    run_name = format_run_name(combo, base, args.prefix, row_index=row_index) 
    LOG_DIR = base_path = Path(args.output_dir)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"{base}.yaml"
    print(f"writing to log file: {log_file}")
    if run_name in completed_runs(log_file):
        print(f"[yellow]Skipping already-run:[/] {run_name}")
        return True

    # Prepare output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') if args.use_timestamp else None
    out_dir_name = f"{timestamp}_{run_name}" if timestamp else run_name
    combo['out_dir'] = os.path.join(args.output_dir, out_dir_name)

    # Prepare tensorboard run name
    combo['tensorboard_run_name'] = run_name

    # Show parameters
    console = Console()
    table = Table("Parameters", show_header=False)
    for k, v in combo.items():
        table.add_row(k, str(v))
    console.print(table)

    # Build and run
    cmd = build_command(combo)
    print(f"Running: {' '.join(cmd)}")

    # Dry run: only print command, do not execute or log
    if getattr(args, "dry_run", False):
        print("[cyan]Dry run enabled — skipping execution.[/]")
        return True
    
    # Set environment variables for memory management
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    try:
        subprocess.run(cmd, check=True, env=env)
        proc_ok = True
    except subprocess.CalledProcessError:
        print(f"[red]Process exited with error for run:[/] {run_name}")
        proc_ok = False

    # Read metrics (use existing or nan on failure)
    try:
        metrics = read_metrics(str(combo['out_dir']))
    except Exception:
        metrics = {k: float("nan") for k in METRIC_KEYS}

    append_log(log_file, run_name, combo, metrics)

    print(f"[green]Experiment completed for:[/] {run_name}")
    return proc_ok

def main(yaml_path, base, args) -> int:
    with open(yaml_path, "r") as f:
        # Load YAML data - expect a list of configuration dictionaries
        yaml_data = yaml.safe_load(f)
        
        # Handle different YAML structures
        if isinstance(yaml_data, list):
            configs = yaml_data
        elif isinstance(yaml_data, dict) and 'configs' in yaml_data:
            configs = yaml_data['configs']
        else:
            raise ValueError("YAML file should contain either a list of configs or a dict with 'configs' key")
        
        # Parse user-provided overrides once; CLI should have highest precedence
        cli_overrides = _parse_override_args(getattr(args, "override_args", None))

    any_failed = False
    for row_index, config in enumerate(configs):
            # Start with the config from YAML
            dynamic_cfg = config.copy()

            # Ensure required training runtime parameters are set/overridden locally.
            overrides = {
                "batch_size": 64,  # Reduced from 128 to save memory
                "device": "cuda",
                "dataset": "minipile",
                "eval_iters": 50,  # Reduced from default to save memory
                "gradient_accumulation_steps": 2,  # Compensate for smaller batch with grad accumulation
                # "compute_model_stats": False,  # Disable model stats to save memory
                "dtype": "bfloat16",  # Use bfloat16 to save memory vs float16
                # "norm_variant_abs": "hyperspherenorm",
                # "hsnorm_gain": True,
                # "hsnorm_scale": 5.0,
                # "attn_residual_combination": "slerp",
                # "mlp_residual_combination": "slerp",
            }
            # Apply overrides (explicit local precedence)
            dynamic_cfg.update(overrides)

            # Finally, apply CLI overrides with highest precedence
            if cli_overrides:
                dynamic_cfg.update(cli_overrides)

            # Run experiment
            ok = run_experiment(dynamic_cfg, base, args, row_index=row_index)
            if not ok:
                any_failed = True

    return 1 if any_failed else 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", type=str, required=True, help="Path to YAML file with configs")
    parser.add_argument("--output_dir", type=str, default="out", help="Base output directory")
    parser.add_argument("--use_timestamp", action="store_true", help="Use timestamp in output directory names")
    parser.add_argument("--prefix", type=str, default="train", help="Prefix for run names")
    parser.add_argument("--override_args", type=str, nargs='*', help="Additional args to override YAML configs, e.g., --override_args batch_size=32 learning_rate=0.001")
    parser.add_argument("--dry_run", action="store_true", help="If set, only print commands without executing")
    args, unknown = parser.parse_known_args()

    yaml_path = Path(args.yaml)
    exit_code = 0
    try:
        exit_code = int(main(yaml_path, args.output_dir, args))
    except Exception as e:
        print(f"[red]Fatal error:[/] {e}")
        exit_code = 2
    sys.exit(exit_code)

