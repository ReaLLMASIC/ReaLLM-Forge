import json
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from itertools import product
import argparse
import os
import re
from copy import deepcopy

import yaml
from rich import print
from rich.console import Console
from rich.table import Table

# Constants
LOG_DIR = Path("exploration_logs")
LOG_DIR.mkdir(exist_ok=True)
METRICS_FILENAME = "best_val_loss_and_iter.txt"
METRIC_KEYS = [
    "best_val_loss",
    "best_val_iter",
    "best_val_tokens",
    "num_params",
    "better_than_chance",
    "btc_per_param",
    "peak_gpu_mb",
    "iter_latency_avg",
    "avg_top1_prob",
    "avg_top1_correct",
    "avg_target_rank",
    "avg_target_left_prob",
    "avg_target_prob",
    "target_rank_95",
    "left_prob_95",
    "avg_ln_f_cosine",
    "ln_f_cosine_95",
    "rankme",
    "areq",
]




def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run experiments based on a configuration file (JSON or YAML)."
    )
    parser.add_argument(
        '-c', '--config', required=True,
        help="Path to the configuration file."
    )
    parser.add_argument(
        '--config_format', choices=['json', 'yaml'], default='yaml',
        help="Configuration file format (json or yaml)."
    )
    parser.add_argument(
        '-o', '--output_dir', default="out",
        help="Directory to place experiment outputs."
    )
    parser.add_argument(
        '--prefix', default='',
        help="Optional prefix for run names and output directories."
    )
    parser.add_argument(
        '--use_timestamp', action='store_true',
        help="Prepend timestamp to run names and out_dir."
    )
    parser.add_argument(
        '--include_common_group_in_name', action='store_true',
        help=(
            "Include parameters defined in `common_group` when building run "
            "names, output directories, and CSV filenames. By default these "
            "common parameters are omitted to keep names short."
        ),
    )
    parser.add_argument(
        '--expand_named_groups_in_names', action='store_true',
        help=(
            "Include explicit parameter values contributed by named groups in run "
            "names. By default named groups are abbreviated using their group names "
            "to keep run identifiers shorter."
        ),
    )
    parser.add_argument(
        '--executorch-export',
        dest='executorch_export',
        action='store_true',
        default=True,
        help='Automatically export ExecuTorch programs after each run.',
    )
    parser.add_argument(
        '--no-executorch-export',
        dest='executorch_export',
        action='store_false',
        help='Disable automatic ExecuTorch exports.',
    )
    parser.add_argument(
        '--executorch-delegate',
        choices=['none', 'xnnpack'],
        default='none',
        help='Delegate to target when exporting to ExecuTorch.',
    )
    parser.add_argument(
        '--executorch-smoke-test-tokens',
        type=int,
        default=0,
        help='If >0, run a random-token smoke test after export.',
    )
    parser.add_argument(
        '--executorch-smoke-test-prompt',
        help='Optional prompt to evaluate with the exported program.',
    )
    parser.add_argument(
        '--executorch-tokenizer-vocab',
        help='Path to a vocab.json for ExecuTorch prompt smoke tests.',
    )
    parser.add_argument(
        '--executorch-max-output-tokens',
        type=int,
        default=32,
        help='Maximum decode tokens when running ExecuTorch smoke tests.',
    )
    return parser.parse_args()


def _sanitize_run_name(name: str) -> str:
    sanitized = re.sub(r'[^A-Za-z0-9._-]+', '_', name).strip('_')
    return sanitized or 'model'

def _merge_parameter_groups(existing, new):
    existing_list = []
    if existing is not None:
        existing_list = existing if isinstance(existing, list) else [existing]
    new_list = []
    if new is not None:
        new_list = new if isinstance(new, list) else [new]

    if not existing_list:
        return [deepcopy(g) for g in new_list]
    if not new_list:
        return [deepcopy(g) for g in existing_list]

    combined = []
    for existing_group in existing_list:
        group_copy = deepcopy(existing_group)
        nested_existing = group_copy.get('parameter_groups')
        group_copy['parameter_groups'] = _merge_parameter_groups(nested_existing, new_list)
        combined.append(group_copy)
    return combined


def _merge_named_metadata(dest: dict, src: dict) -> None:
    if '_named_group_fragments' in src:
        dest.setdefault('_named_group_fragments', [])
        dest['_named_group_fragments'].extend(src['_named_group_fragments'])
    if '_named_group_param_keys' in src:
        dest.setdefault('_named_group_param_keys', {})
        for key, group in src['_named_group_param_keys'].items():
            dest['_named_group_param_keys'].setdefault(key, group)


def _merge_config_dicts(base: dict, addition: dict) -> dict:
    result = {k: deepcopy(v) for k, v in base.items()}
    for key, value in addition.items():
        if key == '_named_group_fragments' or key == '_named_group_param_keys':
            result.setdefault(key, []) if key == '_named_group_fragments' else result.setdefault(key, {})
            _merge_named_metadata(result, {key: deepcopy(value)})
            continue
        if key == 'parameter_groups':
            merged = _merge_parameter_groups(result.get(key), value)
            if merged == []:
                result.pop(key, None)
            else:
                result[key] = merged
            continue
        if key in result and result[key] != value:
            raise ValueError(
                f"Conflicting values for parameter '{key}' when merging named groups: "
                f"{result[key]!r} vs {value!r}"
            )
        result[key] = deepcopy(value)
    return result


def _collect_param_keys(cfg: dict) -> set[str]:
    keys: set[str] = set()
    for key in cfg:
        if key in {'parameter_groups', '_named_group_fragments', '_named_group_param_keys'}:
            continue
        if key.startswith('_'):
            continue
        keys.add(key)
    return keys


class NamedGroupRegistry:
    def __init__(self, static_groups: dict[str, dict], variation_groups: dict[str, dict]):
        self.static_groups = static_groups
        self.variation_groups = variation_groups

    @classmethod
    def from_config(cls, cfg: dict) -> tuple['NamedGroupRegistry', dict]:
        cfg = dict(cfg)
        raw_static = cfg.pop('named_static_groups', [])
        raw_variation = cfg.pop('named_variation_groups', [])

        static_groups: dict[str, dict] = {}
        for entry in _ensure_list(raw_static):
            if not isinstance(entry, dict):
                raise TypeError("Entries in 'named_static_groups' must be mappings")
            name = entry.get('named_group')
            if not name:
                raise ValueError("Each named static group must include 'named_group'")
            if name in static_groups:
                raise ValueError(f"Duplicate named static group '{name}'")
            body = {k: deepcopy(v) for k, v in entry.items() if k != 'named_group'}
            static_groups[name] = body

        variation_groups: dict[str, dict] = {}
        for entry in _ensure_list(raw_variation):
            if not isinstance(entry, dict):
                raise TypeError("Entries in 'named_variation_groups' must be mappings")
            name = entry.get('named_group')
            if not name:
                raise ValueError("Each named variation group must include 'named_group'")
            if name in variation_groups:
                raise ValueError(f"Duplicate named variation group '{name}'")
            body = {k: deepcopy(v) for k, v in entry.items() if k != 'named_group'}
            variation_groups[name] = body

        return cls(static_groups, variation_groups), cfg

    def resolve_static(self, name: str, visited_static: set[str] | None, visited_variation: set[str] | None):
        if name not in self.static_groups:
            raise KeyError(f"Unknown static named group '{name}'")
        visited_static = set(visited_static or set())
        if name in visited_static:
            raise ValueError(f"Circular dependency detected in static named groups involving '{name}'")
        visited_static.add(name)
        body = deepcopy(self.static_groups[name])
        resolved = expand_named_config(body, self, visited_static, visited_variation)
        if len(resolved) != 1:
            raise ValueError(
                f"Static named group '{name}' expanded to {len(resolved)} configurations; expected exactly one"
            )
        for cfg in resolved:
            fragments = cfg.get('_named_group_fragments', [])
            fragments.append(name)
            cfg['_named_group_fragments'] = fragments
            keys = _collect_param_keys(cfg)
            key_map = cfg.get('_named_group_param_keys', {})
            for key in keys:
                key_map.setdefault(key, name)
            cfg['_named_group_param_keys'] = key_map
        return resolved

    def resolve_variation(self, name: str, visited_static: set[str] | None, visited_variation: set[str] | None):
        if name not in self.variation_groups:
            raise KeyError(f"Unknown variation named group '{name}'")
        visited_variation = set(visited_variation or set())
        if name in visited_variation:
            raise ValueError(
                f"Circular dependency detected in variation named groups involving '{name}'"
            )
        visited_variation.add(name)
        body = deepcopy(self.variation_groups[name])
        return expand_named_config(body, self, visited_static, visited_variation)

    def resolve(self, name: str, visited_static: set[str] | None, visited_variation: set[str] | None):
        if name in self.static_groups:
            return self.resolve_static(name, visited_static, visited_variation)
        if name in self.variation_groups:
            return self.resolve_variation(name, visited_static, visited_variation)
        raise KeyError(f"Unknown named group '{name}'")


def expand_named_config(
    cfg: dict,
    registry: NamedGroupRegistry,
    visited_static: set[str] | None = None,
    visited_variation: set[str] | None = None,
):
    base = {k: deepcopy(v) for k, v in cfg.items()}
    fragments = base.pop('_named_group_fragments', [])
    param_key_map = base.pop('_named_group_param_keys', {})

    settings = base.pop('named_group_settings', None)
    if settings is not None:
        if not isinstance(settings, dict):
            raise TypeError("'named_group_settings' must be a mapping")
        base = _merge_config_dicts(base, settings)

    alternate_names = _ensure_list(base.pop('named_group_alternates', []))
    static_names = _ensure_list(base.pop('named_group_static', []))
    variation_names = _ensure_list(base.pop('named_group_variations', []))

    if alternate_names:
        expanded = []
        for name in alternate_names:
            resolved_list = registry.resolve(name, visited_static, visited_variation)
            if not resolved_list:
                resolved_list = [{}]
            for resolved in resolved_list:
                merged = _merge_config_dicts(base, resolved)
                if static_names:
                    merged['named_group_static'] = list(static_names) + _ensure_list(merged.get('named_group_static', []))
                if variation_names:
                    merged['named_group_variations'] = list(variation_names) + _ensure_list(merged.get('named_group_variations', []))
                _merge_named_metadata(merged, {'_named_group_fragments': fragments})
                _merge_named_metadata(merged, {'_named_group_param_keys': param_key_map})
                expanded.extend(expand_named_config(merged, registry, visited_static, visited_variation))
        return expanded

    cfg_list: list[tuple[dict, list[str], dict]] = [(base, fragments, param_key_map)]

    for name in static_names:
        resolved_list = registry.resolve_static(name, visited_static, visited_variation)
        new_list: list[tuple[dict, list[str], dict]] = []
        for current_cfg, current_frags, current_map in cfg_list:
            for resolved in resolved_list:
                merged = _merge_config_dicts(current_cfg, resolved)
                fragments_copy = list(current_frags)
                fragments_copy.extend(resolved.get('_named_group_fragments', []))
                param_map = dict(current_map)
                for key, group_name in resolved.get('_named_group_param_keys', {}).items():
                    param_map.setdefault(key, group_name)
                new_list.append((merged, fragments_copy, param_map))
        cfg_list = new_list

    for name in variation_names:
        resolved_list = registry.resolve_variation(name, visited_static, visited_variation)
        if not resolved_list:
            resolved_list = [{}]
        new_list = []
        for current_cfg, current_frags, current_map in cfg_list:
            for resolved in resolved_list:
                merged = _merge_config_dicts(current_cfg, resolved)
                fragments_copy = list(current_frags)
                fragments_copy.extend(resolved.get('_named_group_fragments', []))
                param_map = dict(current_map)
                for key, group_name in resolved.get('_named_group_param_keys', {}).items():
                    param_map.setdefault(key, group_name)
                new_list.append((merged, fragments_copy, param_map))
        cfg_list = new_list

    final = []
    for current_cfg, current_frags, current_map in cfg_list:
        if current_frags:
            current_cfg['_named_group_fragments'] = current_frags
        if current_map:
            current_cfg['_named_group_param_keys'] = current_map
        final.append(current_cfg)
    return final


def _extract_common_group(cfg: dict) -> tuple[dict, set[str]]:
    """Extract shared parameters defined under ``common_group``.

    Returns a tuple of (common_values, common_keys). The ``cfg`` mapping is
    mutated in-place to remove the ``common_group`` entry so it is not processed
    again during combination generation.
    """

    raw_common = cfg.pop('common_group', None)
    if raw_common is None:
        return {}, set()

    if not isinstance(raw_common, dict):
        raise TypeError("'common_group' must be a mapping of parameter names to values")

    common: dict[str, object] = {}
    for key, value in raw_common.items():
        normalized = expand_range(value)
        if isinstance(normalized, list):
            if len(normalized) != 1:
                raise ValueError(
                    "Values in 'common_group' must resolve to a single option; "
                    f"got {len(normalized)} options for '{key}'"
                )
            normalized = normalized[0]
        if isinstance(normalized, dict):
            raise ValueError(
                "Values in 'common_group' cannot contain nested option structures"
            )
        common[key] = normalized

    return common, set(common)


def generate_combinations(config: dict):
    """Yield all valid parameter combinations for a config dict.

    Supports arbitrarily nested ``parameter_groups`` and the optional
    ``common_group`` block.
    """

    registry, cfg = NamedGroupRegistry.from_config(config)
    cfg = dict(cfg)
    common_values, common_keys = _extract_common_group(cfg)

    def _expand_base_and_conditionals(cfg: dict):
        # Split plain parameters (base) from conditional specs
        base = {
            k: (expand_range(v) if isinstance(v, dict) and 'range' in v else v)
            for k, v in cfg.items()
            if not (isinstance(v, dict) and 'conditions' in v)
               and k != 'parameter_groups'
        }
        # Ensure each base value is iterable for cartesian product
        base = {k: (v if isinstance(v, list) else [v]) for k, v in base.items()}

        conditionals = {
            k: v for k, v in cfg.items()
            if isinstance(v, dict) and 'conditions' in v
        }
        return base, conditionals

    def _conditions_match(combo: dict, raw_conditions):
        # dict => AND of all pairs; list[dict] => OR across dicts, AND within each dict
        if isinstance(raw_conditions, dict):
            return all(combo.get(k) == v for k, v in raw_conditions.items())
        if isinstance(raw_conditions, list):
            clauses = [d for d in raw_conditions if isinstance(d, dict)]
            if not clauses:
                return False
            return any(all(combo.get(k) == v for k, v in d.items()) for d in clauses)
        return False

    def _apply_conditionals(combo_dict: dict, conditionals: dict):
        valid = [combo_dict]
        for param, spec in conditionals.items():
            next_valid = []
            raw_conditions = spec.get('conditions', {})
            opts = spec.get('options', [])
            options = opts if isinstance(opts, list) else [opts]

            for c in valid:
                if _conditions_match(c, raw_conditions):
                    for opt in options:
                        new_c = dict(c)
                        new_c[param] = opt
                        next_valid.append(new_c)
                else:
                    # If conditions don't match, leave combo unchanged
                    next_valid.append(c)
            valid = next_valid
        return valid

    def recurse(cfg: dict):
        expanded_cfgs = expand_named_config(cfg, registry)
        for expanded in expanded_cfgs:
            metadata = {
                key: deepcopy(expanded[key])
                for key in ('_named_group_fragments', '_named_group_param_keys')
                if key in expanded
            }
            expanded_clean = {
                k: v for k, v in expanded.items() if k not in metadata
            }

            groups = expanded_clean.get('parameter_groups')
            if groups:
                groups_list = groups if isinstance(groups, list) else [groups]
                base_cfg = {k: v for k, v in expanded_clean.items() if k != 'parameter_groups'}
                for key, value in metadata.items():
                    base_cfg[key] = deepcopy(value)
                for grp in groups_list:
                    merged = _merge_config_dicts(base_cfg, grp)
                    yield from recurse(merged)
                continue

            base, conditionals = _expand_base_and_conditionals(expanded_clean)
            keys = list(base)
            for combo in product(*(base[k] for k in keys)):
                combo_dict = dict(zip(keys, combo))
                for key, value in metadata.items():
                    combo_dict[key] = deepcopy(value)
                for final in _apply_conditionals(combo_dict, conditionals):
                    yield final

    for combo in recurse(cfg):
        merged = dict(common_values)
        for key, value in combo.items():
            if key in merged and merged[key] != value:
                raise ValueError(
                    "Parameters defined in 'common_group' must not be overridden elsewhere"
                )
            merged[key] = value
        yield merged, common_keys


def format_run_name(
    combo: dict,
    base: str,
    prefix: str,
    exclude_keys: set[str] | None = None,
    named_fragments: list[str] | None = None,
    named_param_keys: dict[str, str] | None = None,
    expand_named_group_values: bool = False,
) -> str:
    """Create a unique run name from parameter values."""

    exclude_keys = set(exclude_keys or set())
    if not expand_named_group_values and named_param_keys:
        exclude_keys.update(named_param_keys.keys())

    parts: list[str] = []
    if not expand_named_group_values and named_fragments:
        parts.extend(list(dict.fromkeys(named_fragments)))

    for k, v in combo.items():
        if k in exclude_keys:
            continue
        if k.startswith('_'):
            continue
        if isinstance(v, str) and RUN_NAME_VAR in v:
            continue
        parts.append(str(v))

    base_name = f"{prefix}{base}"
    return f"{base_name}-{'-'.join(parts)}" if parts else base_name


def read_metrics(out_dir: str) -> dict:
    """
    Read best_val_loss_and_iter.txt and parse metrics.

    Returns:
        Dict with keys from METRIC_KEYS.
    """
    path = Path(out_dir) / METRICS_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    line = path.read_text().strip()
    parts = [p.strip() for p in line.split(',')]

    casts = [
        float,
        int,
        int,
        int,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ]

    return {k: typ(v) for k, typ, v in zip(METRIC_KEYS, casts, parts)}


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


def append_progress(log_file: Path, message: str) -> None:
    """Append a timestamped progress message to a log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with log_file.open('a') as f:
        f.write(f"[{timestamp}] {message}\n")


def maybe_export_executorch(run_name: str, out_dir: str, args: argparse.Namespace) -> None:
    if not getattr(args, 'executorch_export', False):
        return

    ckpt_path = Path(out_dir) / 'ckpt.pt'
    if not ckpt_path.exists():
        print(f"[yellow]ExecuTorch export skipped (missing {ckpt_path}).")
        return

    try:
        from model_exports.executorch.exporter import ExportConfig, export_checkpoint_to_pte
    except ImportError as exc:
        print(f"[yellow]ExecuTorch export unavailable: {exc}")
        return

    export_dir = ckpt_path.parent / 'executorch'
    export_dir.mkdir(parents=True, exist_ok=True)
    export_name = _sanitize_run_name(run_name)
    pte_path = export_dir / f"{export_name}.pte"

    tokenizer_path = getattr(args, 'executorch_tokenizer_vocab', None)
    if tokenizer_path:
        tokenizer_path = Path(tokenizer_path)

    config = ExportConfig(
        delegate=getattr(args, 'executorch_delegate', 'none'),
        generate_etrecord=False,
        smoke_test_tokens=max(0, getattr(args, 'executorch_smoke_test_tokens', 0)),
        smoke_test_prompt=getattr(args, 'executorch_smoke_test_prompt', None),
        tokenizer_path=tokenizer_path,
        max_output_tokens=getattr(args, 'executorch_max_output_tokens', 32),
        metadata=True,
    )

    try:
        export_checkpoint_to_pte(ckpt_path, pte_path, config)
        print(f"[green]ExecuTorch export complete:[/] {pte_path}")
    except ImportError as exc:
        print(f"[yellow]ExecuTorch export failed (missing dependency): {exc}")
    except Exception as exc:
        print(f"[red]ExecuTorch export failed for {run_name}: {exc}")


def build_command(combo: dict) -> list[str]:
    """
    Construct the command-line invocation for train.py.
    """
    cmd = ['python3', 'train.py']
    for k, v in combo.items():
        if k.startswith('_'):
            continue
        if isinstance(v, bool):
            cmd.append(f"--{'' if v else 'no-'}{k}")
        elif isinstance(v, list):
            if v:
                cmd.append(f"--{k}")
                cmd.extend([str(x) for x in v])
            else:
                cmd.append(f"--{k}")
        else:
            cmd += [f"--{k}", str(v)]
    return cmd


def run_experiment(
    combo: dict,
    base: str,
    args: argparse.Namespace,
    common_keys: set[str],
) -> None:
    """
    Execute one experiment combo: skip if done, run train.py, record metrics.
    """
    named_fragments = combo.pop('_named_group_fragments', [])
    named_param_keys = combo.pop('_named_group_param_keys', {})
    exclude = set() if args.include_common_group_in_name else common_keys
    run_name = format_run_name(
        combo,
        base,
        args.prefix,
        exclude,
        named_fragments=named_fragments,
        named_param_keys=named_param_keys,
        expand_named_group_values=args.expand_named_groups_in_names,
    )
    log_file = LOG_DIR / f"{base}.yaml"
    if run_name in completed_runs(log_file):
        print(f"[yellow]Skipping already-run:[/] {run_name}")
        return

    # Prepare output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') if args.use_timestamp else None
    out_dir_name = f"{timestamp}_{run_name}" if timestamp else run_name
    combo['out_dir'] = os.path.join(args.output_dir, out_dir_name)

    # Prepare tensorboard run name
    combo['tensorboard_run_name'] = run_name

    # Substitute special run-name token in string parameters
    combo = _substitute_run_name(combo, run_name)

    # Show parameters
    console = Console()
    table = Table("Parameters", show_header=False)
    for k, v in combo.items():
        if k.startswith('_'):
            continue
        table.add_row(k, str(v))
    console.print(table)

    # Build and run
    cmd = build_command(combo)
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print(f"[red]Process exited with error for run:[/] {run_name}")

    maybe_export_executorch(run_name, combo['out_dir'], args)

    # Read metrics (use existing or nan on failure)
    try:
        metrics = read_metrics(str(combo['out_dir']))
    except Exception:
        metrics = {k: float("nan") for k in METRIC_KEYS}

    append_log(log_file, run_name, combo, metrics)


def main():
    args = parse_args()
    base = Path(args.config).stem
    configs = load_configurations(args.config, args.config_format)

    # Precompute all combinations to know total experiment count
    all_combos: list[tuple[dict, set[str]]] = []
    for cfg in configs:
        all_combos.extend(list(generate_combinations(cfg)))

    total = len(all_combos)
    start_time = datetime.now()
    progress_log = LOG_DIR / f"{base}_progress.log"
    for idx, (combo, common_keys) in enumerate(all_combos, 1):
        configs_left = total - idx + 1
        if idx == 1:
            message = (
                "Starting config "
                f"{idx}/{total} ({configs_left} configs left). "
                "Estimated time remaining: N/A. Estimated completion: N/A"
            )
            print(f"[green]{message}[/]")
            append_progress(progress_log, message)
        else:
            now = datetime.now()
            elapsed = (now - start_time).total_seconds()
            avg = elapsed / (idx - 1)
            eta_seconds = int(avg * configs_left)
            eta = timedelta(seconds=eta_seconds)
            finish_time = now + timedelta(seconds=eta_seconds)
            finish_formatted = finish_time.strftime("%Y-%m-%d %H:%M:%S")
            message = (
                "Starting config "
                f"{idx}/{total} ({configs_left} configs left). "
                f"Estimated time remaining: {eta}. "
                f"Estimated completion: {finish_formatted}"
            )
            print(f"[green]{message}[/]")
            append_progress(progress_log, message)
        run_experiment(combo, base, args, common_keys)


if __name__ == '__main__':
    main()
