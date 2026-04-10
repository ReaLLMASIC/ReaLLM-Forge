import json
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from itertools import product
import argparse
import os
from copy import deepcopy

import yaml
from rich import print
from rich.console import Console
from rich.table import Table

# Constants
LOG_DIR = Path("exploration_logs")
LOG_DIR.mkdir(exist_ok=True)
METRICS_FILENAME = "best_val_loss_and_iter.txt"
ZEUS_SUMMARY_FILENAME = "zeus_summary.json"
METRIC_KEYS = [
    "best_val_loss",
    "best_val_iter",
    "best_val_tokens",
    "num_params",
    "better_than_chance",
    "btc_per_param",
    "peak_torch_allocated_mb",
    "peak_torch_reserved_mb",
    "peak_process_gpu_mb",
    "iter_latency_avg",
    "zeus_best_train_step_energy_j",
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
    "zeus_total_energy_j",
    "zeus_total_time_s",
    "zeus_avg_power_w",
    "zeus_train_step_energy_j",
    "zeus_energy_per_token_j",
]


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
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
    return parser.parse_args()


def load_configurations(path: str, fmt: str) -> list[dict]:
    """
    Load experiment configurations from a JSON or YAML file.

    Args:
        path: File path.
        fmt: 'json' or 'yaml'.

    Returns:
        A list of configuration dictionaries.
    """
    text = Path(path).read_text()
    if fmt == 'yaml':
        # YAML may contain multiple documents or a single list
        loaded = list(yaml.safe_load_all(text))
        # Flatten if outer list-of-lists
        if len(loaded) == 1 and isinstance(loaded[0], list):
            return loaded[0]
        return loaded
    else:
        return json.loads(text)


RUN_NAME_VAR = "${RUN_NAME}"


def expand_range(val):
    """Expand dicts with 'range' into a list of values."""
    if isinstance(val, dict) and 'range' in val:
        r = val['range']
        start, end = r['start'], r['end']
        step = r.get('step', 1 if isinstance(start, int) else 0.1)
        if isinstance(start, int):
            return list(range(start, end + 1, step))
        count = int(round((end - start) / step)) + 1
        return [start + i * step for i in range(count)]
    return val


def _substitute_run_name(obj, run_name: str):
    """Recursively substitute the run name placeholder inside ``obj``."""
    if isinstance(obj, str):
        return obj.replace(RUN_NAME_VAR, run_name)
    if isinstance(obj, list):
        return [_substitute_run_name(o, run_name) for o in obj]
    if isinstance(obj, dict):
        return {k: _substitute_run_name(v, run_name) for k, v in obj.items()}
    return obj


def _ensure_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


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


NON_EXPANDING_KEYS = {"sequential_runs"}


def _collect_param_keys(cfg: dict) -> set[str]:
    keys: set[str] = set()
    for key in cfg:
        if key in {'parameter_groups', '_named_group_fragments', '_named_group_param_keys'}:
            continue
        if key in NON_EXPANDING_KEYS:
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
        base = {}
        for key, value in cfg.items():
            if isinstance(value, dict) and 'conditions' in value:
                continue
            if key == 'parameter_groups':
                continue
            if key in NON_EXPANDING_KEYS:
                base[key] = [value]
                continue
            normalized = expand_range(value) if isinstance(value, dict) and 'range' in value else value
            base[key] = normalized
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

    base_metric_keys = [
        "best_val_loss",
        "best_val_iter",
        "best_val_tokens",
        "num_params",
        "better_than_chance",
        "btc_per_param",
        "peak_torch_allocated_mb",
        "peak_torch_reserved_mb",
        "peak_process_gpu_mb",
        "iter_latency_avg",
        "zeus_best_train_step_energy_j",
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
        float,
        float,
        float,
    ]

    if len(base_metric_keys) != len(casts):
        raise ValueError(
            f"Metric schema mismatch: {len(base_metric_keys)} keys vs {len(casts)} casts."
        )
    if len(parts) < len(base_metric_keys):
        raise ValueError(
            f"Expected at least {len(base_metric_keys)} metrics in {path}, got {len(parts)}."
        )

    metrics: dict[str, float] = {}
    for key, typ, value in zip(base_metric_keys, casts, parts):
        metrics[key] = float("nan") if value == "" else typ(value)

    zeus_summary_path = Path(out_dir) / ZEUS_SUMMARY_FILENAME
    if zeus_summary_path.exists():
        summary = json.loads(zeus_summary_path.read_text())
        metrics["zeus_total_energy_j"] = float(summary.get("zeus_total_energy_j", float("nan")))
        metrics["zeus_total_time_s"] = float(summary.get("zeus_total_time_s", float("nan")))
        metrics["zeus_avg_power_w"] = float(summary.get("zeus_avg_power_w", float("nan")))
        metrics["zeus_train_step_energy_j"] = float(summary.get("zeus_train_step_energy_j", float("nan")))
        metrics["zeus_energy_per_token_j"] = float(summary.get("zeus_energy_per_token_j", float("nan")))
    else:
        metrics["zeus_total_energy_j"] = float("nan")
        metrics["zeus_total_time_s"] = float("nan")
        metrics["zeus_avg_power_w"] = float("nan")
        metrics["zeus_train_step_energy_j"] = float("nan")
        metrics["zeus_energy_per_token_j"] = float("nan")

    return {k: metrics.get(k, float("nan")) for k in METRIC_KEYS}


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


def build_command(combo: dict, script: str = "train.py") -> list[str]:
    """
    Construct the command-line invocation for a training script.
    """
    cmd = ['python3', script]
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


def _normalize_sequential_runs(sequential_runs: object) -> list[dict]:
    if sequential_runs is None:
        return []
    if isinstance(sequential_runs, dict):
        return [sequential_runs]
    if isinstance(sequential_runs, list):
        return sequential_runs
    raise TypeError("'sequential_runs' must be a list of stage mappings or a single mapping")


def _prepare_stage_combo(base_combo: dict, stage_args: dict) -> dict:
    combined = deepcopy(base_combo)
    for key, value in stage_args.items():
        combined[key] = deepcopy(value)
    return combined


def _apply_resume_defaults(
    stage_script: str,
    stage_combo: dict,
    prev_out_dir: str,
    prev_ckpt_name: str,
    stage_idx: int,
) -> dict:
    if stage_script == "train_recurrent.py":
        resume_ckpt = stage_combo.get("resume_ckpt")
        if resume_ckpt is None:
            stage_combo["resume_ckpt"] = os.path.join(prev_out_dir, prev_ckpt_name)
        return stage_combo

    if stage_script in {"train.py", "train_mezo.py"}:
        stage_combo.setdefault("init_from", "resume")
        stage_combo.setdefault("init_from_ckpt", prev_ckpt_name)
        return stage_combo

    if stage_idx > 0:
        print(
            f"[yellow]Warning:[/] resume defaults are only supported for train.py, "
            f"train_mezo.py, and train_recurrent.py (got {stage_script})."
        )
    return stage_combo


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
    sequential_runs = _normalize_sequential_runs(combo.pop('sequential_runs', None))
    log_combo = deepcopy(combo)
    if sequential_runs:
        log_combo['sequential_runs'] = deepcopy(sequential_runs)
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
    out_dir = os.path.join(args.output_dir, out_dir_name)
    combo['out_dir'] = out_dir

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

    if not sequential_runs:
        sequential_runs = [{"script": "train.py"}]

    prev_out_dir = out_dir
    prev_ckpt_name = combo.get("init_from_ckpt", "ckpt.pt")
    stage_outputs: dict[str, str] = {}

    for idx, stage in enumerate(sequential_runs, 1):
        if not isinstance(stage, dict):
            raise TypeError("Each sequential run stage must be a mapping")
        stage_script = stage.get("script", "train.py")
        stage_args = stage.get("args", {})
        if not isinstance(stage_args, dict):
            raise TypeError("Stage 'args' must be a mapping of CLI arguments")
        stage_name = stage.get("name", f"stage_{idx}")

        stage_combo = _prepare_stage_combo(combo, stage_args)
        stage_combo['out_dir'] = out_dir
        stage_combo.setdefault('tensorboard_run_name', run_name)

        input_ckpt = stage.get("input_ckpt")
        if input_ckpt:
            if input_ckpt in stage_outputs:
                prev_ckpt_name = stage_outputs[input_ckpt]
            elif input_ckpt in stage_outputs.values():
                prev_ckpt_name = input_ckpt
            elif not str(input_ckpt).endswith(".pt"):
                raise ValueError(
                    f"Unknown input_ckpt '{input_ckpt}'. Expected a prior stage name or .pt filename."
                )
            else:
                prev_ckpt_name = input_ckpt

        resume = stage.get("resume", idx > 1)
        if resume:
            stage_combo = _apply_resume_defaults(
                stage_script,
                stage_combo,
                prev_out_dir=prev_out_dir,
                prev_ckpt_name=prev_ckpt_name,
                stage_idx=idx,
            )

        stage_combo = _substitute_run_name(stage_combo, run_name)
        cmd = build_command(stage_combo, script=stage_script)
        print(f"Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print(f"[red]Process exited with error for run:[/] {run_name}")
            break

        prev_out_dir = out_dir
        if stage_script == "train_recurrent.py":
            prev_ckpt_name = stage_combo.get("recurrent_output_ckpt", prev_ckpt_name)
        elif stage_script == "train_mezo.py":
            prev_ckpt_name = stage_combo.get("mezo_output_ckpt", prev_ckpt_name)
        else:
            prev_ckpt_name = stage_combo.get("output_ckpt", prev_ckpt_name)
        stage_outputs[stage_name] = prev_ckpt_name

    # Read metrics (use existing or nan on failure)
    try:
        metrics = read_metrics(str(combo['out_dir']))
    except Exception:
        metrics = {k: float("nan") for k in METRIC_KEYS}

    log_combo['out_dir'] = out_dir
    log_combo['tensorboard_run_name'] = run_name
    append_log(log_file, run_name, log_combo, metrics)


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
