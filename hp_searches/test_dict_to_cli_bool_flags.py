#!/usr/bin/env python3
"""Lightweight checks for hyperparameter-search boolean CLI serialization."""

import argparse
import ast
import sys
from pathlib import Path
from typing import Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from hyperparam_search import dict_to_cli


def _boolean_optional_defaults() -> Dict[str, Optional[bool]]:
    """Read train_args.py without importing torch-dependent training modules."""
    tree = ast.parse((REPO_ROOT / "train_args.py").read_text())
    defaults: Dict[str, Optional[bool]] = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        option = node.args[0].value
        if not isinstance(option, str) or not option.startswith("--"):
            continue

        uses_boolean_optional_action = False
        default: bool | None = None
        for keyword in node.keywords:
            if keyword.arg == "action" and isinstance(keyword.value, ast.Attribute):
                uses_boolean_optional_action = keyword.value.attr == "BooleanOptionalAction"
            elif keyword.arg == "default" and isinstance(keyword.value, ast.Constant):
                if isinstance(keyword.value.value, bool) or keyword.value.value is None:
                    default = keyword.value.value

        if uses_boolean_optional_action:
            defaults[option.removeprefix("--")] = default

    return defaults


def _parser_for(flags: Dict[str, Optional[bool]]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    for flag, default in flags.items():
        parser.add_argument(
            f"--{flag}", default=default, action=argparse.BooleanOptionalAction
        )
    return parser


def test_false_booleans_emit_no_flags() -> None:
    default_true_flags = [
        "lr_decay_match_max_iters",
        "csv_log",
        "tensorboard_log",
        "tensorboard_graph",
        "log_rankme",
        "log_areq",
        "print_model_info",
    ]
    cfg = {"compile": False, **{flag: False for flag in default_true_flags}}

    cli = dict_to_cli(cfg)

    expected = ["--no-compile"] + [f"--no-{flag}" for flag in default_true_flags]
    assert cli == expected, cli

    train_arg_defaults = _boolean_optional_defaults()
    assert train_arg_defaults["compile"] is False
    for flag in default_true_flags:
        assert train_arg_defaults[flag] is True, flag

    args = _parser_for({flag: train_arg_defaults[flag] for flag in cfg}).parse_args(cli)
    assert args.compile is False
    for flag in default_true_flags:
        assert getattr(args, flag) is False, flag


def test_true_booleans_emit_positive_flags() -> None:
    cli = dict_to_cli({"compile": True, "csv_log": True})

    assert cli == ["--compile", "--csv_log"], cli

    train_arg_defaults = _boolean_optional_defaults()
    args = _parser_for(
        {"compile": train_arg_defaults["compile"], "csv_log": train_arg_defaults["csv_log"]}
    ).parse_args(cli)
    assert args.compile is True
    assert args.csv_log is True


if __name__ == "__main__":
    test_false_booleans_emit_no_flags()
    test_true_booleans_emit_positive_flags()
    print("dict_to_cli boolean flag checks passed")
