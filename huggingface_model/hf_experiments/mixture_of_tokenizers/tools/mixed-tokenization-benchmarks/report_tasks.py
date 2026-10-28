#!/usr/bin/env python3
"""Validate and compare four fixed checkpoints on identical benchmark items.

Task confidence intervals resample paired items, conditional on the supplied
trained models. They do not estimate uncertainty over future training seeds.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import score_tasks as task_scorer

METRICS = ("accuracy", "option_log_loss", "correct_vs_best_incorrect_margin", "byte_normalized_accuracy")
IDENTITY = ("id", "task", "language", "split", "answer_index", "num_choices", "chance_accuracy", "prompt_utf8_bytes", "choices_utf8_bytes", "example_sha256")
ARCHITECTURE = ("hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size", "max_position_embeddings")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual: Any, expected: float, name: str) -> None:
    require(isinstance(actual, (int, float)) and math.isfinite(actual), f"{name}: expected a finite number")
    require(math.isclose(actual, expected, abs_tol=1e-7, rel_tol=1e-7), f"{name}: metric arithmetic mismatch ({actual} != {expected})")


def choice_metrics(scores: list[float], gold: int) -> dict[str, Any]:
    return task_scorer.metrics_from_scores(scores, gold)


def validate_record(record: dict[str, Any]) -> None:
    require(all(k in record for k in IDENTITY), "record is missing identity fields")
    require(isinstance(record["id"], str) and bool(record["id"]), "record id must be a nonempty string")
    require(record["language"] in ("zh", "ko", "ja", "en"), "unknown language")
    n = record["num_choices"]
    require(type(n) is int and n >= 2, "num_choices must be at least two")
    close(record["chance_accuracy"], 1 / n, "chance_accuracy")
    lengths = record["choices_utf8_bytes"]
    require(len(lengths) == n and all(type(x) is int and x > 0 for x in lengths), "invalid choices_utf8_bytes")
    require(set(record["routes"]) == {"native", "byte"}, "each record must contain native and byte routes")
    for route, item in record["routes"].items():
        raw = item["sum_log_probs"]
        require(len(raw) == n, "choice score count differs from num_choices")
        normalized = [x / size for x, size in zip(raw, lengths)]
        require(len(item["sum_log_probs_per_utf8_byte"]) == n, "normalized choice score count mismatch")
        for actual, expected in zip(item["sum_log_probs_per_utf8_byte"], normalized):
            close(actual, expected, f"{record['id']}/{route}/byte score")
        for mode, scores in (("raw", raw), ("byte_normalized", normalized)):
            expected = choice_metrics(scores, record["answer_index"])
            for key, value in expected.items():
                if key in ("prediction", "tied_predictions", "tie_count"):
                    require(item[mode][key] == value, f"{record['id']}/{route}/{mode}/{key} mismatch")
                else:
                    close(item[mode][key], value, f"{record['id']}/{route}/{mode}/{key}")


def checkpoint_signature(result: dict[str, Any]) -> str:
    files = result.get("provenance", {}).get("checkpoint_files_sha256", {})
    require(isinstance(files, dict) and bool(files), "missing checkpoint_files_sha256 provenance")
    weights = {Path(name).name: value for name, value in files.items()
               if name.endswith((".safetensors", ".bin", ".pt", ".pth"))}
    require(bool(weights), "checkpoint provenance has no weight hashes")
    return canonical(weights)


def load_results(paths: list[str | Path], seeds=(1337, 1338), sizes=(10000, 20000)) -> dict[tuple[int, int], dict[str, Any]]:
    expected = {(size, seed) for size in sizes for seed in seeds}
    require(len(seeds) == len(set(seeds)) == 2 and len(sizes) == len(set(sizes)) == 2, "require two distinct seeds and two distinct lexical sizes")
    require(len(paths) == len(expected), "require exactly four task result files")
    loaded: dict[tuple[int, int], dict[str, Any]] = {}
    model_paths, signatures = set(), set()
    baseline = None
    for path in paths:
        path = Path(path)
        result = json.loads(path.read_text(encoding="utf-8"))
        require(result.get("schema_version") == 1 and result.get("result_type") == "mixed_tokenization_multiple_choice", f"{path}: unsupported task result schema")
        key = (result.get("lexical_size"), result.get("training_seed"))
        require(key in expected, f"{path}: unexpected lexical size/seed {key}")
        require(key not in loaded, f"duplicate lexical size/seed configuration: {key}")
        model = str(Path(result["model"]).resolve())
        signature = checkpoint_signature(result)
        require(model not in model_paths, "duplicate checkpoint path across configurations")
        require(signature not in signatures, "duplicate checkpoint weights across configurations")
        model_paths.add(model)
        signatures.add(signature)
        for field in ("tasks_sha256", "manifest_sha256", "subset_sha256"):
            require(bool(re.fullmatch(r"[0-9a-f]{64}", result.get(field, ""))), f"invalid or missing {field}")
        require(bool(result.get("protocol")), "missing scoring protocol")
        context = result["protocol"].get("eligibility_context")
        require(type(context) is int and context >= 2 and result["protocol"] == task_scorer.protocol(context), "unsupported scoring protocol")
        require(result["model_config"].get("max_position_embeddings") == context, "model context differs from scoring protocol")
        records = result["records"]
        require(bool(records), "no eligible benchmark items")
        for record in records:
            validate_record(record)
        ids = [record["id"] for record in records]
        skipped = result["skipped"]
        skipped_ids = [record["id"] for record in skipped]
        require(len(ids) == len(set(ids)), "duplicate eligible item IDs")
        require(len(skipped_ids) == len(set(skipped_ids)), "duplicate skipped item IDs")
        require(not set(ids).intersection(skipped_ids), "item is both eligible and skipped")
        require(result["eligible_ids"] == ids and result["skipped_ids"] == skipped_ids, "record IDs disagree with eligible/skipped lists")
        require(result["subset_sha256"] == task_scorer.canonical_hash([[r["id"], r["example_sha256"]] for r in records]), "eligible IDs/source hashes disagree with subset_sha256")
        for record in records:
            require(1 + record["prompt_utf8_bytes"] + max(record["choices_utf8_bytes"]) <= context, "eligible item exceeds byte context")
        for record in skipped:
            size = 1 + record["prompt_utf8_bytes"] + max(record["choices_utf8_bytes"])
            require(size > context and record["max_sequence_bytes_with_bos"] == size and record["reason"] == "byte_context_exceeded", "invalid skipped item eligibility/reason")
        manifest = result["manifest"]
        require(manifest.get("schema_version") == 1 and manifest.get("tasks_sha256") == result["tasks_sha256"], "manifest/task digest mismatch")
        require(manifest.get("record_count") == len(records) + len(skipped), "manifest record count mismatch")
        require(manifest.get("split_set") in ("dev", "final") and bool(manifest.get("template_version")), "manifest split/template identity missing")
        aggregates = task_scorer.aggregate(records, skipped)
        require(result.get("aggregates") == aggregates, "stored aggregates disagree with item metrics")
        declared = manifest.get("tasks", [])
        require(len(declared) == len(aggregates["by_task"]), "manifest task group count mismatch")
        require(len({item["task"] for item in declared}) == len(declared), "duplicate task group in manifest")
        for item in declared:
            group = [row for row in records + skipped if row["task"] == item["task"]]
            require(len(group) == item["count"] and all(row["language"] == item["language"] and row["split"] == item["split"] and row["num_choices"] == item["num_choices"] for row in group), "manifest task identity/count mismatch")
            close(item["chance_accuracy"], 1 / item["num_choices"], "manifest chance_accuracy")
        if baseline is not None:
            for field in ("tasks_sha256", "manifest_sha256", "subset_sha256", "protocol", "manifest", "eligible_ids", "skipped_ids"):
                require(result.get(field) == baseline.get(field), f"incompatible {field}: cannot pair benchmark results")
            for field in ARCHITECTURE:
                require(result.get("model_config", {}).get(field) == baseline.get("model_config", {}).get(field), f"architecture differs: {field}")
            for field in ("experiment_sources_sha256", "scorer_sha256", "precision_policy", "packages", "device", "threads"):
                require(result["provenance"].get(field) == baseline["provenance"].get(field), f"scoring provenance differs: {field}")
            for left, right in zip(records, baseline["records"]):
                require(all(left[field] == right[field] for field in IDENTITY), f"item identity mismatch: {left['id']}")
            require(skipped == baseline["skipped"], "skipped item identities/reasons differ")
        else:
            baseline = result
        result["_input_path"] = str(path.resolve())
        result["_input_sha256"] = file_sha256(path)
        loaded[key] = result
    require(set(loaded) == expected, "missing expected lexical size/seed configurations")
    return loaded


def metric_vector(record: dict[str, Any], route: str) -> list[float]:
    scores = record["routes"][route]
    return [scores["raw"]["correct_fraction"], scores["raw"]["option_log_loss"],
            scores["raw"]["correct_vs_best_incorrect_margin"], scores["byte_normalized"]["correct_fraction"]]


def paired_bootstrap(left: list[list[float]], right: list[list[float]], *, samples: int, seed: int) -> dict[str, Any]:
    """Use one item-index draw for both models and all metrics in each replicate."""
    import numpy as np
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    require(a.shape == b.shape and a.ndim == 2 and a.shape[0] > 0 and a.shape[1] == len(METRICS), "paired bootstrap requires matching nonempty item matrices")
    require(samples >= 100, "at least 100 bootstrap samples required")
    delta = a - b
    rng = np.random.default_rng(seed)
    replicates = []
    for start in range(0, samples, 128):
        indices = rng.integers(0, len(delta), size=(min(128, samples - start), len(delta)))
        replicates.append(delta[indices].mean(axis=1))
    bounds = np.quantile(np.concatenate(replicates), [0.025, 0.975], axis=0)
    return {name: {"difference": float(delta[:, i].mean()), "ci95": [float(bounds[0, i]), float(bounds[1, i])]}
            for i, name in enumerate(METRICS)}


def task_report(results: dict[tuple[int, int], dict[str, Any]], *, samples: int, bootstrap_seed: int) -> dict[str, Any]:
    first = next(iter(results.values()))
    groups = sorted({(row["task"], row["language"]) for row in first["records"] + first["skipped"]})
    score_rows, comparisons, coverage = [], [], []
    sizes = sorted({key[0] for key in results})
    seeds = sorted({key[1] for key in results})
    for task, language in groups:
        selected = [i for i, row in enumerate(first["records"]) if (row["task"], row["language"]) == (task, language)]
        skipped = [r for r in first["skipped"] if (r["task"], r["language"]) == (task, language)]
        source_count = len(selected) + len(skipped)
        chance = sum(first["records"][i]["chance_accuracy"] for i in selected) / len(selected) if selected else None
        coverage.append({"task": task, "language": language, "eligible_count": len(selected), "skipped_count": len(skipped), "source_count": source_count,
                         "coverage": len(selected) / source_count, "chance_accuracy": chance,
                         "eligible_ids": [first["records"][i]["id"] for i in selected], "skipped_ids": [r["id"] for r in skipped]})
        if not selected:
            continue
        matrices = {}
        for (size, seed), result in sorted(results.items()):
            for route in ("native", "byte"):
                matrix = [metric_vector(result["records"][i], route) for i in selected]
                matrices[size, seed, route] = matrix
                row = {"kind": "task", "task": task, "language": language, "lexical_size": size, "training_seed": seed, "route": route,
                       "count": len(selected), "skipped_count": len(skipped), "source_count": source_count, "coverage": len(selected) / source_count,
                       "chance_accuracy": chance, "exact_tie_count": sum(result["records"][i]["routes"][route]["raw"]["tie_count"] > 1 for i in selected)}
                row.update({name: sum(values[m] for values in matrix) / len(matrix) for m, name in enumerate(METRICS)})
                score_rows.append(row)
        pairs = []
        for seed in seeds:
            pairs += [("native_minus_byte", seed, str(size), matrices[size, seed, "native"], matrices[size, seed, "byte"]) for size in sizes]
            pairs += [("smaller_k_minus_larger_k", seed, route, matrices[sizes[0], seed, route], matrices[sizes[1], seed, route]) for route in ("native", "byte")]
        # Average the fixed models' item-level deltas before resampling: one item
        # draw is shared by seeds, preserving their document/item dependence.
        def seed_average(size, route):
            return [[sum(matrices[size, seed, route][i][m] for seed in seeds) / len(seeds)
                     for m in range(len(METRICS))] for i in range(len(selected))]
        pairs += [("native_minus_byte", "fixed_seed_mean", str(size), seed_average(size, "native"), seed_average(size, "byte")) for size in sizes]
        pairs += [("smaller_k_minus_larger_k", "fixed_seed_mean", route, seed_average(sizes[0], route), seed_average(sizes[1], route)) for route in ("native", "byte")]
        for contrast, seed, subgroup, left, right in pairs:
            comparisons.append({"task": task, "language": language, "contrast": contrast, "training_seed": seed, "subgroup": subgroup,
                                "count": len(selected), "metrics": paired_bootstrap(left, right, samples=samples, seed=bootstrap_seed)})
    return {"scores": score_rows, "coverage": coverage, "comparisons": comparisons}


def infer_lm_seed(path: Path, result: dict[str, Any]) -> int:
    candidates = []
    if "training_seed" in result:
        candidates.append(int(result["training_seed"]))
    sibling = path.parent / "run_config.json"
    if sibling.exists():
        config = json.loads(sibling.read_text(encoding="utf-8"))
        training = config.get("effective_training", config.get("config", {}).get("training", {}))
        if "seed" in training:
            candidates.append(int(training["seed"]))
    candidates += [int(seed) for seed in re.findall(r"seed[-_]?(\d+)", str(path))]
    require(bool(candidates) and len(set(candidates)) == 1, f"{path}: missing or conflicting LM training seed")
    return candidates[0]


def load_lm_results(paths: list[str | Path], task_results: dict[tuple[int, int], dict[str, Any]]) -> list[dict[str, Any]]:
    rows, seen, references = [], set(), {}
    for filename in paths:
        path = Path(filename)
        result = json.loads(path.read_text(encoding="utf-8"))
        key = (int(result["lexical_size"]), infer_lm_seed(path, result))
        require(key in task_results, f"{path}: LM model configuration has no matching task result")
        require(Path(result["model"]).resolve() == Path(task_results[key]["model"]).resolve(), f"{path}: LM/task checkpoint paths differ")
        if result.get("evaluation_kind") == "heldout_corpus_lm":
            require(bool(result.get("provenance")), f"{path}: new LM wrapper result is missing provenance")
        if result.get("provenance"):
            for field in ("checkpoint_files_sha256", "experiment_sources_sha256", "precision_policy", "packages", "device", "threads"):
                require(result["provenance"].get(field) == task_results[key]["provenance"].get(field), f"{path}: LM/task {field} differs")
        mode = result["context_mode"]
        require(mode in ("bytes", "token"), "unknown LM context mode")
        require((mode, *key) not in seen, "duplicate LM context/model configuration")
        seen.add((mode, *key))
        identity = {field: result[field] for field in ("metric", "context_tokens", "context_policy", "eos_policy", "split", "evaluated_text_sha256")}
        require(identity["metric"] == "canonical_tokenization_bits_per_utf8_byte", "unsupported LM metric")
        if mode in references:
            require(identity == references[mode], "LM text hashes/context/split differ across checkpoints")
        else:
            references[mode] = identity
        require(set(result["languages"]) == {"zh", "ko", "ja", "en"}, "LM results must contain all four languages")
        for language, routes in result["languages"].items():
            require(set(routes) == {language, "byte"}, "LM routes must be the native language and byte")
            require(routes[language]["documents"] == routes["byte"]["documents"] and routes[language]["source_bytes"] == routes["byte"]["source_bytes"], "LM route document/byte counts differ")
            for route, score in routes.items():
                require(score["source_bytes"] > 0 and score["documents"] > 0, "empty LM data")
                close(score["bpb"], score["text_nll"] / score["source_bytes"] / math.log(2), "LM BPB")
                rows.append({"kind": "lm", "task": "held_out_language_modeling", "language": language, "lexical_size": key[0], "training_seed": key[1],
                             "route": "byte" if route == "byte" else "native", "context_mode": mode, "split": result["split"], "count": score["documents"],
                             "source_bytes": score["source_bytes"], "bpb": score["bpb"], "byte_perplexity": 2 ** score["bpb"],
                             "overlap_audit_status": result.get("overlap_audit", {}).get("status", "not_recorded"),
                             "text_sha256": result["evaluated_text_sha256"][language], "input_path": str(path.resolve()), "input_sha256": file_sha256(path)})
    for mode in references:
        require({(size, seed) for m, size, seed in seen if m == mode} == set(task_results), "LM context mode must include all four configurations")
    for mode in references:
        for language in ("zh", "ko", "ja", "en"):
            comparable = [row for row in rows if row["context_mode"] == mode and row["language"] == language]
            require(len({(row["count"], row["source_bytes"]) for row in comparable}) == 1, "LM document/byte counts differ across checkpoints")
    return rows


def markdown(summary: dict[str, Any]) -> str:
    lines = ["# Tokenizer-route benchmark comparison", "", "The byte route shares its trained backbone with the native routes; it is not an independently trained byte-only model.", "",
             "Task scores use the pinned custom adapters and continuation-scoring protocol described in this report; they are not official benchmark or leaderboard scores.", "",
             "Task intervals are paired item-bootstrap 95% intervals conditional on these four trained models. Fixed-seed means use the same item resample across both seeds; they do not measure training-seed uncertainty. Accuracy ties receive fractional credit. Choice log loss is a softmax over candidate sequence scores, not text perplexity.", "",
             "## Task coverage", "", "| Task | Language | Evaluated / source | Coverage | Random chance |", "|---|---|---:|---:|---:|"]
    for row in summary["coverage"]:
        chance = f"{100 * row['chance_accuracy']:.1f}%" if row["chance_accuracy"] is not None else "—"
        lines.append(f"| {row['task']} | {row['language']} | {row['eligible_count']} / {row['source_count']} | {100 * row['coverage']:.1f}% | {chance} |")
    lines += ["", "All checkpoints use the identical eligible question set. Excluded questions exceed the common byte-route context limit; coverage refers to the selected benchmark split, not the complete dataset.", "", "## Task scores", "", "| Task | K | Seed | Route | Accuracy | Byte-normalized accuracy | Gold option log loss ↓ | Gold − best other margin ↑ |", "|---|---:|---:|---|---:|---:|---:|---:|"]
    for row in summary["scores"]:
        lines.append(f"| {row['task']} | {row['lexical_size']} | {row['training_seed']} | {row['route']} | {100 * row['accuracy']:.2f}% | {100 * row['byte_normalized_accuracy']:.2f}% | {row['option_log_loss']:.4f} | {row['correct_vs_best_incorrect_margin']:.4f} |")
    lines += ["", "Raw continuation log probability is primary. Byte-normalized accuracy is a declared secondary score. Log loss and margin use raw scores; margins are in nats and depend on continuation length.", "", "## Paired accuracy differences", "", "Positive values favor the first route or smaller K. Values and intervals are percentage points. JSON also contains intervals for choice log loss, margin, and secondary accuracy.", "", "| Task | Contrast | K / route | Seed | Difference [95% CI] |", "|---|---|---|---|---:|"]
    for row in summary["comparisons"]:
        metric = row["metrics"]["accuracy"]
        lo, hi = metric["ci95"]
        lines.append(f"| {row['task']} | {row['contrast']} | {row['subgroup']} | {row['training_seed']} | {100 * metric['difference']:+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}] |")
    if summary["lm_scores"]:
        lines += ["", "## Held-out language modeling", "", "Compare routes and K within each language and context policy. Raw BPB values are not a ranking of language difficulty. Byte perplexity = 2^BPB; ordinary token perplexity is not comparable across these vocabularies. EOS loss is excluded. These aggregate inputs do not provide new document confidence intervals.", "", "| Language | Context | Split | K | Seed | Route | Documents | BPB ↓ | Byte perplexity ↓ |", "|---|---|---|---:|---:|---|---:|---:|---:|"]
        for row in sorted(summary["lm_scores"], key=lambda r: (r["language"], r["context_mode"], r["lexical_size"], r["training_seed"], r["route"])):
            lines.append(f"| {row['language']} | {row['context_mode']} | {row['split']} | {row['lexical_size']} | {row['training_seed']} | {row['route']} | {row['count']} | {row['bpb']:.5f} | {row['byte_perplexity']:.5f} |")
    lines += ["", "## Provenance", "", f"Tasks SHA256: `{summary['tasks_sha256']}`", "", f"Manifest SHA256: `{summary['manifest_sha256']}`", "", f"Eligible subset SHA256: `{summary['subset_sha256']}`", "", f"Bootstrap: {summary['bootstrap']['samples']} paired resamples, seed {summary['bootstrap']['seed']}. No claim about a separately trained byte-only model or causality from tokenizer choice alone follows from this comparison.", ""]
    return "\n".join(lines)


def create_report(paths: list[str | Path], output_dir: str | Path, *, lm_paths=(), samples=5000, bootstrap_seed=20261027, seeds=(1337, 1338), sizes=(10000, 20000)) -> dict[str, Any]:
    results = load_results(paths, seeds=seeds, sizes=sizes)
    first = next(iter(results.values()))
    summary = {"schema_version": 1, "result_type": "mixed_tokenization_benchmark_report",
               **{key: first[key] for key in ("tasks_sha256", "manifest_sha256", "subset_sha256", "protocol", "manifest")},
               "bootstrap": {"samples": samples, "seed": bootstrap_seed, "unit": "paired benchmark item", "interval": "percentile_95", "scope": "conditional on the supplied trained models; not training-seed uncertainty"},
               "inputs": [{"lexical_size": size, "training_seed": seed, "model": result["model"], "path": result["_input_path"], "sha256": result["_input_sha256"], "provenance": result["provenance"]} for (size, seed), result in sorted(results.items())],
               **task_report(results, samples=samples, bootstrap_seed=bootstrap_seed),
               "lm_scores": load_lm_results(list(lm_paths), results)}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "report.md").write_text(markdown(summary), encoding="utf-8")
    rows = summary["scores"] + summary["lm_scores"]
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", required=True, help="Four score_tasks.py JSON outputs")
    parser.add_argument("--lm-results", nargs="*", default=[], help="Optional four-per-context original evaluate.py LM JSON outputs")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", "--bootstrap-repetitions", dest="bootstrap_samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261027)
    parser.add_argument("--seeds", nargs=2, type=int, default=[1337, 1338])
    parser.add_argument("--lexical-sizes", "--sizes", dest="lexical_sizes", nargs=2, type=int, default=[10000, 20000])
    args = parser.parse_args()
    try:
        summary = create_report(args.results, args.output_dir, lm_paths=args.lm_results, samples=args.bootstrap_samples,
                                bootstrap_seed=args.bootstrap_seed, seeds=args.seeds, sizes=args.lexical_sizes)
    except (ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(f"Wrote {Path(args.output_dir).resolve()}: {len(summary['scores'])} task score rows, {len(summary['lm_scores'])} LM rows")


if __name__ == "__main__":
    main()
