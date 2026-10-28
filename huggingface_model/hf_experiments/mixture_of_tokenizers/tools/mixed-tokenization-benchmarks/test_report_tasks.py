"""Tests of comparison validity, pairing, arithmetic, and aggregate LM import."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import report_tasks as report


def fixture_record(item_id="en/0", scores=(-1.0, -2.0), gold=0):
    lengths = [2, 4]
    normalized = [s / n for s, n in zip(scores, lengths)]
    route = {"sum_log_probs": list(scores), "sum_log_probs_per_utf8_byte": normalized,
             "prompt_tokens": 3, "choice_tokens": [2, 4],
             "raw": report.choice_metrics(list(scores), gold), "byte_normalized": report.choice_metrics(normalized, gold)}
    return {"id": item_id, "task": "piqa_en", "language": "en", "split": "validation", "answer_index": gold,
            "num_choices": 2, "chance_accuracy": 0.5, "prompt_utf8_bytes": 3, "choices_utf8_bytes": lengths,
            "example_sha256": "e" * 64, "routes": {"native": copy.deepcopy(route), "byte": copy.deepcopy(route)}}


def fixture_result(size, seed):
    records = [fixture_record(), fixture_record("en/1", (-2.0, -1.0), 1)]
    manifest = {"schema_version": 1, "tasks_sha256": "a" * 64, "record_count": 2, "split_set": "final", "template_version": "v1",
                "tasks": [{"task": "piqa_en", "language": "en", "split": "validation", "count": 2, "num_choices": 2, "chance_accuracy": 0.5}]}
    return {"schema_version": 1, "result_type": "mixed_tokenization_multiple_choice", "model": f"/models/seed{seed}/k{size}/checkpoint-best",
            "lexical_size": size, "training_seed": seed, "model_config": {"hidden_size": 384, "max_position_embeddings": 512},
            "protocol": report.task_scorer.protocol(512), "manifest": manifest,
            "tasks_sha256": "a" * 64, "manifest_sha256": "b" * 64,
            "subset_sha256": report.task_scorer.canonical_hash([[r["id"], r["example_sha256"]] for r in records]),
            "provenance": {"checkpoint_files_sha256": {"model.safetensors": f"{size:032d}{seed:032d}"},
                           "packages": {"torch": "test-version"}, "device": "cpu", "threads": 1},
            "records": records, "skipped": [], "aggregates": report.task_scorer.aggregate(records, []),
            "eligible_ids": ["en/0", "en/1"], "skipped_ids": []}


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.paths = []
        for size in (10000, 20000):
            for seed in (1337, 1338):
                path = self.root / f"seed{seed}" / f"k{size}" / "tasks.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(fixture_result(size, seed)))
                self.paths.append(path)

    def tearDown(self):
        self.temp.cleanup()

    def mutate(self, index, mutate):
        result = json.loads(self.paths[index].read_text())
        mutate(result)
        self.paths[index].write_text(json.dumps(result))

    def test_exact_ties_and_option_loss(self):
        metrics = report.choice_metrics([-3.0, -3.0], 1)
        self.assertEqual(metrics["correct_fraction"], 0.5)
        self.assertEqual(metrics["tied_predictions"], [0, 1])
        self.assertAlmostEqual(metrics["option_log_loss"], math.log(2))
        self.assertEqual(metrics["correct_vs_best_incorrect_margin"], 0)

    def test_numerically_stable_log_loss(self):
        metrics = report.choice_metrics([-10000.0, -10001.0], 0)
        self.assertAlmostEqual(metrics["option_log_loss"], math.log1p(math.exp(-1)))
        self.assertEqual(metrics["correct_vs_best_incorrect_margin"], 1)

    def test_corrupt_reported_accuracy_is_rejected(self):
        self.mutate(0, lambda d: d["records"][0]["routes"]["native"]["raw"].update(correct_fraction=0.0))
        with self.assertRaisesRegex(ValueError, "arithmetic mismatch"):
            report.load_results(self.paths)

    def test_different_eligible_ids_cannot_be_paired_even_if_hash_claims_match(self):
        def change(d):
            d["records"][0]["id"] = "en/different"
            d["eligible_ids"][0] = "en/different"
        self.mutate(1, change)
        with self.assertRaisesRegex(ValueError, "eligible IDs"):
            report.load_results(self.paths)

    def test_same_ids_with_changed_answer_or_source_are_rejected(self):
        self.mutate(1, lambda d: d["records"][0].update(example_sha256="f" * 64))
        with self.assertRaisesRegex(ValueError, "source hashes"):
            report.load_results(self.paths)

    def test_protocol_and_dataset_identity_mismatch_are_rejected(self):
        self.mutate(1, lambda d: d["protocol"].update(context=1024))
        with self.assertRaisesRegex(ValueError, "protocol"):
            report.load_results(self.paths)

    def test_changed_runtime_environment_is_rejected(self):
        self.mutate(1, lambda d: d["provenance"]["packages"].update(torch="different-version"))
        with self.assertRaisesRegex(ValueError, "packages"):
            report.load_results(self.paths)

    def test_relabelled_duplicate_checkpoint_is_rejected(self):
        original = json.loads(self.paths[0].read_text())["provenance"]
        self.mutate(1, lambda d: d.update(provenance=original))
        with self.assertRaisesRegex(ValueError, "duplicate checkpoint weights"):
            report.load_results(self.paths)

    def test_duplicate_seed_configuration_is_rejected(self):
        self.mutate(1, lambda d: d.update(training_seed=1337))
        with self.assertRaisesRegex(ValueError, "duplicate lexical size/seed"):
            report.load_results(self.paths)

    def test_paired_bootstrap_keeps_identical_model_delta_zero(self):
        # Independent resampling of the two sides would produce a nonzero CI.
        left = [[0, 50, -40, 0], [1, 0.1, 9, 1], [0, 4, -3, 1]]
        intervals = report.paired_bootstrap(left, left, samples=500, seed=42)
        for metric in intervals.values():
            self.assertEqual(metric, {"difference": 0.0, "ci95": [0.0, 0.0]})

    def test_paired_constant_difference_has_zero_sampling_width(self):
        right = [[0, 5, -1, 0], [1, 4, 2, 1], [0, 1, 7, 1]]
        left = [[x + 2 for x in row] for row in right]
        for metric in report.paired_bootstrap(left, right, samples=500, seed=1).values():
            self.assertEqual(metric, {"difference": 2.0, "ci95": [2.0, 2.0]})

    def test_fixed_seed_mean_preserves_item_dependence(self):
        # Opposite seed effects on each item cancel before bootstrapping.
        def alter(d):
            flip = d["training_seed"] == 1338
            for index, rec in enumerate(d["records"]):
                native = [0.0, -1.0] if ((index == 0) != flip) else [-1.0, 0.0]
                rec["answer_index"] = 0
                rec["routes"]["native"] = fixture_record(scores=native)["routes"]["native"]
                rec["routes"]["byte"] = fixture_record(scores=(-1.0, -1.0))["routes"]["byte"]
            d["aggregates"] = report.task_scorer.aggregate(d["records"], [])
        for i in range(4):
            self.mutate(i, alter)
        results = report.task_report(report.load_results(self.paths), samples=500, bootstrap_seed=1)
        means = [c for c in results["comparisons"] if c["training_seed"] == "fixed_seed_mean" and c["contrast"] == "native_minus_byte"]
        self.assertEqual(len(means), 2)
        for comparison in means:
            self.assertEqual(comparison["metrics"]["accuracy"], {"difference": 0.0, "ci95": [0.0, 0.0]})

    def lm_fixture(self, task_result):
        score = {"documents": 2, "source_bytes": 100, "text_nll": 100 * math.log(2), "bpb": 1.0}
        return {"lexical_size": task_result["lexical_size"], "training_seed": task_result["training_seed"], "model": task_result["model"],
                "metric": "canonical_tokenization_bits_per_utf8_byte", "context_mode": "bytes", "context_tokens": 512,
                "context_policy": "same UTF8 chunks", "eos_policy": "separate", "split": "test", "evaluated_text_sha256": {lang: "a" * 64 for lang in ("zh", "ko", "ja", "en")},
                "languages": {lang: {lang: dict(score), "byte": dict(score)} for lang in ("zh", "ko", "ja", "en")}}

    def test_complete_report_and_lm_metric_arithmetic(self):
        lm_paths = []
        for path in self.paths:
            lm = path.parent / "lm-bytes.json"
            lm.write_text(json.dumps(self.lm_fixture(json.loads(path.read_text()))))
            lm_paths.append(lm)
        summary = report.create_report(self.paths, self.root / "report", lm_paths=lm_paths, samples=100)
        self.assertEqual(len(summary["scores"]), 8)
        self.assertEqual(len(summary["comparisons"]), 12)
        self.assertEqual(len(summary["lm_scores"]), 32)
        self.assertTrue(all(row["byte_perplexity"] == 2.0 for row in summary["lm_scores"]))
        for name in ("report.md", "summary.json", "scores.csv"):
            self.assertTrue((self.root / "report" / name).is_file())
        self.assertIn("not official benchmark or leaderboard scores", (self.root / "report" / "report.md").read_text())
        self.assertIn("not training-seed uncertainty", summary["bootstrap"]["scope"])
        corrupt = json.loads(lm_paths[0].read_text())
        corrupt["languages"]["zh"]["zh"]["bpb"] = 2.0
        lm_paths[0].write_text(json.dumps(corrupt))
        with self.assertRaisesRegex(ValueError, "LM BPB"):
            report.load_lm_results(lm_paths, report.load_results(self.paths))

    def test_lm_incomplete_model_group_is_rejected(self):
        path = self.paths[0].parent / "lm-bytes.json"
        path.write_text(json.dumps(self.lm_fixture(json.loads(self.paths[0].read_text()))))
        with self.assertRaisesRegex(ValueError, "all four configurations"):
            report.load_lm_results([path], report.load_results(self.paths))

    def test_new_lm_wrapper_requires_provenance(self):
        path = self.paths[0].parent / "lm-bytes.json"
        fixture = self.lm_fixture(json.loads(self.paths[0].read_text()))
        fixture["evaluation_kind"] = "heldout_corpus_lm"
        path.write_text(json.dumps(fixture))
        with self.assertRaisesRegex(ValueError, "missing provenance"):
            report.load_lm_results([path], report.load_results(self.paths))


if __name__ == "__main__":
    unittest.main()
