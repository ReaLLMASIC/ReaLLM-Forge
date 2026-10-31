"""CJK preparation checks that do not require IPA conversion or a GPU."""

import csv
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from data.template.utils.build_char_fallback_vocab import read_characters, select_characters
from utils.cjk_comparison_metrics import FIELDS, snapshot_metrics, training_token_stats


ROOT = Path(__file__).resolve().parents[1]


class FallbackPipelineTests(unittest.TestCase):
    def test_tiny_models_train_and_export_new_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory) / "data"
            source_root = data_root / "_source"
            source_root.mkdir(parents=True)
            for split in ("train", "val"):
                (source_root / f"ja.{split}.original.txt").write_text("甲乙 a\n" * 8, encoding="utf-8")
            out_root = Path(directory) / "out"
            env = dict(os.environ, CJK_DATA_ROOT=str(data_root), CJK_OUT_ROOT=str(out_root),
                       PYTHON_BIN=sys.executable, FALLBACK_CHAR_LIMIT="1",
                       COMPARISON_TOKENIZER="char_byte_fallback", FALLBACK_CHARS_FILE="",
                       RUN_FILTER="^ja_original_", OMP_NUM_THREADS="1", MPLBACKEND="Agg")
            env.pop("RANK", None)
            preparation = subprocess.run([
                "bash", "-c", "source demos/cjk_ipa_char_byte_fallback_compare.sh; "
                "prepare_dataset ja original char; prepare_dataset ja original char_byte_fallback",
            ], cwd=ROOT, env=env, capture_output=True, text=True, timeout=90)
            self.assertEqual(preparation.returncode, 0, preparation.stdout + preparation.stderr)
            for method in ("char", "char_byte_fallback"):
                run = "ja_original_" + method
                out_dir = out_root / run
                # Full metrics tests cover real PNG rendering. Keep actual training,
                # geometry, CSV and HTML here while skipping expensive repeated PNGs.
                command = [sys.executable, "-c",
                           "import utils.per_token_static as s; s.write_static_dashboards=lambda *a: []; "
                           "import train; train.main()",
                           "--dataset", str(data_root / run), "--out_dir", str(out_dir),
                           "--device", "cpu", "--dtype", "float32", "--n_layer", "1",
                           "--n_head", "1", "--n_kv_group", "1", "--n_embd", "8",
                           "--block_size", "4", "--batch_size", "1", "--eval_iters", "1",
                           "--eval_interval", "1", "--max_iters", "1", "--warmup_iters", "0",
                           "--gradient_accumulation_steps", "1", "--no-compile",
                           "--no-tensorboard_log", "--no-csv_log", "--no-wandb_log",
                           "--no-print_model_info", "--log_per_token_metrics",
                           "--only_save_checkpoint_at_end", "--sample_file", str(out_dir / "sample.txt")]
                result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                                        text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                env["TEST_RUN_OUT"] = str(out_dir)
                validation = subprocess.run([
                    "bash", "-c", "source demos/cjk_ipa_char_byte_fallback_compare.sh; "
                    'validate_training_outputs "$TEST_RUN_OUT"',
                ], cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
                self.assertEqual(validation.returncode, 0, validation.stdout + validation.stderr)
                with (out_dir / "per_token_metrics" / "per_token_metrics.csv").open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual({int(row["iteration"]) for row in rows}, {0, 1})
                self.assertGreater(sum(int(row["training_seen_count"]) for row in rows), 0)
                stats_result = subprocess.run([
                    "bash", "-c", "source demos/cjk_ipa_char_byte_fallback_compare.sh; "
                    'write_training_token_stats "$TEST_RUN_OUT"',
                ], cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
                self.assertEqual(stats_result.returncode, 0, stats_result.stdout + stats_result.stderr)
                stats = json.loads((out_dir / "training_token_stats.json").read_text())
                latest = [row for row in rows if int(row["iteration"]) == 1]
                self.assertEqual(stats["vocab_size"], len(latest))
                self.assertEqual(stats["training_tokens_processed"],
                                 sum(int(row["training_seen_count"]) for row in latest))
                self.assertEqual(stats["training_unique_token_ids"],
                                 sum(int(row["training_seen_count"]) > 0 for row in latest))
                (out_dir / ".complete").write_text("test-completed\n", encoding="utf-8")
            summary = subprocess.run([
                "bash", "-c", "source demos/cjk_ipa_char_byte_fallback_compare.sh; write_comparison_summary",
            ], cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(summary.returncode, 0, summary.stdout + summary.stderr)
            with (out_root / "comparison_summary.tsv").open(newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertGreater(float(row["val_avg_target_probability"]), 0)
                self.assertGreaterEqual(float(row["val_avg_target_rank"]), 1)
                self.assertEqual(int(row["training_tokens_iteration"]), 1)
                self.assertGreater(int(row["training_tokens_processed"]), 0)
                self.assertLessEqual(int(row["training_unique_token_ids"]), int(row["vocab_size"]))

    def test_training_token_counts_use_latest_snapshot_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=("iteration", "token_id", "training_seen_count"))
                writer.writeheader()
                for iteration, counts in ((0, [0, 0, 0]), (5, [4, 2, 0]), (10, [8, 3, 0])):
                    for token_id, count in enumerate(counts):
                        writer.writerow(dict(iteration=iteration, token_id=token_id, training_seen_count=count))
            self.assertEqual(training_token_stats(path), {
                "vocab_size": 3, "training_tokens_iteration": 10,
                "training_tokens_processed": 11, "training_unique_token_ids": 2,
            })

    def test_training_only_selection_and_whitelist_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.txt"
            path.write_text("乙甲乙甲\n 丙\t", encoding="utf-8")
            self.assertEqual(select_characters(path, 2), ["乙", "甲"])
            path.write_text("甲\n甲\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_characters(path)
            path.write_text("ab\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "one Unicode"):
                read_characters(path)

    def test_comparison_uses_validation_counts_at_requested_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=("iteration", "val_eval_count", *FIELDS))
                writer.writeheader()
                for iteration, count, values in (
                    (10, 1, (0.2, 5, 0.8)), (10, 3, (0.6, 1, 0.0)),
                    (10, 0, (float("nan"),) * 3), (20, 8, (1, 1, 0)),
                ):
                    writer.writerow(dict(iteration=iteration, val_eval_count=count,
                                         **dict(zip(FIELDS, values))))
            for actual, expected in zip(snapshot_metrics(path, 10), (0.5, 2, 0.2)):
                self.assertAlmostEqual(actual, expected)
            with self.assertRaisesRegex(ValueError, "no evaluated"):
                snapshot_metrics(path, 30)

    def test_real_preparation_char_fallback_and_byte(self):
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory) / "data"
            source_root = data_root / "_source"
            source_root.mkdir(parents=True)
            train_text = "甲甲乙 a\n"
            val_text = "甲丙😊 a\n"
            for split, value in (("train", train_text), ("val", val_text)):
                (source_root / f"ja.{split}.original.txt").write_text(value, encoding="utf-8")
            env = dict(os.environ, CJK_DATA_ROOT=str(data_root), PYTHON_BIN=sys.executable,
                       FALLBACK_CHAR_LIMIT="1", COMPARISON_TOKENIZER="char_byte_fallback",
                       FALLBACK_CHARS_FILE="", RUN_FILTER="")
            command = '''
source demos/cjk_ipa_char_byte_fallback_compare.sh
for method in char char_byte_fallback byte; do
  prepare_dataset ja original "$method"
done
prepare_dataset ja original char_byte_fallback
'''
            result = subprocess.run(["bash", "-c", command], cwd=ROOT, env=env,
                                    capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("skipping tokenization", result.stdout)
            self.assertEqual(len(list(data_root.glob("*/train.bin"))), 3)
            with (data_root / "ja_original_char_byte_fallback" / "meta.pkl").open("rb") as handle:
                meta = pickle.load(handle)
            self.assertEqual(meta["custom_chars"], ["甲"])
            self.assertEqual(meta["vocab_size"], 257)
            dtype = meta.get("dtype", "uint16")
            ids = np.fromfile(data_root / "ja_original_char_byte_fallback" / "val.bin", dtype=dtype)
            expected = [256] + list("丙😊 a\n".encode("utf-8"))
            self.assertEqual(ids.tolist(), expected)
            self.assertEqual(meta["byte_metrics"]["val_tokens_per_byte"],
                             len(expected) / len(val_text.encode("utf-8")))
            env["FALLBACK_CHAR_LIMIT"] = "2"
            result = subprocess.run(["bash", "-c", "source demos/cjk_ipa_char_byte_fallback_compare.sh; prepare_dataset ja original char_byte_fallback"],
                                    cwd=ROOT, env=env, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            with (data_root / "ja_original_char_byte_fallback" / "meta.pkl").open("rb") as handle:
                self.assertEqual(pickle.load(handle)["vocab_size"], 258)


if __name__ == "__main__":
    unittest.main()
