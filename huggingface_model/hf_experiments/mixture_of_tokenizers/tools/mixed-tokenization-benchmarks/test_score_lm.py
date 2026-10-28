"""Analytical tests for held-out LM auditing; no torch/transformers required."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import score_lm as lm


def write_corpus(path, texts):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"text": text}, ensure_ascii=False) + "\n" for text in texts),
                    encoding="utf-8")


class FakeShared:
    def __init__(self):
        self.loads = 0
        self.corrupt_result = False
        self.fail_scoring = False

    def read_training_identity(self, model_dir, expected_seed=None):
        if expected_seed not in (None, 1337):
            raise ValueError("training seed mismatch")
        return {"training_seed": 1337, "lexical_size": 10000,
                "model_config": {"max_position_embeddings": 512},
                "run_config_sha256": "a" * 64, "summary_sha256": "b" * 64}

    def build_provenance(self, package_dir, model_dir, device, threads, script_path):
        return {"checkpoint_files_sha256": {"model.safetensors": lm.sha256_file(model_dir / "model.safetensors")},
                "experiment_sources_sha256": {"evaluate.py": lm.sha256_file(package_dir / "evaluate.py")},
                "scorer_sha256": lm.sha256_file(script_path), "device": device, "threads": threads,
                "precision_policy": "analytical fixture", "packages": {}}

    def load_checkpoint(self, package_dir, model_dir, device):
        self.loads += 1
        return self, SimpleNamespace(config=SimpleNamespace(max_position_embeddings=512)), SimpleNamespace(lexical_size=10000)

    def evaluate_corpus(self, model, tokenizer, data, split, max_documents, context_mode, device):
        if self.fail_scoring:
            raise ValueError("synthetic scoring failure")
        selected, hashes, _ = lm.select_documents(data, split, max_documents)
        languages = {}
        for language, records in selected.items():
            source_bytes = sum(row["source_bytes"] for row in records)
            source_chars = sum(row["source_characters"] for row in records)
            languages[language] = {}
            for route, bpb in ((language, 1.0), ("byte", 2.0)):
                languages[language][route] = {
                    "documents": len(records), "text_nll": bpb * source_bytes * math.log(2),
                    "text_tokens": source_bytes, "source_bytes": source_bytes,
                    "source_characters": source_chars, "fallback_tokens": source_bytes,
                    "fallback_bytes": source_bytes, "eos_nll": float(len(records)),
                    "eos_count": len(records), "windows": len(records), "bpb": bpb,
                }
        if self.corrupt_result:
            languages["zh"]["zh"]["source_bytes"] += 1
        return {"metric": lm.METRIC, "context_mode": context_mode, "context_tokens": 512,
                "context_policy": lm.POLICIES[context_mode], "eos_policy": lm.EOS_POLICY,
                "split": split, "evaluated_text_sha256": hashes, "languages": languages,
                "macro_average": {"native_bpb": 1.0, "byte_bpb": 2.0}}


class LMAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="lm-audit-tests-")
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        self.package = self.root / "project"
        self.model = self.package / "runs/k10000/checkpoint-best"
        self.model.mkdir(parents=True)
        (self.model / "model.safetensors").write_bytes(b"synthetic weights")
        (self.package / "evaluate.py").write_text("# synthetic source\n")
        for language in lm.LANGUAGES:
            write_corpus(self.data / language / "test.jsonl", [f"{language} test 你好", f"{language} test two"])
            write_corpus(self.data / language / "val.jsonl", [f"{language} validation one", f"{language} validation two"])
            write_corpus(self.data / language / "train.jsonl", [f"{language} training one", f"{language} training two"])
        self.args = lm.build_parser().parse_args([
            "--package-dir", str(self.package), "--model", str(self.model), "--data", str(self.data),
            "--output", str(self.root / "result.json"), "--max-documents", "2", "--device", "cpu",
            "--training-seed", "1337", "--overlap-cache", str(self.root / "cache")])
        self.shared = FakeShared()

    def tearDown(self):
        self.temporary.cleanup()

    def evaluate(self):
        return lm.run(self.args, shared=self.shared, runtime_setup=lambda device, threads: None)

    def test_exact_unicode_selection_and_requested_count(self):
        selected, hashes, files = lm.select_documents(self.data, "test", 2)
        texts = list(lm.iter_texts(self.data / "zh/test.jsonl"))
        digest = hashlib.sha256()
        for text in texts:
            encoded = text.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
        self.assertEqual(hashes["zh"], digest.hexdigest())
        self.assertGreater(selected["zh"][0]["source_bytes"], selected["zh"][0]["source_characters"])
        self.assertEqual(len(files), 4)
        with self.assertRaisesRegex(ValueError, "only 2 are available"):
            lm.select_documents(self.data, "test", 3)

    def test_empty_selected_document_rejected_without_filtering(self):
        write_corpus(self.data / "zh/test.jsonl", ["", "nonempty"])
        with self.assertRaisesRegex(ValueError, "Empty selected document"):
            self.evaluate()
        self.assertEqual(self.shared.loads, 0)
        self.assertFalse(Path(self.args.output).exists())

    def test_train_and_validation_overlap_each_rejected(self):
        text = next(lm.iter_texts(self.data / "zh/test.jsonl"))
        for split in ("train", "val"):
            path = self.data / f"zh/{split}.jsonl"
            previous = path.read_bytes()
            write_corpus(path, [text])
            with self.subTest(split=split), self.assertRaisesRegex(ValueError, "Held-out overlap"):
                self.evaluate()
            path.write_bytes(previous)
        self.assertEqual(self.shared.loads, 0)
        self.assertFalse(Path(self.args.output).exists())

    def test_validation_mode_does_not_compare_against_itself(self):
        self.args.split = "val"
        result = self.evaluate()
        self.assertEqual(result["split"], "val")
        self.assertEqual(set(result["overlap_audit"]["comparisons"]["zh"]), {"train"})

    def test_overlap_cache_reuse_and_content_change_invalidation(self):
        selected, _, _ = lm.select_documents(self.data, "test", 2)
        first = lm.audit_overlap(self.data, "test", selected, self.root / "cache")
        with mock.patch.object(lm, "scan_overlap", side_effect=AssertionError("should reuse cache")):
            self.assertEqual(lm.audit_overlap(self.data, "test", selected, self.root / "cache"), first)
        write_corpus(self.data / "en/train.jsonl", [next(lm.iter_texts(self.data / "en/test.jsonl"))])
        with self.assertRaisesRegex(ValueError, "Held-out overlap"):
            lm.audit_overlap(self.data, "test", selected, self.root / "cache")

    def test_corrupt_cache_fails_closed(self):
        selected, _, _ = lm.select_documents(self.data, "test", 2)
        lm.audit_overlap(self.data, "test", selected, self.root / "cache")
        cache = next((self.root / "cache").glob("*.json"))
        body = lm.read_json(cache)
        body["result"]["matching_selected_documents"] = 99
        cache.write_text(json.dumps(body))
        with self.assertRaisesRegex(ValueError, "integrity check"):
            lm.audit_overlap(self.data, "test", selected, self.root / "cache")

    def test_analytical_aggregate_and_strict_reuse(self):
        first = self.evaluate()
        self.assertEqual(first["macro_average"], {"native_bpb": 1.0, "byte_bpb": 2.0})
        self.assertEqual(first["training_seed"], 1337)
        self.assertEqual(first["overlap_audit"]["status"], "passed")
        original = Path(self.args.output).read_bytes()
        self.args.skip_existing = True
        second = self.evaluate()
        self.assertEqual(first, second)
        self.assertEqual(self.shared.loads, 1)
        self.assertEqual(Path(self.args.output).read_bytes(), original)
        (self.model / "model.safetensors").write_bytes(b"changed weights")
        with self.assertRaisesRegex(ValueError, "provenance changed"):
            self.evaluate()
        self.assertEqual(Path(self.args.output).read_bytes(), original)

    def test_reuse_rechecks_stored_aggregate_arithmetic(self):
        self.evaluate()
        saved = lm.read_json(self.args.output)
        saved["languages"]["zh"]["zh"]["bpb"] = 999
        Path(self.args.output).write_text(json.dumps(saved))
        self.args.skip_existing = True
        with self.assertRaisesRegex(ValueError, "BPB arithmetic mismatch"):
            self.evaluate()

    def test_source_total_failure_and_scoring_failure_publish_nothing(self):
        self.shared.corrupt_result = True
        with self.assertRaisesRegex(ValueError, "Selected source totals"):
            self.evaluate()
        self.assertFalse(Path(self.args.output).exists())
        self.shared.corrupt_result = False
        self.shared.fail_scoring = True
        with self.assertRaisesRegex(ValueError, "synthetic scoring failure"):
            self.evaluate()
        self.assertFalse(Path(self.args.output).exists())

    def test_atomic_publish_does_not_replace(self):
        path = self.root / "atomic.json"
        lm.publish_json(path, {"original": True})
        with self.assertRaises(FileExistsError):
            lm.publish_json(path, {"original": False})
        self.assertEqual(lm.read_json(path), {"original": True})
        self.assertEqual(list(self.root.glob("*.partial")), [])

    def test_explicit_overlap_opt_out_is_recorded(self):
        self.args.check_overlap = False
        write_corpus(self.data / "zh/train.jsonl", [next(lm.iter_texts(self.data / "zh/test.jsonl"))])
        result = self.evaluate()
        self.assertEqual(result["overlap_audit"]["status"], "not_checked")
        self.assertFalse(result["provenance"]["request"]["check_overlap"])

    def test_positive_controls_and_seed_identity(self):
        for field in ("max_documents", "threads"):
            args = copy.copy(self.args)
            setattr(args, field, 0)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "must be positive"):
                lm.run(args, shared=self.shared, runtime_setup=lambda *_: None)
        self.args.training_seed = 1338
        with self.assertRaisesRegex(ValueError, "seed mismatch"):
            self.evaluate()

    def test_shared_helper_provenance_interface(self):
        import score_tasks
        for name in ("modeling_mixture.py", "configuration_mixture.py", "tokenization_mixture.py"):
            (self.package / name).write_text("# analytical fixture\n")
        (self.model / "tokenizer").mkdir()
        (self.model / "tokenizer/mixture_manifest.json").write_text("{}\n")
        (self.model / "config.json").write_text(json.dumps({"lexical_size": 10000,
            "max_position_embeddings": 512, "tokenizer_fingerprint": "fixture"}))
        (self.model.parent / "run_config.json").write_text(json.dumps({
            "effective_training": {"seed": 1337}, "tokenizer_fingerprint": "fixture"}))
        (self.model.parent / "summary.json").write_text(json.dumps({
            "settings": {"seed": 1337}, "tokenizer_fingerprint": "fixture"}))
        with mock.patch.object(score_tasks, "load_checkpoint", side_effect=self.shared.load_checkpoint):
            report = lm.run(self.args, shared=score_tasks, runtime_setup=lambda *_: None)
        proof = report["provenance"]
        self.assertEqual(set(proof["experiment_sources_sha256"]),
                         {"evaluate.py", "modeling_mixture.py", "configuration_mixture.py", "tokenization_mixture.py"})
        self.assertIn("model.safetensors", proof["checkpoint_files_sha256"])
        self.assertIn("tokenizer/mixture_manifest.json", proof["checkpoint_files_sha256"])
        self.assertEqual(proof["training_identity"]["training_seed"], 1337)


if __name__ == "__main__":
    unittest.main()
