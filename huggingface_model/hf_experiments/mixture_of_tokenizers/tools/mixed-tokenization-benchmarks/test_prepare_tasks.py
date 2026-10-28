"""Fixture and integrity tests; no network access or benchmark data required."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_tasks as prep


def jsonl(*rows):
    return ("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n").encode("utf-8")


class AdapterTests(unittest.TestCase):
    def test_xcopa_both_causal_directions_and_original_ids(self):
        examples = [
            {"premise": "地面湿了。", "choice1": "下雨了。", "choice2": "太阳出来了。", "question": "cause", "label": 0, "idx": 17, "changed": False},
            {"premise": "下雨了。", "choice1": "路面干了。", "choice2": "路面湿了。", "question": "effect", "label": 1, "idx": 42, "changed": True},
        ]
        rows = prep.normalize_task("xcopa_zh", "test", jsonl(*examples), expected_count=2)
        self.assertEqual(rows[0]["prompt"], "地面湿了，因为")
        self.assertEqual(rows[1]["prompt"], "下雨了，所以")
        self.assertEqual(rows[0]["id"], "xcopa_zh:test:17")
        self.assertEqual(rows[0]["choices"][rows[0]["answer_index"]], "下雨了。")
        self.assertEqual(rows[1]["choices"][rows[1]["answer_index"]], "路面湿了。")
        self.assertEqual(set(rows[0]), {"id", "task", "language", "split", "prompt", "choices", "answer_index"})
        with self.assertRaisesRegex(prep.PreparationError, "Duplicate source ID"):
            prep.normalize_task("xcopa_zh", "test", jsonl(examples[0], examples[0]))

    def test_kobest_korean_question_codes_and_answer_offset(self):
        example = {"premise": "물을 오래 끓였다.", "question": "결과", "alternative_1": "물이 늘었다.", "alternative_2": "물이 줄었다.", "label": 1}
        row = prep.normalize_task("kobest_copa_ko", "validation", jsonl(example), expected_count=1)[0]
        self.assertEqual(row["prompt"], "물을 오래 끓였다. 그래서")
        self.assertEqual(row["choices"][row["answer_index"]], " 물이 줄었다.")
        example["question"] = "원인"
        self.assertTrue(prep.normalize_task("kobest_copa_ko", "validation", jsonl(example))[0]["prompt"].endswith(" 왜냐하면"))
        example["question"] = "cause"
        with self.assertRaisesRegex(prep.PreparationError, "Unknown KoBEST question"):
            prep.normalize_task("kobest_copa_ko", "validation", jsonl(example))

    def test_jcommonsenseqa_five_choices_keep_source_label(self):
        example = {"q_id": 8939, "question": "空の色は？", "choice0": "赤", "choice1": "緑", "choice2": "青", "choice3": "黄", "choice4": "白", "label": 2}
        row = prep.normalize_task("jcommonsenseqa_ja", "validation", jsonl(example), expected_count=1)[0]
        self.assertEqual(row["prompt"], "質問：空の色は？\n答え：")
        self.assertEqual(row["choices"][row["answer_index"]], "青")
        self.assertEqual(row["id"], "jcommonsenseqa_ja:validation:8939")
        example["label"] = 5
        with self.assertRaisesRegex(prep.PreparationError, "Invalid answer index"):
            prep.normalize_task("jcommonsenseqa_ja", "validation", jsonl(example))
        example["label"] = True
        with self.assertRaisesRegex(prep.PreparationError, "Invalid answer index"):
            prep.normalize_task("jcommonsenseqa_ja", "validation", jsonl(example))

    def test_piqa_separate_labels_stay_aligned(self):
        examples = [{"goal": "Keep water cold.", "sol1": "Add ice.", "sol2": "Add heat."}, {"goal": "Dry a plate.", "sol1": "Add water.", "sol2": "Use a towel."}]
        rows = prep.normalize_task("piqa_en", "validation", jsonl(*examples), b"0\n1\n", expected_count=2)
        self.assertEqual(rows[0]["prompt"], "Question: Keep water cold.\nAnswer:")
        self.assertEqual([r["choices"][r["answer_index"]] for r in rows], [" Add ice.", " Use a towel."])
        self.assertEqual(rows[1]["id"], "piqa_en:validation:1")
        for bad_labels in (None, b"0\n", b"0\n2\n", b"0\ntrue\n", b" 0\n1\n"):
            with self.subTest(labels=bad_labels), self.assertRaises(prep.PreparationError):
                prep.normalize_task("piqa_en", "validation", jsonl(*examples), bad_labels)

    def test_rejects_bad_counts_schema_and_duplicate_json_keys(self):
        fixture = jsonl({"goal": "Do something.", "sol1": "One.", "sol2": "Two."})
        with self.assertRaisesRegex(prep.PreparationError, "expected 2 rows"):
            prep.normalize_task("piqa_en", "validation", fixture, b"0\n", expected_count=2)
        with self.assertRaisesRegex(prep.PreparationError, "unexpected schema"):
            prep.normalize_task("piqa_en", "validation", jsonl({"goal": "Only a goal"}), b"0\n")
        with self.assertRaisesRegex(prep.PreparationError, "Duplicate JSON key"):
            prep.parse_jsonl(b'{"label":0,"label":1}\n')
        with self.assertRaisesRegex(prep.PreparationError, "Blank JSONL"):
            prep.parse_jsonl(fixture + b"\n")


class CacheIntegrityTests(unittest.TestCase):
    def test_offline_cache_tampering_fails_without_network_or_replacement(self):
        original = b'{"fixture":true}\n'
        source = prep.Source("fixture", "test", "fixture.jsonl", "https://invalid.example/fixture", hashlib.sha256(original).hexdigest(), "commit", "fixture-license", "https://invalid.example/license")
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            path = output / source.cache_path
            path.parent.mkdir(parents=True)
            path.write_bytes(original)
            with patch("prepare_tasks.urllib.request.urlopen", side_effect=AssertionError("Network must not be called")):
                self.assertEqual(prep.read_source(source, output, offline=True), original)
                path.write_bytes(b"tampered")
                for offline in (True, False):
                    with self.subTest(offline=offline), self.assertRaisesRegex(prep.PreparationError, "SHA-256 mismatch"):
                        prep.read_source(source, output, offline=offline)
                    self.assertEqual(path.read_bytes(), b"tampered")

    def test_offline_missing_source_and_artifact_fail_without_creation(self):
        source = prep.sources_for("final")[0]
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "missing"
            with self.assertRaisesRegex(prep.PreparationError, "Offline source cache is missing"):
                prep.read_source(source, output, offline=True)
            with self.assertRaisesRegex(prep.PreparationError, "Offline artifact is missing"):
                prep.write_or_verify_artifact(output / "tasks.jsonl", b"fixture", offline=True)
            self.assertFalse(output.exists())

    def test_stale_normalized_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tasks.jsonl"
            prep.write_or_verify_artifact(path, b"original", offline=False)
            prep.write_or_verify_artifact(path, b"original", offline=True)
            with self.assertRaisesRegex(prep.PreparationError, "Existing artifact differs"):
                prep.write_or_verify_artifact(path, b"new protocol", offline=False)
            self.assertEqual(path.read_bytes(), b"original")

    def test_split_selection_keeps_piqa_validation_explicit(self):
        for split_set, cjk_split, total in (("dev", "validation", 3557), ("final", "test", 4456)):
            sources = prep.sources_for(split_set)
            examples = [x for x in sources if x.role == "examples"]
            self.assertEqual([x.split for x in examples], [cjk_split, cjk_split, cjk_split, "validation"])
            self.assertEqual(sum(prep.EXPECTED_COUNTS[x.task][x.split] for x in examples), total)
            self.assertTrue(all(len(x.sha256) == 64 for x in sources))


if __name__ == "__main__":
    unittest.main()
