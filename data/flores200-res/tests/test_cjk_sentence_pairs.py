import json
import tempfile
import unittest
from pathlib import Path

from cjk_sentence_pairs import build_sentence_pairs


class CjkSentencePairsTest(unittest.TestCase):
    def test_builds_deterministic_utf8_records_and_skips_invalid_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "text_kor_Hang.txt").write_text(
                "안녕하세요\n중복\n중복\n\n", encoding="utf-8"
            )
            (root / "text_zho_Hans.txt").write_text(
                "你好\n重复\n重复\n空韩语\n", encoding="utf-8"
            )
            (root / "text_jpn_Jpan.txt").write_text(
                "こんにちは\n重複\n重複\n空の韓国語\n", encoding="utf-8"
            )

            out = root / "data" / "cjk_sentence_pairs.json"
            result = build_sentence_pairs(root / "text_zho_Hans.txt", out)
            payload = json.loads(out.read_text(encoding="utf-8"))

            self.assertEqual(payload["source"], "flores200-res")
            self.assertEqual(payload["languages"], ["ko", "zh", "ja"])
            self.assertEqual(result.records, 2)
            self.assertEqual(result.skipped, {"duplicate_aligned_record": 1, "empty_sentence": 1})
            self.assertEqual(
                [record["translations"] for record in payload["records"]],
                sorted(
                    [record["translations"] for record in payload["records"]],
                    key=lambda translations: record_sort_key(payload, translations),
                ),
            )
            self.assertIn("안녕하세요", out.read_text(encoding="utf-8"))
            self.assertNotIn("\\uc548", out.read_text(encoding="utf-8"))


def record_sort_key(payload, translations):
    for record in payload["records"]:
        if record["translations"] == translations:
            return (record["id"], record["line_number"])
    raise AssertionError("missing record")


if __name__ == "__main__":
    unittest.main()
