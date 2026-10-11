import json
import tempfile
import unittest
from pathlib import Path

from cjk_sentence_pairs import build_sentence_pairs


class CjkSentencePairsTest(unittest.TestCase):
    def test_builds_deterministic_utf8_records_and_skips_invalid_rows(self):
        sample = """#EN:
hello
#KO:
안녕하세요
#ZH:
你好
#JA:
こんにちは
#EN:
duplicate
#KO:
중복
#ZH:
重复
#JA:
重複
#EN:
duplicate again
#KO:
중복
#ZH:
重复
#JA:
重複
#EN:
empty zh
#KO:
비어 있음
#ZH:

#JA:
空
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.txt"
            input_path.write_text(sample, encoding="utf-8")
            out = root / "data" / "cjk_sentence_pairs.json"

            result = build_sentence_pairs(input_path, out)
            payload = json.loads(out.read_text(encoding="utf-8"))

            self.assertEqual(payload["source"], "ntrex")
            self.assertEqual(payload["languages"], ["kor_Hang", "zho_Hans", "jpn_Jpan"])
            self.assertEqual(payload["canonical_language_codes"], ["ja", "ko", "zh"])
            self.assertEqual(result.records, 2)
            self.assertEqual(
                result.skipped,
                {"duplicate_aligned_record": 1, "empty_sentence": 1},
            )
            self.assertEqual(
                [record["id"] for record in payload["records"]],
                sorted(record["id"] for record in payload["records"]),
            )
            self.assertIn("안녕하세요", out.read_text(encoding="utf-8"))
            self.assertNotIn("\\uc548", out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
