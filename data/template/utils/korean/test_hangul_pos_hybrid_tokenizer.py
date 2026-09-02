#!/usr/bin/env python3
"""Comprehensive unit tests for HangulHybridPosTokenizer."""
from __future__ import annotations

import os
import tempfile
import unittest

import sentencepiece as spm
from hangul_pos_hybrid_tokenizer import (
    HangulHybridPosTokenizer,
    train_sentencepiece_non_korean,
)


class TestHangulHybridPosTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.corpus_path = os.path.join(cls.temp_dir.name, "sample_corpus.txt")
        with open(cls.corpus_path, "w", encoding="utf-8") as f:
            f.write(
                "Hello world! This is a test corpus with numbers 12345, punctuation !?, "
                "slang ㅋㅋㅋ ㅎㅎㅎ ㅠㅠ, symbols @#$%^&*(), and lines.\nLine 2\tTabbed!"
            )
        cls.spm_prefix = os.path.join(cls.temp_dir.name, "spm_test")
        cls.spm_model_path = train_sentencepiece_non_korean(
            input_file=cls.corpus_path,
            output_prefix=cls.spm_prefix,
            vocab_size=512,
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_lane_structure(self):
        """Verify that exactly 24 lanes are created."""
        tok = HangulHybridPosTokenizer(self.spm_model_path, use_pos=True, pos_mode="coarse")
        self.assertEqual(len(tok.lanes), 24)
        self.assertEqual(len(tok.lane_names), 24)
        self.assertEqual(tok.lane_names[0], "script_bpe")
        self.assertEqual(tok.lane_names[1], "choseong")
        self.assertEqual(tok.lane_names[2], "jungseong")
        self.assertEqual(tok.lane_names[3], "jongseong")
        self.assertEqual(tok.lane_names[-1], "pos")

    def test_roundtrip_pure_korean(self):
        """Test roundtrip for purely Korean text."""
        tok = HangulHybridPosTokenizer(self.spm_model_path, use_pos=True, pos_mode="coarse")
        text = "안녕하세요반갑습니다대한민국"
        encoded = tok.encode_text(text)
        self.assertEqual(len(encoded), len(text))
        for step in encoded:
            self.assertEqual(len(step), 24)
            self.assertEqual(step[0], tok.hangul_id)
        decoded = tok.decode_sequence(encoded)
        self.assertEqual(text, decoded)

    def test_roundtrip_mixed_text(self):
        """Test roundtrip for mixed Korean, English, numbers, slang, and punctuation."""
        tok = HangulHybridPosTokenizer(self.spm_model_path, use_pos=True, pos_mode="coarse")
        test_cases = [
            "안녕하세요! 반갑습니다.",
            "Hello World! 12345 special symbols: @#$%^&*()",
            "학생들이 밥을 먹었습니다. ㅋㅋㅋ ㅎㅎㅎ",
            "Mixed English and Korean: GPT-4와 LLM 모델들의 융합 연구.",
            "Multi-line:\nLine 1\nLine 2\tTabbed!",
            "Unicode tests: 훈민정음 漢字 🚀 🔥 🧑‍💻 $\\alpha + \\beta = \\gamma$",
        ]
        for s in test_cases:
            encoded = tok.encode_text(s)
            for step in encoded:
                self.assertEqual(len(step), 24)
            decoded = tok.decode_sequence(encoded)
            self.assertEqual(s, decoded, f"Failed on: {s!r}")

    def test_full_pos_mode(self):
        """Test full 46-tag Sejong POS mode."""
        tok = HangulHybridPosTokenizer(self.spm_model_path, use_pos=True, pos_mode="full")
        self.assertEqual(len(tok.lanes), 24)
        self.assertEqual(tok.pos_mode, "full")
        text = "학생들이 밥을 먹었다."
        encoded = tok.encode_text(text)
        decoded = tok.decode_sequence(encoded)
        self.assertEqual(text, decoded)


if __name__ == "__main__":
    unittest.main()
