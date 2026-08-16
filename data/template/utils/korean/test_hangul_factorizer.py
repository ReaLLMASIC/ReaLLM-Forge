#!/usr/bin/env python3
"""Tests for HangulFactorizedTokenizer with and without POS lane."""
from __future__ import annotations

import unittest
from hangul_factorizer import (
    HangulFactorizedTokenizer,
    HangulPosFactorizedTokenizer,
    HangulFactorizedPosTokenizer,
    LANES,
    LANES_WITH_POS,
)


class TestHangulFactorizedTokenizer(unittest.TestCase):
    def test_default_without_pos(self):
        """Test default HangulFactorizedTokenizer without POS lane (23 lanes)."""
        tok = HangulFactorizedTokenizer()
        self.assertFalse(tok.use_pos)
        self.assertEqual(len(tok.lanes), 23)
        self.assertEqual(len(tok.lane_names), 23)
        self.assertNotIn("pos", tok.lane_names)

        # Encode character
        ids = tok.encode_char("가")
        self.assertEqual(len(ids), 23)

        # Decode character
        decoded = tok.decode_indices(ids)
        self.assertEqual(decoded, "가")

        # Metadata
        meta = tok.metadata_for_char("가", position=0)
        self.assertEqual(len(meta["lanes"]), 23)
        self.assertNotIn("pos", meta["lanes"])

    def test_with_pos_flag(self):
        """Test HangulFactorizedTokenizer with use_pos=True (24 lanes)."""
        tok = HangulFactorizedTokenizer(use_pos=True)
        self.assertTrue(tok.use_pos)
        self.assertEqual(len(tok.lanes), 24)
        self.assertEqual(len(tok.lane_names), 24)
        self.assertEqual(tok.lane_names[-1], "pos")

        # Encode character with pos tag
        ids_nng = tok.encode_char("가", pos_tag="NNG")
        self.assertEqual(len(ids_nng), 24)
        pos_lane_idx = 23
        self.assertEqual(tok.id_to_value[pos_lane_idx][ids_nng[pos_lane_idx]], "NOUN")

        ids_jks = tok.encode_char("가", pos_tag="JKS")
        self.assertEqual(len(ids_jks), 24)
        self.assertEqual(tok.id_to_value[pos_lane_idx][ids_jks[pos_lane_idx]], "JOSA")
        self.assertNotEqual(ids_nng[pos_lane_idx], ids_jks[pos_lane_idx])

        # Decode character still works
        decoded = tok.decode_indices(ids_nng)
        self.assertEqual(decoded, "가")

        # Metadata
        meta = tok.metadata_for_char("가", position=0, pos_tag="NNG")
        self.assertEqual(len(meta["lanes"]), 24)
        self.assertEqual(meta["lanes"]["pos"]["value"], "NOUN")

    def test_new_pos_tokenizer_class(self):
        """Test HangulPosFactorizedTokenizer / HangulFactorizedPosTokenizer class."""
        tok = HangulPosFactorizedTokenizer()
        self.assertTrue(tok.use_pos)
        self.assertEqual(len(tok.lanes), 24)
        self.assertEqual(tok.lane_names[-1], "pos")

        alias_tok = HangulFactorizedPosTokenizer()
        self.assertTrue(alias_tok.use_pos)
        self.assertEqual(len(alias_tok.lanes), 24)

    def test_encode_text_with_pos(self):
        """Test encode_text method with POS tagger fallback or kiwipiepy."""
        tok = HangulFactorizedTokenizer(use_pos=True)
        text = "학생들이 밥을 먹었습니다"
        seq = tok.encode_text(text, return_tags=False)
        self.assertEqual(len(seq), len(text))
        for item in seq:
            self.assertIn("char", item)
            self.assertIn("indices", item)
            self.assertEqual(len(item["indices"]), 24)


    def test_full_pos_tokenizer(self):
        """Test HangulFullPosFactorizedTokenizer with fine-grained Sejong tags."""
        from hangul_factorizer import HangulFullPosFactorizedTokenizer, POS_TAGS_FULL
        tok = HangulFullPosFactorizedTokenizer()
        self.assertTrue(tok.use_pos)
        self.assertEqual(tok.pos_mode, "full")
        self.assertEqual(len(tok.lanes), 24)
        self.assertEqual(len(tok.id_to_value[23]), len(POS_TAGS_FULL) + 2)  # PAD + UNK + 46 tags

        # Encode character with fine-grained POS tag
        ids_nng = tok.encode_char("가", pos_tag="NNG")
        self.assertEqual(tok.id_to_value[23][ids_nng[23]], "NNG")
        ids_jks = tok.encode_char("가", pos_tag="JKS")
        self.assertEqual(tok.id_to_value[23][ids_jks[23]], "JKS")

    def test_byte_fallback_meta(self):
        """Test make_byte_fallback_meta for 256-byte companion stream."""
        from hangul_factorizer import make_byte_fallback_meta
        meta = make_byte_fallback_meta()
        self.assertEqual(meta["vocab_size"], 256)
        stoi = meta["stoi"]
        # ASCII characters map directly to byte value
        self.assertEqual(stoi.get("A"), ord("A"))
        self.assertEqual(stoi.get("\n"), ord("\n"))
        self.assertEqual(len(stoi), 256)


if __name__ == "__main__":
    unittest.main()
