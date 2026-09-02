#!/usr/bin/env python3
"""Tokenizers for Seq2Seq English-to-Korean translation benchmark.

Implements:
1. EnglishBPETokenizer: SentencePiece BPE tokenizer for source English text.
2. ThreeHotSeq2SeqTokenizer: Triplet (initial, vowel, final) tokenizer matching Cognetta et al. (EACL 2023).
3. HangulFactorizerSeq2SeqTokenizer: 23-lane articulatory/phonetic factorizer based on HangulFactorizedTokenizer.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import sentencepiece as spm
import torch

from data.template.utils.korean.hangul_factorizer import (
    CHOSEONG,
    JUNGSEONG,
    JONGSEONG,
    S_BASE,
    S_COUNT,
    N_COUNT,
    V_COUNT,
    T_COUNT,
    PAD,
    LANES,
    HangulFactorizedTokenizer,
)

# Shared Special Tokens
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
HANGUL_FLAG = "<hangul>"


class EnglishBPETokenizer:
    """Wraps SentencePiece model for source English sentences."""

    def __init__(self, model_path: str | Path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(model_path))
        self.pad_id = self.sp.pad_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.unk_id = self.sp.unk_id()
        self.vocab_size = self.sp.get_piece_size()

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        ids = self.sp.encode(text, out_type=int)
        if add_special_tokens:
            return [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        clean_ids = [
            i for i in ids if i not in (self.pad_id, self.bos_id, self.eos_id, self.unk_id)
        ]
        return self.sp.decode(clean_ids)


class ThreeHotSeq2SeqTokenizer:
    """3-lane triplet tokenizer matching Cognetta et al. (EACL 2023).

    Korean syllables are represented as (initial, medial, final) triplets.
    Non-Korean characters and special tokens are represented as (token_id, PAD, PAD).
    """

    def __init__(self, non_ko_vocab: Optional[List[str]] = None):
        self.special_tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
        self.pad_id = 0
        self.sos_id = 1
        self.eos_id = 2
        self.unk_id = 3

        # Lane 0: Specials (0..3) + Choseong (4..22) + Non-Korean chars (23..)
        self.choseong_list = list(CHOSEONG)
        self.non_ko_chars = sorted(list(set(non_ko_vocab or [])))

        self.lane0_tokens = self.special_tokens + self.choseong_list + self.non_ko_chars
        self.lane0_c2i = {c: i for i, c in enumerate(self.lane0_tokens)}
        self.lane0_i2c = self.lane0_tokens

        # Lane 1: PAD (0) + Jungseong (1..21)
        self.jungseong_list = list(JUNGSEONG)
        self.lane1_tokens = [PAD_TOKEN] + self.jungseong_list
        self.lane1_c2i = {c: i for i, c in enumerate(self.lane1_tokens)}
        self.lane1_i2c = self.lane1_tokens

        # Lane 2: PAD/No batchim (0) + Jongseong (1..27)
        self.jongseong_list = list(JONGSEONG)  # JONGSEONG[0] is already PAD in hangul_factorizer
        self.lane2_tokens = self.jongseong_list
        self.lane2_c2i = {c: i for i, c in enumerate(self.lane2_tokens)}
        self.lane2_i2c = self.lane2_tokens

        self.vocab_sizes = [len(self.lane0_tokens), len(self.lane1_tokens), len(self.lane2_tokens)]

    @staticmethod
    def is_hangul_syllable(char: str) -> bool:
        return len(char) == 1 and S_BASE <= ord(char) < S_BASE + S_COUNT

    @staticmethod
    def decompose(char: str) -> Tuple[int, int, int]:
        s = ord(char) - S_BASE
        return s // N_COUNT, (s % N_COUNT) // T_COUNT, s % T_COUNT

    def encode(self, text: str, add_special_tokens: bool = True) -> List[Tuple[int, int, int]]:
        triplets: List[Tuple[int, int, int]] = []
        if add_special_tokens:
            triplets.append((self.sos_id, 0, 0))

        for char in text:
            if self.is_hangul_syllable(char):
                l, v, t = self.decompose(char)
                id0 = 4 + l  # choseong
                id1 = 1 + v  # jungseong
                id2 = t      # jongseong
                triplets.append((id0, id1, id2))
            else:
                id0 = self.lane0_c2i.get(char, self.unk_id)
                triplets.append((id0, 0, 0))

        if add_special_tokens:
            triplets.append((self.eos_id, 0, 0))
        return triplets

    def decode(self, triplets: Sequence[Tuple[int, int, int]]) -> str:
        chars = []
        for id0, id1, id2 in triplets:
            if id0 in (self.pad_id, self.sos_id, self.eos_id):
                continue
            if 4 <= id0 < 4 + len(self.choseong_list) and id1 > 0:
                l = id0 - 4
                v = id1 - 1
                t = id2 if 0 <= id2 < len(self.jongseong_list) else 0
                char = chr(S_BASE + (l * V_COUNT + v) * T_COUNT + t)
                chars.append(char)
            else:
                if 0 <= id0 < len(self.lane0_i2c):
                    tok = self.lane0_i2c[id0]
                    if tok not in self.special_tokens:
                        chars.append(tok)
        return "".join(chars)


class HangulFactorizerSeq2SeqTokenizer:
    """23-lane factorized tokenizer for Seq2Seq translation.

    Lane 0 encodes: <pad>, <sos>, <eos>, <unk>, <hangul>, followed by non-Korean characters.
    Lanes 1..22 encode the phonetic, articulatory, and structural features from LANES[1:23].
    """

    def __init__(self, non_ko_vocab: Optional[List[str]] = None):
        self.factorizer = HangulFactorizedTokenizer(use_pos=False)
        self.pad_id = 0
        self.sos_id = 1
        self.eos_id = 2
        self.unk_id = 3
        self.hangul_flag_id = 4

        self.special_tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN, HANGUL_FLAG]
        self.non_ko_chars = sorted(list(set(non_ko_vocab or [])))

        # Lane 0
        self.lane0_tokens = self.special_tokens + self.non_ko_chars
        self.lane0_c2i = {c: i for i, c in enumerate(self.lane0_tokens)}
        self.lane0_i2c = self.lane0_tokens

        # Lanes 1..22 correspond to LANES[1:23]
        self.factor_lanes = LANES[1:23]
        self.lane_value_to_id = [self.factorizer.value_to_id[i] for i in range(1, 23)]
        self.lane_id_to_value = [self.factorizer.id_to_value[i] for i in range(1, 23)]

        self.vocab_sizes = [len(self.lane0_tokens)] + [
            len(lane.values) for lane in self.factor_lanes
        ]

    def encode(self, text: str, add_special_tokens: bool = True) -> List[List[int]]:
        seq: List[List[int]] = []
        if add_special_tokens:
            seq.append([self.sos_id] + [0] * 22)

        for char in text:
            if self.factorizer.is_hangul_syllable(char):
                lane0_val = self.hangul_flag_id
                indices = self.factorizer.encode_char(char)
                factor_indices = indices[1:23]  # lanes 1..22
                seq.append([lane0_val] + list(factor_indices))
            else:
                lane0_val = self.lane0_c2i.get(char, self.unk_id)
                seq.append([lane0_val] + [0] * 22)

        if add_special_tokens:
            seq.append([self.eos_id] + [0] * 22)
        return seq

    def decode(self, sequence: Sequence[Sequence[int]]) -> str:
        chars = []
        for step in sequence:
            lane0 = step[0]
            if lane0 in (self.pad_id, self.sos_id, self.eos_id):
                continue
            if lane0 == self.hangul_flag_id:
                # Reconstruct syllable from choseong (lane 1), jungseong (lane 2), jongseong (lane 3)
                l = step[1] - 1 if step[1] > 0 else 0
                v = step[2] - 1 if step[2] > 0 else 0
                t = step[3] if 0 <= step[3] < len(JONGSEONG) else 0
                l = min(max(0, l), len(CHOSEONG) - 1)
                v = min(max(0, v), len(JUNGSEONG) - 1)
                char = chr(S_BASE + (l * V_COUNT + v) * T_COUNT + t)
                chars.append(char)
            else:
                if 0 <= lane0 < len(self.lane0_i2c):
                    tok = self.lane0_i2c[lane0]
                    if tok not in self.special_tokens:
                        chars.append(tok)
        return "".join(chars)


def build_non_korean_vocab(texts: Sequence[str]) -> List[str]:
    """Collects all unique non-Hangul characters from a corpus."""
    non_ko = set()
    for text in texts:
        for char in text:
            if not HangulFactorizedTokenizer.is_hangul_syllable(char):
                non_ko.add(char)
    return sorted(list(non_ko))
