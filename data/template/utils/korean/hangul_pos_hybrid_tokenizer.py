#!/usr/bin/env python3
"""Hangul Factorizer without Character Stream Overhead (24 Lanes Total).

This tokenizer splits text into surface segments:
1. Korean Hangul precomposed syllables ([가-힣]+): factorized into 24 categorical
   feature lanes (Lane 0: <hangul>, Lanes 1..22: phonetic/articulatory/structural
   features, Lane 23: part-of-speech tag from Kiwi).
2. Non-Korean segments ([^가-힣]+): tokenized using a Byte-Fallback BPE model into
   Lane 0, with Lanes 1..23 set to PAD.

This eliminates the 25th companion character stream completely while providing
100% exact lossless reconstruction for all text, full open-vocabulary byte fallback,
and superior sequence compression.
"""
from __future__ import annotations

import json
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import sentencepiece as spm
from kiwipiepy import Kiwi

# Import constants from standard hangul_factorizer
from hangul_factorizer import (
    CHOSEONG,
    FINAL_COMPONENTS,
    JONGSEONG,
    JUNGSEONG,
    L_COUNT,
    LANES,
    N_COUNT,
    PAD,
    POS_TAG_MAP,
    POS_TAGS_COARSE,
    POS_TAGS_FULL,
    S_BASE,
    S_COUNT,
    T_COUNT,
    V_COUNT,
    VOWEL_COMPONENTS,
    Lane,
)


@dataclass(frozen=True)
class HybridLane:
    name: str
    values: Sequence[str]
    description: str


class HangulHybridPosTokenizer:
    """24-lane hybrid tokenizer: Hangul factorizer + Byte-Fallback BPE in Lane 0."""

    def __init__(
        self,
        spm_model_path: str,
        use_pos: bool = True,
        pos_mode: str = "coarse",
    ) -> None:
        self.spm_model_path = spm_model_path
        self.sp = spm.SentencePieceProcessor()
        if not os.path.exists(spm_model_path):
            raise FileNotFoundError(f"SentencePiece model not found at: {spm_model_path}")
        self.sp.load(spm_model_path)

        self.hangul_id = self.sp.piece_to_id("<hangul>")
        if self.hangul_id == -1:
            raise ValueError("SentencePiece model must contain '<hangul>' user-defined symbol.")
        self.pad_id = self.sp.pad_id()

        self.use_pos = use_pos
        self.pos_mode = pos_mode.lower()
        self.pos_tags = POS_TAGS_FULL if self.pos_mode == "full" else POS_TAGS_COARSE

        # Build 24 lanes
        # Lane 0: script_bpe (BPE vocabulary with byte-fallback and <hangul>)
        bpe_values = [self.sp.id_to_piece(i) for i in range(self.sp.get_piece_size())]
        lane_0 = HybridLane(
            "script_bpe",
            bpe_values,
            "BPE subword/byte token for non-Korean; '<hangul>' for Hangul syllables.",
        )

        # Lanes 1..22: 22 factor lanes (from LANES[1:23])
        factor_lanes = [
            HybridLane(lane.name, lane.values, lane.description) for lane in LANES[1:23]
        ]

        # Lane 23: POS tag lane
        pos_lane = HybridLane(
            "pos",
            [PAD, "UNK", *self.pos_tags],
            f"Part-of-speech tag from Kiwi ({self.pos_mode}).",
        )

        self.lanes = [lane_0, *factor_lanes, pos_lane]
        self.lane_names = [l.name for l in self.lanes]
        self.value_to_id = [{v: i for i, v in enumerate(l.values)} for l in self.lanes]
        self.id_to_value = [list(l.values) for l in self.lanes]

        self.kiwi: Optional[Kiwi] = None
        self._encode_cache: Dict[Tuple[str, str], Tuple[int, ...]] = {}

    @staticmethod
    def is_hangul_syllable(char: str) -> bool:
        return len(char) == 1 and S_BASE <= ord(char) < S_BASE + S_COUNT

    @staticmethod
    def _decompose(char: str) -> Tuple[int, int, int]:
        s = ord(char) - S_BASE
        return s // N_COUNT, (s % N_COUNT) // T_COUNT, s % T_COUNT

    def _features(self, char: str) -> List[str]:
        l, v, t = self._decompose(char)
        cho, jung, jong = CHOSEONG[l], JUNGSEONG[v], JONGSEONG[t]
        vb1, vb2 = VOWEL_COMPONENTS[jung]
        jb1, jb2, jb3 = FINAL_COMPONENTS[jong]
        place = (
            "null"
            if cho == "NG"
            else (
                "velar"
                if cho in {"G", "GG", "K"}
                else (
                    "labial"
                    if cho in {"M", "B", "BB", "P"}
                    else "glottal" if cho == "H" else "coronal"
                )
            )
        )
        height = (
            "low"
            if "A" in jung
            else (
                "high"
                if jung in {"O", "YO", "U", "YU", "EU", "YI", "I", "WI"}
                else "mid"
            )
        )
        back = (
            "front"
            if jung in {"AE", "E", "YAE", "YE", "OE", "WE", "WI", "I"}
            else (
                "back"
                if jung in {"O", "WA", "WAE", "YO", "U", "WEO", "YU"}
                else "central"
            )
        )
        return [
            cho,
            jung,
            jong,
            vb1,
            vb2,
            str(int(jung in {"WA", "WAE", "OE", "WEO", "WE", "WI"})),
            str(int(jung.startswith("Y"))),
            str(int("I" in (vb1, vb2) or jung in {"AE", "E", "OE", "WE", "WI", "YI", "I"})),
            jb1,
            jb2,
            jb3,
            str(int(cho in {"GG", "DD", "BB", "SS", "JJ"})),
            str(int(cho in {"CH", "K", "T", "P", "H"})),
            str(int(cho in {"N", "R", "M", "NG"})),
            place,
            height,
            back,
            str(int(jung in {"O", "WA", "WAE", "OE", "YO", "U", "WEO", "WE", "WI", "YU"})),
            str(int(t in {2, 3, 5, 6, 9, 10, 11, 12, 13, 14, 15, 18, 20})),
            str(int(t != 0)),
            str(t),
            str(ord(char) % 64),
        ]

    def encode_text(
        self,
        text: str,
        char_pos_tags: Optional[Dict[int, str]] = None,
    ) -> List[List[int]]:
        """Encode string into a sequence of 24-element integer ID vectors."""
        if not text:
            return []

        # 1. Run Kiwi on entire text in context if POS tags not provided
        if char_pos_tags is None:
            if self.kiwi is None:
                self.kiwi = Kiwi()
            tokens = self.kiwi.tokenize(text)
            char_pos_tags = {}
            for token in tokens:
                tag = token.tag if self.pos_mode == "full" else POS_TAG_MAP.get(token.tag, token.tag)
                for idx in range(token.start, token.start + token.len):
                    char_pos_tags[idx] = tag

        # 2. Segment text into Korean syllables vs non-Korean chunks
        pattern = re.compile(r"([가-힣]+|[^가-힣]+)")
        segments = [m.group(0) for m in pattern.finditer(text)]

        encoded_steps: List[List[int]] = []
        cur_pos = 0

        for seg in segments:
            if re.match(r"^[가-힣]+$", seg):
                # Pure Korean syllable segment
                for ch in seg:
                    raw_tag = char_pos_tags.get(cur_pos, "UNK")
                    tag = raw_tag if raw_tag in self.pos_tags else "UNK"

                    cache_key = (ch, tag)
                    cached = self._encode_cache.get(cache_key)
                    if cached is None:
                        feats = self._features(ch)
                        step = [self.hangul_id]
                        for lane_idx, val in enumerate(feats, start=1):
                            step.append(self.value_to_id[lane_idx].get(val, 0))
                        step.append(self.value_to_id[23].get(tag, 1))  # 1 is UNK
                        cached = tuple(step)
                        self._encode_cache[cache_key] = cached

                    encoded_steps.append(list(cached))
                    cur_pos += 1
            else:
                # Non-Korean segment (Byte-Fallback BPE in Lane 0, PAD in Lanes 1..23)
                bpe_ids = self.sp.encode_as_ids(seg)
                for b_id in bpe_ids:
                    step = [b_id] + [0] * 23  # Index 0 is PAD for lanes 1..23
                    encoded_steps.append(step)
                cur_pos += len(seg)

        return encoded_steps

    def decode_sequence(self, steps: Sequence[Sequence[int]]) -> str:
        """Deterministically decode a sequence of 24-element vectors back into exact text."""
        out_tokens: List[str] = []
        bpe_buffer: List[int] = []

        def flush_bpe():
            if bpe_buffer:
                out_tokens.append(self.sp.decode_ids(bpe_buffer))
                bpe_buffer.clear()

        for step in steps:
            l0 = step[0]
            if l0 == self.hangul_id:
                flush_bpe()
                cho_val = self.id_to_value[1][step[1]]
                jung_val = self.id_to_value[2][step[2]]
                jong_val = self.id_to_value[3][step[3]]
                try:
                    l = CHOSEONG.index(cho_val)
                    v = JUNGSEONG.index(jung_val)
                    t = JONGSEONG.index(jong_val)
                    out_tokens.append(chr(S_BASE + (l * V_COUNT + v) * T_COUNT + t))
                except (ValueError, IndexError):
                    out_tokens.append("")
            else:
                bpe_buffer.append(l0)

        flush_bpe()
        return "".join(out_tokens)

    def lane_metadata(self) -> List[Dict[str, Any]]:
        return [
            {
                "index": i,
                "name": lane.name,
                "description": lane.description,
                "values": list(lane.values),
                "vocab_size": len(lane.values),
            }
            for i, lane in enumerate(self.lanes)
        ]

    def save_meta_pkls(self, output_dir: str | Path) -> None:
        """Write meta.pkl for each of the 24 lanes into output_dir/<lane_name>/meta.pkl."""
        out_root = Path(output_dir)
        for i, lane in enumerate(self.lanes):
            lane_dir = out_root / lane.name
            lane_dir.mkdir(parents=True, exist_ok=True)
            stoi = {v: idx for idx, v in enumerate(lane.values)}
            itos = {idx: v for idx, v in enumerate(lane.values)}
            meta = {
                "vocab_size": len(lane.values),
                "tokenizer": f"hangul_hybrid_lane_{i}_{lane.name}",
                "lane_index": i,
                "lane_name": lane.name,
                "stoi": stoi,
                "itos": itos,
            }
            with open(lane_dir / "meta.pkl", "wb") as f:
                pickle.dump(meta, f)


def train_sentencepiece_non_korean(
    input_file: str,
    output_prefix: str,
    vocab_size: int = 2048,
    max_chars: Optional[int] = None,
) -> str:
    """Train SentencePiece model on non-Korean text with byte-fallback and exact whitespace preservation."""
    print(f"Extracting non-Korean segments from {input_file}...")
    pattern = re.compile(r"[^가-힣]+")
    non_korean_chunks = []
    total_chars = 0

    with open(input_file, "r", encoding="utf-8", errors="replace") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            for match in pattern.finditer(chunk):
                s = match.group(0)
                if s.strip():
                    non_korean_chunks.append(s)
                    total_chars += len(s)
            if max_chars and total_chars >= max_chars:
                break

    temp_input = f"{output_prefix}_input_temp.txt"
    with open(temp_input, "w", encoding="utf-8") as f:
        f.write("\n".join(non_korean_chunks))

    print(f"Training SentencePiece model (vocab_size={vocab_size}) on {len(non_korean_chunks):,} chunks...")
    spm.SentencePieceTrainer.train(
        input=temp_input,
        model_prefix=output_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        byte_fallback=True,
        character_coverage=1.0,
        add_dummy_prefix=False,
        remove_extra_whitespaces=False,
        normalization_rule_name="identity",
        hard_vocab_limit=False,
        pad_id=0,
        unk_id=1,
        bos_id=-1,
        eos_id=-1,
        user_defined_symbols=["<hangul>"],
    )

    if os.path.exists(temp_input):
        os.remove(temp_input)

    model_path = f"{output_prefix}.model"
    print(f"SentencePiece model saved to {model_path}")
    return model_path
