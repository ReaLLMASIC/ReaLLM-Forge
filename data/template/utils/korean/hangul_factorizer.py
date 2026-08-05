#!/usr/bin/env python3
"""Utilities for factorizing modern Korean Hangul syllables into 23 or 24 lanes.

The tokenizer keeps non-decomposable characters losslessly in the companion
character stream/metadata while Hangul syllables are represented by fixed,
per-lane categorical ids suitable for multicontext training.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

S_BASE = 0xAC00
L_BASE = 0x1100
V_BASE = 0x1161
T_BASE = 0x11A7
L_COUNT = 19
V_COUNT = 21
T_COUNT = 28
N_COUNT = V_COUNT * T_COUNT
S_COUNT = L_COUNT * N_COUNT
PAD = "PAD"
NON_HANGUL = "NON_HANGUL"
PUA_BASE = 0xE000

CHOSEONG = ["G", "GG", "N", "D", "DD", "R", "M", "B", "BB", "S", "SS", "NG", "J", "JJ", "CH", "K", "T", "P", "H"]
JUNGSEONG = ["A", "AE", "YA", "YAE", "EO", "E", "YEO", "YE", "O", "WA", "WAE", "OE", "YO", "U", "WEO", "WE", "WI", "YU", "EU", "YI", "I"]
JONGSEONG = [PAD, "G", "GG", "GS", "N", "NJ", "NH", "D", "R", "RG", "RM", "RB", "RS", "RT", "RP", "RH", "M", "B", "BS", "S", "SS", "NG", "J", "CH", "K", "T", "P", "H"]

VOWEL_COMPONENTS = {
    "A": ("A", PAD), "AE": ("A", "I"), "YA": ("YA", PAD), "YAE": ("YA", "I"),
    "EO": ("EO", PAD), "E": ("EO", "I"), "YEO": ("YEO", PAD), "YE": ("YEO", "I"),
    "O": ("O", PAD), "WA": ("O", "A"), "WAE": ("O", "AE"), "OE": ("O", "I"), "YO": ("YO", PAD),
    "U": ("U", PAD), "WEO": ("U", "EO"), "WE": ("U", "E"), "WI": ("U", "I"), "YU": ("YU", PAD),
    "EU": ("EU", PAD), "YI": ("EU", "I"), "I": ("I", PAD),
}
FINAL_COMPONENTS = {
    PAD: (PAD, PAD, PAD), "G": ("G", PAD, PAD), "GG": ("G", "G", PAD), "GS": ("G", "S", PAD),
    "N": ("N", PAD, PAD), "NJ": ("N", "J", PAD), "NH": ("N", "H", PAD), "D": ("D", PAD, PAD),
    "R": ("R", PAD, PAD), "RG": ("R", "G", PAD), "RM": ("R", "M", PAD), "RB": ("R", "B", PAD),
    "RS": ("R", "S", PAD), "RT": ("R", "T", PAD), "RP": ("R", "P", PAD), "RH": ("R", "H", PAD),
    "M": ("M", PAD, PAD), "B": ("B", PAD, PAD), "BS": ("B", "S", PAD), "S": ("S", PAD, PAD),
    "SS": ("S", "S", PAD), "NG": ("NG", PAD, PAD), "J": ("J", PAD, PAD), "CH": ("CH", PAD, PAD),
    "K": ("K", PAD, PAD), "T": ("T", PAD, PAD), "P": ("P", PAD, PAD), "H": ("H", PAD, PAD),
}

POS_TAGS = [
    "NNG", "NNP", "NNB", "NR", "NP", 
    "VV", "VA", "VX", "VCP", "VCN", 
    "MM", "MAG", "MAJ", "IC", 
    "JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC", 
    "EP", "EF", "EC", "ETN", "ETM", 
    "XPN", "XSN", "XSV", "XSA", "XR", 
    "SF", "SP", "SS", "SE", "SO", "SW",
    "SL", "SH", "SN", 
    "W_URL", "W_EMAIL", "W_HASHTAG", "W_MENTION"
]

@dataclass(frozen=True)
class Lane:
    name: str
    values: Sequence[str]
    description: str

LANES: List[Lane] = [
    Lane("script", [PAD, "HANGUL", NON_HANGUL], "Character class."),
    Lane("choseong", [PAD, *CHOSEONG], "Initial consonant index."),
    Lane("jungseong", [PAD, *JUNGSEONG], "Medial vowel index."),
    Lane("jongseong", JONGSEONG, "Final consonant index; PAD means no batchim."),
    Lane("jung_base1", [PAD, "A", "YA", "EO", "YEO", "O", "YO", "U", "YU", "EU", "I"], "First vowel component."),
    Lane("jung_base2", [PAD, "A", "AE", "EO", "E", "I"], "Second vowel component for compound vowels."),
    Lane("jung_has_w", [PAD, "0", "1"], "Vowel contains w/glide composition."),
    Lane("jung_has_y", [PAD, "0", "1"], "Vowel contains y/iotized onset."),
    Lane("jung_has_i", [PAD, "0", "1"], "Vowel contains an i element."),
    Lane("jong_base1", [PAD, *CHOSEONG], "First final-consonant component."),
    Lane("jong_base2", [PAD, *CHOSEONG], "Second final-consonant component."),
    Lane("jong_base3", [PAD, *CHOSEONG], "Reserved third final-consonant component."),
    Lane("choseong_tense", [PAD, "0", "1"], "Initial is tense/doubled."),
    Lane("choseong_aspirated", [PAD, "0", "1"], "Initial is aspirated."),
    Lane("choseong_nasal_liquid", [PAD, "0", "1"], "Initial is nasal/liquid/ng."),
    Lane("choseong_place", [PAD, "velar", "coronal", "labial", "glottal", "null"], "Coarse initial place."),
    Lane("jung_height", [PAD, "low", "mid", "high"], "Coarse vowel height."),
    Lane("jung_backness", [PAD, "front", "central", "back"], "Coarse vowel backness."),
    Lane("jung_round", [PAD, "0", "1"], "Rounded vowel family."),
    Lane("jong_complex", [PAD, "0", "1"], "Final consonant is a cluster/double."),
    Lane("has_batchim", [PAD, "0", "1"], "Syllable has a final consonant."),
    Lane("syllable_index_mod", [PAD, *[str(i) for i in range(28)]], "Hangul syllable index modulo 28."),
    Lane("codepoint_mod", [PAD, *[str(i) for i in range(64)]], "Unicode codepoint modulo 64, for tie-breaking diagnostics."),
]
LANE_NAMES = [lane.name for lane in LANES]

POS_LANE = Lane("pos", [PAD, "UNK", *POS_TAGS], "Part-of-speech tag from Kiwi.")
LANES_WITH_POS: List[Lane] = [*LANES, POS_LANE]
LANE_NAMES_WITH_POS = [lane.name for lane in LANES_WITH_POS]

class HangulFactorizedTokenizer:
    lanes = LANES
    lane_names = LANE_NAMES

    def __init__(self, use_pos: bool = False) -> None:
        self.use_pos = use_pos
        self.lanes = LANES_WITH_POS if use_pos else LANES
        self.lane_names = LANE_NAMES_WITH_POS if use_pos else LANE_NAMES
        self.value_to_id = [{v: i for i, v in enumerate(lane.values)} for lane in self.lanes]
        self.id_to_value = [list(lane.values) for lane in self.lanes]
        self._encode_cache: Dict[Any, tuple[int, ...]] = {}
        self.kiwi = None

    @staticmethod
    def is_hangul_syllable(char: str) -> bool:
        return len(char) == 1 and S_BASE <= ord(char) < S_BASE + S_COUNT

    @staticmethod
    def _decompose(char: str) -> tuple[int, int, int]:
        s = ord(char) - S_BASE
        return s // N_COUNT, (s % N_COUNT) // T_COUNT, s % T_COUNT

    def _features(self, char: str, pos_tag: str = "UNK") -> List[str]:
        if self.use_pos and pos_tag not in POS_TAGS:
            pos_tag = "UNK"
        if not self.is_hangul_syllable(char):
            base_features = [NON_HANGUL] + [PAD] * 22
            return base_features + [pos_tag] if self.use_pos else base_features
        l, v, t = self._decompose(char)
        cho, jung, jong = CHOSEONG[l], JUNGSEONG[v], JONGSEONG[t]
        vb1, vb2 = VOWEL_COMPONENTS[jung]
        jb1, jb2, jb3 = FINAL_COMPONENTS[jong]
        place = "null" if cho == "NG" else ("velar" if cho in {"G","GG","K"} else "labial" if cho in {"M","B","BB","P"} else "glottal" if cho == "H" else "coronal")
        height = "low" if "A" in jung else "high" if jung in {"O","YO","U","YU","EU","YI","I","WI"} else "mid"
        back = "front" if jung in {"AE","E","YAE","YE","OE","WE","WI","I"} else "back" if jung in {"O","WA","WAE","YO","U","WEO","YU"} else "central"
        base_features = ["HANGUL", cho, jung, jong, vb1, vb2, str(int(jung in {"WA","WAE","OE","WEO","WE","WI"})), str(int(jung.startswith("Y"))), str(int("I" in (vb1, vb2) or jung in {"AE","E","OE","WE","WI","YI","I"})), jb1, jb2, jb3, str(int(cho in {"GG","DD","BB","SS","JJ"})), str(int(cho in {"CH","K","T","P","H"})), str(int(cho in {"N","R","M","NG"})), place, height, back, str(int(jung in {"O","WA","WAE","OE","YO","U","WEO","WE","WI","YU"})), str(int(t in {2,3,5,6,9,10,11,12,13,14,15,18,20})), str(int(t != 0)), str(t), str(ord(char) % 64)]
        return base_features + [pos_tag] if self.use_pos else base_features

    def encode_char(self, char: str, pos_tag: str = "UNK", return_tags: bool = False) -> List[Any]:
        if self.use_pos and pos_tag not in POS_TAGS:
            pos_tag = "UNK"
        cache_key = (char, pos_tag) if self.use_pos else char
        cached = self._encode_cache.get(cache_key)
        if cached is None:
            vals = self._features(char, pos_tag=pos_tag)
            cached = tuple(self.value_to_id[i].get(v, 0) for i, v in enumerate(vals))
            self._encode_cache[cache_key] = cached
        if return_tags:
            return [self.id_to_value[i][idx] for i, idx in enumerate(cached)]
        return list(cached)

    def encode_text(self, text: str, return_tags: bool = False) -> List[Dict[str, Any]]:
        """Translates a text string into a sequence of factorized arrays, contextualized with POS if enabled."""
        char_idx_to_pos = {}
        if self.use_pos:
            try:
                from kiwipiepy import Kiwi
                if self.kiwi is None:
                    self.kiwi = Kiwi()
            except ImportError:
                self.kiwi = None
                
            if self.kiwi is not None:
                tokens = self.kiwi.tokenize(text)
                for token in tokens:
                    for i in range(token.start, token.start + token.len):
                        char_idx_to_pos[i] = token.tag
                        
        encoded_sequence = []
        for i, char in enumerate(text):
            pos_tag = char_idx_to_pos.get(i, "UNK")
            if pos_tag not in POS_TAGS:
                pos_tag = "UNK"
                
            encoded = self.encode_char(char, pos_tag=pos_tag, return_tags=return_tags)
            encoded_sequence.append({"char": char, "indices": encoded})
            
        return encoded_sequence

    def decode_indices(self, indices: Sequence[int]) -> str:
        vals = [self.id_to_value[i][idx] if 0 <= idx < len(self.id_to_value[i]) else PAD for i, idx in enumerate(indices[:23])]
        if vals[0] != "HANGUL":
            return ""
        try:
            l, v, t = CHOSEONG.index(vals[1]), JUNGSEONG.index(vals[2]), JONGSEONG.index(vals[3])
        except ValueError:
            return ""
        return chr(S_BASE + (l * V_COUNT + v) * T_COUNT + t)

    def token_for(self, lane_index: int, value_id: int) -> str:
        return chr(PUA_BASE + lane_index * 256 + value_id)

    def id_from_token_or_label(self, lane_index: int, token: str) -> int:
        token = token.strip()
        if len(token) == 1:
            code = ord(token) - PUA_BASE - lane_index * 256
            if 0 <= code < len(self.id_to_value[lane_index]):
                return code
        return self.value_to_id[lane_index].get(token, 0)

    def metadata_for_char(self, char: str, position: int, pos_tag: str = "UNK") -> Dict[str, Any]:
        ids = self.encode_char(char, pos_tag=pos_tag)
        return {"position": position, "char": char, "codepoint": f"U+{ord(char):04X}", "is_hangul": self.is_hangul_syllable(char), "lanes": {n: {"id": ids[i], "value": self.id_to_value[i][ids[i]]} for i, n in enumerate(self.lane_names)}}

    def lane_metadata(self) -> List[Dict[str, Any]]:
        return [{"index": i, "name": lane.name, "description": lane.description, "values": list(lane.values)} for i, lane in enumerate(self.lanes)]


class HangulPosFactorizedTokenizer(HangulFactorizedTokenizer):
    """Hangul factorized tokenizer equipped with the 24th part-of-speech (POS) lane."""
    lanes = LANES_WITH_POS
    lane_names = LANE_NAMES_WITH_POS

    def __init__(self, use_pos: bool = True) -> None:
        super().__init__(use_pos=use_pos)


HangulFactorizedPosTokenizer = HangulPosFactorizedTokenizer


def dump_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
