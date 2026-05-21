from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The lightweight test environment may not have transformers installed. The app
# only needs these names at import time; model loading is not used here.
if "transformers" not in sys.modules:
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = object
    fake_transformers.AutoTokenizer = object
    fake_transformers.PreTrainedTokenizerBase = object
    sys.modules["transformers"] = fake_transformers

from app.model_service import TokenInfo, search_tokens  # noqa: E402


@dataclass
class FakeAssets:
    token_infos: list[TokenInfo]


def make_assets() -> FakeAssets:
    return FakeAssets(
        token_infos=[
            TokenInfo(token_id=0, raw="<0xF9>", display="<0xF9>", normalized="<0xf9> 0xf9 f9"),
            TokenInfo(token_id=1, raw="<0xFA>", display="<0xFA>", normalized="<0xfa> 0xfa fa"),
            TokenInfo(token_id=2, raw="▁F9", display=" F9", normalized="▁f9 f9"),
        ]
    )


def test_literal_search_uses_byte_alias_contents() -> None:
    results = search_tokens(make_assets(), "F9")
    assert [item.token_id for item in results] == [0, 2]


def test_pattern_characters_are_treated_literally() -> None:
    results = search_tokens(make_assets(), "^F9$")
    assert results == []


def test_blank_search_returns_all_tokens_without_limit() -> None:
    results = search_tokens(make_assets(), "")
    assert [item.token_id for item in results] == [0, 1, 2]
