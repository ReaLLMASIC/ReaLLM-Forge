#!/usr/bin/env python3
"""Self-contained Hanzi → IDS → bytes → IDS → Hanzi demonstration.

This tiny mapping is deliberately a demonstration, not a claim of corpus
coverage.  Use the dataset loaders and ``scripts/coverage_utf.py`` for a real
decomposition source.

Run from a checkout without installing the package::

    python examples/roundtrip.py
    python examples/roundtrip.py 汉 森
"""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_TREE = PROJECT_ROOT / "src"
if SOURCE_TREE.is_dir():
    sys.path.insert(0, str(SOURCE_TREE))

from hanzi_factor.codec import HanziCodec  # noqa: E402
from hanzi_factor.ids import format_ids  # noqa: E402


DEMO_DECOMPOSITIONS = {
    "汉": "⿰氵又",
    "国": "⿴囗玉",
    "语": "⿰讠⿱五口",
    "清": "⿰氵青",
    "森": "⿱木⿰木木",
}

# 青 is a reusable component, not a Hanzi identity lookup.  Expanding it makes
# 清 fully structural: ⿰氵⿱龶月.  The binary decoder needs the same component
# codec profile whenever a compact component reference is used.  Framed
# HanziCodec bytes bind both this component table and the reverse catalogue.
DEMO_COMPONENTS = {
    "青": "⿱龶月",
}


def requested_characters(arguments: list[str]) -> list[str]:
    """Allow either space-separated labels or a single string of labels."""

    if not arguments:
        return list(DEMO_DECOMPOSITIONS)
    return [character for argument in arguments for character in argument]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "characters",
        nargs="*",
        help="demo Hanzi to round-trip (default: 汉国语清森)",
    )
    args = parser.parse_args(argv)

    characters = requested_characters(args.characters)
    unknown = [char for char in characters if char not in DEMO_DECOMPOSITIONS]
    if unknown:
        parser.error(
            "not in the self-contained demo mapping: "
            + ", ".join(f"{char} (U+{ord(char):04X})" for char in unknown)
        )

    codec = HanziCodec(DEMO_DECOMPOSITIONS, DEMO_COMPONENTS)

    for character in characters:
        # Forward path: identity is used only to select its registered tree.
        canonical_ids = codec.to_ids(character)
        canonical_tree = codec.index.tree_for(character)
        encoded = codec.encode(character)
        unframed = codec.binary.encode_tree(canonical_tree, framed=False)
        payload_bits = codec.binary.payload_bit_length(canonical_tree)
        encoded_hex = encoded.hex()
        encoded_base64 = base64.b64encode(encoded).decode("ascii")

        # Reverse path: bytes yield a tree; only then does the reverse index
        # recover an identity.  Ambiguous graphical collisions would raise an
        # error instead of silently choosing one Hanzi.
        decoded_tree = codec.binary.decode_tree(encoded)
        decoded_ids = format_ids(decoded_tree)
        recovered = codec.index.lookup(decoded_tree)

        if decoded_ids != canonical_ids or recovered != character:
            raise AssertionError("round-trip invariant failed")

        print(f"{character}  U+{ord(character):04X}")
        print(f"  source IDS    : {DEMO_DECOMPOSITIONS[character]}")
        print(f"  canonical IDS : {canonical_ids}")
        print(f"  payload bits  : {payload_bits}")
        print(f"  unframed bytes: {len(unframed)} ({unframed.hex()})")
        print(f"  framed bytes  : {len(encoded)}")
        print(f"  profile hash  : {codec.profile_fingerprint.hex()[:16]}")
        print(f"  hex           : {encoded_hex}")
        print(f"  base64        : {encoded_base64}")
        print(f"  decoded tree  : {decoded_tree!r}")
        print(f"  decoded IDS   : {decoded_ids}")
        print(f"  reverse lookup: {recovered}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
