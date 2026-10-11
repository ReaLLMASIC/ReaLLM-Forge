"""Build deterministic CJK sentence translation records from FLORES files."""

from .builder import BuildResult, build_sentence_pairs

__all__ = ["BuildResult", "build_sentence_pairs"]
