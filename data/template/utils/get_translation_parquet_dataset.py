"""Utilities for flattening translation columns in Parquet datasets."""
from __future__ import annotations

import argparse
import json
import os
from typing import Iterable, Sequence, Tuple

from get_parquet_dataset import convert_to_json, download_file, find_parquet_links


def emit_translation_items(
    json_path: str,
    output_path: str,
    language_prefixes: Sequence[Tuple[str, str]],
) -> None:
    """Emit flattened translation rows from ``json_path`` into ``output_path``.

    Parameters
    ----------
    json_path:
        Path to the JSON file produced from a Parquet shard.
    output_path:
        File where the flattened text should be appended.
    language_prefixes:
        Ordered collection of (language, prefix) tuples. Each translation entry
        writes one line per language using the associated prefix when the
        translation text is present.
    """
    if not language_prefixes:
        return

    with open(json_path, "r", encoding="utf-8") as handle:
        records = json.load(handle)

    if not isinstance(records, list):
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "a", encoding="utf-8") as out_handle:
        for record in records:
            translation = record.get("translation")
            if not isinstance(translation, dict):
                continue

            segments = []
            for language, prefix in language_prefixes:
                text = translation.get(language)
                if not text:
                    continue
                segments.append(f"{prefix}{text}")

            if segments:
                out_handle.write("\n".join(segments) + "\n\n")


def download_translation_dataset(
    url: str,
    output_text_file: str,
    language_prefixes: Sequence[Tuple[str, str]],
    append: bool = False,
) -> None:
    """Download, convert, and flatten translation datasets from ``url``.

    The function downloads all Parquet files advertised at ``url`` (typically a
    Hugging Face dataset folder), converts them to JSON if necessary, and emits
    flattened text records to ``output_text_file`` using the provided language
    prefixes.
    """
    parquet_links = find_parquet_links(url)
    download_dir = "./downloaded_parquets"
    json_dir = "./json_output"
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    if not append:
        open(output_text_file, "w", encoding="utf-8").close()

    for link in parquet_links:
        file_name = link.split("/")[-1].split("?")[0]
        parquet_path = os.path.join(download_dir, file_name)
        json_path = os.path.join(json_dir, file_name.replace(".parquet", ".json"))

        if not os.path.exists(parquet_path):
            download_file(link, parquet_path)

        convert_to_json(parquet_path, json_path)
        emit_translation_items(json_path, output_text_file, language_prefixes)



def parse_language_prefixes(prefix_args: Iterable[Tuple[str, str]]) -> Sequence[Tuple[str, str]]:
    """Validate and normalize CLI ``--prefix`` arguments."""
    prefixes: list[Tuple[str, str]] = []
    for language, prefix in prefix_args:
        if not language:
            raise ValueError("Language code for --prefix cannot be empty")
        prefixes.append((language, prefix))
    return prefixes


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download Europarl-style translation Parquet files and emit prefixed text."
        )
    )
    parser.add_argument(
        "--url",
        required=True,
        help="Dataset folder URL listing the Parquet shards (e.g. Hugging Face tree view).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="input.txt",
        help="Where to write the flattened text output.",
    )
    parser.add_argument(
        "--prefix",
        nargs=2,
        action="append",
        metavar=("LANG", "PREFIX"),
        required=True,
        help="Language/prefix pairs like --prefix bg 'BG: ' --prefix cs 'CS: '.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to the output file instead of overwriting it.",
    )
    args = parser.parse_args()

    language_prefixes = parse_language_prefixes(args.prefix)
    download_translation_dataset(
        args.url,
        args.output,
        language_prefixes,
        append=args.append,
    )


if __name__ == "__main__":
    main()
