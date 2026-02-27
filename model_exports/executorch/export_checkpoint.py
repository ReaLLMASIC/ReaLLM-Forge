"""Command line interface for exporting nanoGPT checkpoints to ExecuTorch."""

from __future__ import annotations

import argparse
from pathlib import Path

from .exporter import ExportConfig, export_checkpoint_to_pte


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="Path to the ckpt.pt file produced by training.")
    parser.add_argument(
        "--pte-path",
        help=(
            "Destination for the generated .pte file. If omitted, the exporter writes to "
            "<ckpt_dir>/executorch/<ckpt_stem>.pte"
        ),
    )
    parser.add_argument(
        "--delegate",
        default="none",
        choices=["none", "xnnpack"],
        help="ExecuTorch delegate to target during export.",
    )
    parser.add_argument(
        "--generate-etrecord",
        action="store_true",
        help="Generate an ETRecord artifact alongside the .pte file.",
    )
    parser.add_argument(
        "--smoke-test-tokens",
        type=int,
        default=0,
        help="If >0, run a random-token smoke test against the exported program.",
    )
    parser.add_argument(
        "--smoke-test-prompt",
        help="Optional prompt to evaluate with the exported program (requires --tokenizer-vocab).",
    )
    parser.add_argument(
        "--tokenizer-vocab",
        type=Path,
        help="Path to a vocab.json file compatible with the model. Required for prompt smoke tests.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=32,
        help="Maximum number of tokens to request during smoke tests.",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Skip writing an export metadata JSON alongside the .pte file.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    ckpt_path = Path(args.ckpt)
    if args.pte_path:
        pte_path = Path(args.pte_path)
    else:
        default_dir = ckpt_path.parent / "executorch"
        default_dir.mkdir(parents=True, exist_ok=True)
        pte_path = default_dir / f"{ckpt_path.stem}.pte"

    config = ExportConfig(
        delegate=args.delegate,
        generate_etrecord=args.generate_etrecord,
        smoke_test_tokens=args.smoke_test_tokens,
        smoke_test_prompt=args.smoke_test_prompt,
        tokenizer_path=args.tokenizer_vocab,
        max_output_tokens=args.max_output_tokens,
        metadata=not args.no_metadata,
    )

    export_checkpoint_to_pte(ckpt_path, pte_path, config)
    print(f"[executorch] Exported program written to {pte_path}")


if __name__ == "__main__":
    main()
