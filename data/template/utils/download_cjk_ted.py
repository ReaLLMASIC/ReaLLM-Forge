"""Download the original TED texts used by the CJK tokenizer experiments."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="xinyixuu/ted-2020")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--files", nargs="+", choices=("ja.txt", "ko.txt", "zh_cn.txt"),
                        default=["ja.txt", "ko.txt", "zh_cn.txt"])
    args = parser.parse_args()
    snapshot_download(repo_id=args.repo_id, repo_type="dataset", revision=args.revision,
                      local_dir=str(args.local_dir), allow_patterns=args.files)
    for filename in args.files:
        path = args.local_dir / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"Missing or empty dataset file: {path}")
        print(path.resolve())


if __name__ == "__main__":
    main()
