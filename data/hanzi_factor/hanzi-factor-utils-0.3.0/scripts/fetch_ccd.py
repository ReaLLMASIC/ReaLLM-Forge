#!/usr/bin/env python3
"""Fetch the pinned, external CCD JSON audit dataset.

The dataset is intentionally not bundled with hanzi-factor.  This helper
downloads the MIT-licensed npm snapshot and verifies its registry integrity
hash before extracting only ``package/ccd.json``.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import sys
import tarfile
import urllib.request
from pathlib import Path


VERSION = "0.1.0"
URL = (
    "https://registry.npmjs.org/chinese-characters-decomposition/-/"
    f"chinese-characters-decomposition-{VERSION}.tgz"
)
INTEGRITY_SHA512_B64 = (
    "aFIjWb090e6kkQGme7LDLrBX5YYXSMlMOiB0+Lbj+f5Xs//z7u265R317vCsWWjRwqhp"
    "Wh2YDwE+ltMNJth0WQ=="
)
MEMBER = "package/ccd.json"


def fetch(url: str = URL) -> bytes:
    """Return the verified tarball bytes."""

    request = urllib.request.Request(
        url,
        headers={"User-Agent": f"hanzi-factor-fetch/{VERSION}"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()

    actual = hashlib.sha512(payload).digest()
    expected = base64.b64decode(INTEGRITY_SHA512_B64)
    if actual != expected:
        raise ValueError(
            "download failed SHA-512 integrity verification; refusing to extract"
        )
    return payload


def extract_ccd(tarball: bytes) -> bytes:
    """Extract only the known JSON member without writing archive paths."""

    with tarfile.open(fileobj=io.BytesIO(tarball), mode="r:gz") as archive:
        member = archive.getmember(MEMBER)
        stream = archive.extractfile(member)
        if stream is None:
            raise ValueError(f"{MEMBER!r} is not a regular archive member")
        return stream.read()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path("data/ccd.json"),
        help="destination JSON path (default: data/ccd.json)",
    )
    parser.add_argument("--url", default=URL, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        ccd_json = extract_ccd(fetch(args.url))
        # Decode once so a corrupt/non-UTF-8 payload cannot be installed.
        ccd_json.decode("utf-8")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(ccd_json)
    except (OSError, ValueError, tarfile.TarError) as error:
        print(f"fetch_ccd: {error}", file=sys.stderr)
        return 1

    print(f"wrote {args.output} ({len(ccd_json):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
