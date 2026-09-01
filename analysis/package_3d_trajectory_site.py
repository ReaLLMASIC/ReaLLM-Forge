#!/usr/bin/env python3
"""Package completed 3D trajectory runs as a GitHub Pages static site."""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from update_3d_sweep_manifest import update_manifest


SOURCE_DIR = Path("report/threejs/digits-3d")


def package_site(source_dir: Path, output_dir: Path) -> int:
    runs_dir = source_dir / "runs"
    manifest_path = update_manifest(runs_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest["runs"]:
        raise ValueError(f"no completed trajectory JSON files found in {runs_dir}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "runs").mkdir(parents=True)

    selector = (source_dir / "index.html").read_text(encoding="utf-8")
    viewer = (source_dir / "viewer.html").read_text(encoding="utf-8")
    (output_dir / "index.html").write_text(selector, encoding="utf-8")
    (output_dir / "viewer.html").write_text(viewer, encoding="utf-8")
    (output_dir / ".nojekyll").write_text("", encoding="utf-8")

    shutil.copy2(manifest_path, output_dir / "runs/manifest.json")
    for run in manifest["runs"]:
        source = source_dir / run["file"]
        shutil.copy2(source, output_dir / run["file"])

    readme = """# 3D token trajectory static site

This directory is generated. Serve it locally with `python3 -m http.server -d .`
or publish the directory as a GitHub Pages deployment artifact.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    return len(manifest["runs"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("dist/digits-3d-site"))
    args = parser.parse_args()
    count = package_site(args.source_dir, args.output_dir)
    print(f"Packaged {count} completed runs into {args.output_dir}")


if __name__ == "__main__":
    main()
