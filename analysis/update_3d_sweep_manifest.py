#!/usr/bin/env python3
"""Rebuild the 3D trajectory sweep manifest from all completed JSON runs."""

import argparse
import json
from pathlib import Path


def build_manifest(runs_dir: Path) -> dict:
    entries = []
    for path in sorted(runs_dir.glob("dim-*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        projection = payload.get("projection", {})
        entries.append({
            "name": path.stem,
            "file": f"runs/{path.name}",
            "embedding_dim": projection.get("input_dimensions"),
            "projection": projection.get("method"),
            "trained_tokens": len(payload.get("trained_tokens", [])),
            "held_out_tokens": len(payload.get("unseen_tokens", [])),
            "fixed_norm": payload.get("fixed_norm"),
            "wte_weight_tying": payload.get("wte_weight_tying", True),
            "dropped_tokens": len(payload.get("dropped_tokens", [])),
            "dropout_iteration": payload.get("dropout_iteration"),
            "final_iteration": payload.get("frames", [{}])[-1].get("iteration"),
            "transition_mode": payload.get("transition_mode"),
            "transition_iterations": payload.get("transition_iterations", []),
            "duty_cycle": payload.get("duty_cycle"),
            "optimizer": payload.get("optimizer", {}),
        })
    return {"runs": entries}


def update_manifest(runs_dir: Path) -> Path:
    """Atomically publish a manifest so browser refreshes never see partial JSON."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    destination = runs_dir / "manifest.json"
    temporary = runs_dir / ".manifest.json.tmp"
    temporary.write_text(json.dumps(build_manifest(runs_dir), indent=2), encoding="utf-8")
    temporary.replace(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("report/threejs/digits-3d/runs"))
    args = parser.parse_args()
    destination = update_manifest(args.runs_dir)
    count = len(json.loads(destination.read_text(encoding="utf-8"))["runs"])
    print(f"Updated {destination} with {count} completed runs")


if __name__ == "__main__":
    main()
