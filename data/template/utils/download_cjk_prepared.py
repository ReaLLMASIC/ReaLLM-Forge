"""Download selected, checksum-verified TED tokenized datasets for training."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile

from huggingface_hub import hf_hub_download, snapshot_download


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_files(manifest, runs):
    # The charBPE upload uses a flat manifest and a different directory prefix.
    if isinstance(manifest, list):
        files = []
        for entry in manifest:
            path = PurePosixPath(entry["path"])
            if (path.is_absolute() or ".." in path.parts or len(path.parts) != 3
                    or path.parts[0] != "cjk_ipa_charbpe"):
                raise ValueError(f"Invalid manifest path: {path}")
            files.append(dict(entry, source_path=entry["path"],
                              path=str(PurePosixPath("data", *path.parts[1:]))))
        manifest = dict(datasets={PurePosixPath(e["path"]).parts[1] for e in files},
                        files=files)
    missing = set(runs) - set(manifest["datasets"])
    if missing:
        raise ValueError(
            f"Prepared package does not contain: {', '.join(sorted(missing))}. "
            "Use prepare/all for local tokenization or select a matching prepared package."
        )
    entries = []
    for entry in manifest["files"]:
        path = PurePosixPath(entry["path"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Invalid manifest path: {path}")
        if len(path.parts) == 3 and path.parts[:1] == ("data",) and path.parts[1] in runs:
            entries.append(entry)
    for run in runs:
        required = {f"data/{run}/{name}" for name in ("train.bin", "val.bin", "meta.pkl")}
        if not required.issubset({entry["path"] for entry in entries}):
            raise ValueError(f"Manifest is missing required training files for {run}")
    return entries


def source_path(entry):
    return entry.get("source_path", entry["path"])


def matches(path, entry):
    return (path.is_file() and path.stat().st_size == entry["bytes"]
            and entry["bytes"] > 0 and sha256(path) == entry["sha256"])


def download_prepared(repo_id, revision, subdir, data_root, runs):
    subdir = subdir.strip("/")
    # Resolve main/tags once so the manifest and binaries come from one commit.
    manifest_path = Path(hf_hub_download(
        repo_id=repo_id, repo_type="dataset", revision=revision,
        filename=f"{subdir}/manifest.json"))
    pinned_revision = manifest_path.parents[len(PurePosixPath(subdir).parts)].name
    print(f"Prepared package: {repo_id}@{pinned_revision}/{subdir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = selected_files(manifest, runs)
    pending = []
    for entry in entries:
        target = data_root / PurePosixPath(entry["path"]).relative_to("data")
        if matches(target, entry):
            continue
        if target.exists():
            raise ValueError(f"Existing file differs from prepared package: {target}. "
                             "Choose a fresh CJK_DATA_ROOT or move the file aside.")
        pending.append(entry)
    if not pending:
        print("Selected prepared files already verified; skipping download.")
        return pinned_revision
    snapshot = Path(snapshot_download(
        repo_id=repo_id, repo_type="dataset", revision=pinned_revision,
        allow_patterns=[f"{subdir}/{source_path(entry)}" for entry in pending]))
    # Verify the whole selection before installing any files.
    for entry in pending:
        source = snapshot / subdir / source_path(entry)
        if not matches(source, entry):
            raise ValueError(f"Size/SHA256 mismatch: {source}")
    for entry in pending:
        target = data_root / PurePosixPath(entry["path"]).relative_to("data")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
                temporary = Path(handle.name)
                with (snapshot / subdir / source_path(entry)).open("rb") as source:
                    shutil.copyfileobj(source, handle)
            temporary.replace(target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        print(target)
    return pinned_revision


def download_comparison(repo_id, revision, subdir, charbpe_subdir, data_root, runs):
    """Combine baseline and charBPE packages from the same Hub commit."""
    baseline = [run for run in runs if not run.endswith("_char_bpe")]
    bpe = [run for run in runs if run.endswith("_char_bpe")]
    if baseline:
        revision = download_prepared(repo_id, revision, subdir, data_root, baseline)
    if bpe:
        download_prepared(repo_id, revision, charbpe_subdir, data_root, bpe)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="xinyixuu/ted-2020")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--subdir", default="cjk_char_byte_fallback_prepared")
    parser.add_argument("--charbpe-subdir", help="Separate package for char_bpe runs")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    args = parser.parse_args()
    try:
        if args.charbpe_subdir:
            download_comparison(args.repo_id, args.revision, args.subdir,
                                args.charbpe_subdir, args.data_root, args.runs)
        else:
            download_prepared(args.repo_id, args.revision, args.subdir, args.data_root, args.runs)
    except ValueError as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
