"""Check Hub downloads and actual shell-generated training arguments without a GPU."""

import importlib.util
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "download_cjk_ted", ROOT / "data/template/utils/download_cjk_ted.py")
downloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)
spec = importlib.util.spec_from_file_location(
    "download_cjk_prepared", ROOT / "data/template/utils/download_cjk_prepared.py")
prepared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepared)


class HubPipelineTests(unittest.TestCase):
    def test_prepared_download_checks_selection_hashes_and_reuses_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshots" / ("a" * 40)
            package = snapshot / "cjk_char_byte_fallback_prepared"
            run = "ja_original_char"
            run_dir = package / "data" / run
            run_dir.mkdir(parents=True)
            entries = []
            for name in ("train.bin", "val.bin", "meta.pkl"):
                content = (name + " fixture").encode()
                (run_dir / name).write_bytes(content)
                entries.append(dict(path=f"data/{run}/{name}", bytes=len(content),
                                    sha256=hashlib.sha256(content).hexdigest()))
            manifest = dict(datasets=[run], files=entries)
            manifest_path = package / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            target = root / "target"
            with patch.object(prepared, "hf_hub_download", return_value=str(manifest_path)), \
                    patch.object(prepared, "snapshot_download", return_value=str(snapshot)) as fetch:
                prepared.download_prepared("repo", "main", package.name, target, [run])
                fetch.assert_called_once_with(
                    repo_id="repo", repo_type="dataset", revision="a" * 40,
                    allow_patterns=[f"{package.name}/{entry['path']}" for entry in entries])
                self.assertEqual((target / run / "train.bin").read_bytes(),
                                 (run_dir / "train.bin").read_bytes())
                prepared.download_prepared("repo", "main", package.name, target, [run])
                self.assertEqual(fetch.call_count, 1)
                (target / run / "train.bin").write_bytes(b"different")
                with self.assertRaisesRegex(ValueError, "Existing file differs"):
                    prepared.download_prepared("repo", "main", package.name, target, [run])
                (run_dir / "val.bin").write_bytes(b"corrupted")
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                    prepared.download_prepared("repo", "main", package.name, root / "bad", [run])
                self.assertFalse((root / "bad" / run / "train.bin").exists())
                with self.assertRaisesRegex(ValueError, "does not contain"):
                    prepared.download_prepared("repo", "main", package.name, target,
                                               ["ja_original_char_bpe"])

    def test_charbpe_flat_manifest_combines_with_baseline_at_one_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshots" / ("b" * 40)
            packages = ["cjk_char_byte_fallback_prepared", "cjk_charbpe_tokenized"]
            runs = ["ja_original_char", "ja_original_char_bpe"]
            expected_patterns = []
            for package, run in zip(packages, runs):
                prefix = "cjk_ipa_charbpe" if run.endswith("_bpe") else "data"
                entries = []
                names = ["train.bin", "val.bin", "meta.pkl"]
                if prefix == "cjk_ipa_charbpe":
                    names += ["char_bpe_vocab.json", "char_bpe_token_counts.json"]
                for name in names:
                    path = f"{prefix}/{run}/{name}"
                    target = snapshot / package / path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    content = f"{run}/{name}".encode()
                    target.write_bytes(content)
                    entries.append(dict(path=path, bytes=len(content),
                                        sha256=hashlib.sha256(content).hexdigest()))
                manifest = entries if prefix == "cjk_ipa_charbpe" else dict(
                    datasets=[run], files=entries)
                (snapshot / package / "manifest.json").write_text(json.dumps(manifest))
                expected_patterns.append([f"{package}/{e['path']}" for e in entries])

            def fetch_manifest(**kwargs):
                return str(snapshot / kwargs["filename"])

            with patch.object(prepared, "hf_hub_download", side_effect=fetch_manifest) as manifest_fetch, \
                    patch.object(prepared, "snapshot_download", return_value=str(snapshot)) as fetch:
                target = root / "target"
                prepared.download_comparison("repo", "main", *packages, target, runs)
                self.assertEqual([c.kwargs["revision"] for c in manifest_fetch.call_args_list],
                                 ["main", "b" * 40])
                self.assertEqual([c.kwargs["allow_patterns"] for c in fetch.call_args_list],
                                 expected_patterns)
                self.assertTrue(all(c.kwargs["revision"] == "b" * 40
                                    for c in fetch.call_args_list))
                self.assertEqual((target / runs[1] / "char_bpe_vocab.json").read_bytes(),
                                 f"{runs[1]}/char_bpe_vocab.json".encode())
                prepared.download_comparison("repo", "main", *packages, target, runs)
                self.assertEqual(fetch.call_count, 2)
                # A charBPE-only filter must not fetch the baseline package.
                manifest_fetch.reset_mock()
                prepared.download_comparison("repo", "main", *packages, target, runs[1:])
                self.assertEqual(manifest_fetch.call_count, 1)
                self.assertIn(packages[1], manifest_fetch.call_args.kwargs["filename"])

    def test_prepared_manifest_rejects_missing_files_and_unsafe_paths(self):
        for files, message in (([], "missing required"),
                               ([dict(path="data/../train.bin")], "Invalid manifest path")):
            with self.subTest(files=files), self.assertRaisesRegex(ValueError, message):
                prepared.selected_files(dict(datasets=["ja_original_char"], files=files),
                                        ["ja_original_char"])
        for path in ("../ja/train.bin", "/cjk_ipa_charbpe/ja/train.bin",
                     "cjk_ipa_charbpe/../train.bin"):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "Invalid manifest path"):
                prepared.selected_files([dict(path=path)], ["ja_original_char_bpe"])

    def test_download_train_skips_preprocessing_and_selects_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_python = root / "capture-python"
            fake_python.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                "if sys.argv[1] == '-c': sys.exit(0)\n"
                "assert sys.argv[1].endswith('download_cjk_prepared.py'), sys.argv\n"
                "with open(os.environ['CAPTURE_ARGS'], 'w') as f: json.dump(sys.argv[2:], f)\n")
            fake_python.chmod(0o755)
            env = dict(os.environ, PYTHON_BIN=str(fake_python), RUN_FILTER="^ja_original_",
                       COMPARISON_TOKENIZER="char_byte_fallback", FALLBACK_CHAR_LIMIT="7936",
                       FALLBACK_CHARS_FILE="", CJK_DATA_ROOT=str(root / "data"),
                       CAPTURE_ARGS=str(root / "args.json"))
            command = '''
source demos/cjk_ipa_char_byte_fallback_compare.sh
prepare_all() { exit 99; }
assert_dataset_ready_for_training() { :; }
train_all() { echo training-started; }
main download-train
'''
            result = subprocess.run(["bash", "-c", command, "--", "download-train"],
                                    cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("training-started", result.stdout)
            args = json.loads((root / "args.json").read_text())
            self.assertEqual(args[args.index("--runs") + 1:],
                             ["ja_original_char", "ja_original_char_byte_fallback"])
            self.assertEqual(args[args.index("--data-root") + 1], str(root / "data"))
            # Custom tokenization settings require local preparation.
            env["FALLBACK_CHAR_LIMIT"] = "64"
            result = subprocess.run(["bash", "-c", command, "--", "download-train"],
                                    cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Use prepare/all for custom settings", result.stderr)

            env.update(COMPARISON_TOKENIZER="char_bpe", FALLBACK_CHAR_LIMIT="7936")
            command = command.replace("cjk_ipa_char_byte_fallback_compare.sh",
                                      "cjk_ipa_charbpe_compare.sh")
            result = subprocess.run(["bash", "-c", command, "--", "download-train"],
                                    cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("training-started", result.stdout)
            args = json.loads((root / "args.json").read_text())
            self.assertEqual(args[args.index("--runs") + 1:],
                             ["ja_original_char", "ja_original_char_bpe"])
            self.assertEqual(args[args.index("--charbpe-subdir") + 1], "cjk_charbpe_tokenized")

    def test_download_selects_dataset_revision_and_checks_files(self):
        with tempfile.TemporaryDirectory() as directory:
            args = ["download", "--local-dir", directory, "--revision", "fixed-commit",
                    "--files", "ja.txt"]
            with patch.object(sys, "argv", args), patch.object(downloader, "snapshot_download") as fetch:
                with self.assertRaisesRegex(SystemExit, "Missing or empty"):
                    downloader.main()
                (Path(directory) / "ja.txt").write_text("日本語\n")
                downloader.main()
                fetch.assert_called_with(repo_id="xinyixuu/ted-2020", repo_type="dataset",
                                         revision="fixed-commit", local_dir=directory,
                                         allow_patterns=["ja.txt"])

    def test_entry_points_pass_epoch_model_and_logging_settings(self):
        for entry in ("cjk_ipa_charbpe_compare.sh", "cjk_ipa_char_byte_fallback_compare.sh"):
            with self.subTest(entry=entry), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fake_python = root / "capture-python"
                fake_python.write_text(
                    f"#!{sys.executable}\n"
                    "import json, os, sys\n"
                    f"sys.path.insert(0, {str(ROOT)!r})\n"
                    "assert sys.argv.pop(1) == 'train.py'\n"
                    "from train_args import parse_args\n"
                    "args, *_ = parse_args()\n"
                    "with open(os.environ['CAPTURE_ARGS'], 'w') as f: json.dump(vars(args), f)\n")
                fake_python.chmod(0o755)
                env = {k: v for k, v in os.environ.items() if k not in (
                    "MAX_EPOCHS", "N_LAYER", "N_HEAD", "N_EMBD", "MLP_SIZE", "BLOCK_SIZE",
                    "BATCH_SIZE", "RUN_FILTER", "COMPARISON_TOKENIZER")}
                env.update(PYTHON_BIN=str(fake_python), CJK_OUT_ROOT=str(root / "out"),
                           CJK_LOG_ROOT=str(root / "logs"), CAPTURE_ARGS=str(root / "args.json"))
                command = f"""
source demos/{entry}
assert_dataset_ready_for_training() {{ :; }}
validate_training_outputs() {{ :; }}
write_training_token_stats() {{ :; }}
train_one ja original char
"""
                result = subprocess.run(["bash", "-c", command], cwd=ROOT, env=env,
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                args = json.loads((root / "args.json").read_text())
                expected = dict(training_limiters=["max_epochs"], max_epochs=3,
                                n_layer=12, n_head=12, n_embd=768, mlp_size=3072,
                                block_size=1024, batch_size=16, compile=True,
                                use_qk_norm=True, use_qk_norm_scale=True, tensorboard_log=False)
                for key, value in expected.items():
                    self.assertEqual(args[key], value, key)
                self.assertFalse((root / "logs/tensorboard").exists())

    def test_local_overrides_do_not_download(self):
        env = dict(os.environ, PYTHON_BIN="/does/not/exist", JA_TEXT="/local/ja.txt",
                   KO_TEXT="/local/ko.txt", ZH_CN_TEXT="/local/zh_cn.txt")
        result = subprocess.run(["bash", "-c",
                                 "source demos/cjk_ipa_charbpe_compare.sh; download_source_texts; "
                                 'test "$JA_TEXT" = /local/ja.txt'],
                                cwd=ROOT, env=env, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
