"""Check Hub downloads and actual shell-generated training arguments without a GPU."""

import importlib.util
import json
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


class HubPipelineTests(unittest.TestCase):
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
