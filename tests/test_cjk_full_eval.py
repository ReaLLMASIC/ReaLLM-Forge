"""CPU coverage, metric semantics, OOV handling and real-checkpoint integration."""

import csv
import importlib.util
import json
import math
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.cjk_full_eval import (
    FrozenTokenizer, MetricAccumulator, RUN_NAMES, evaluate_stream, export_tokens, file_identity,
    histogram_quantile, iter_windows, load_checkpoint, state_fingerprint, token_bytes,
    verify_identity,
)

spec = importlib.util.spec_from_file_location("prepare_cjk_eval", ROOT / "data/flores200-res/prepare_cjk_eval.py")
flores = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flores)
spec = importlib.util.spec_from_file_location("evaluate_cjk_checkpoints", ROOT / "benchmarks/evaluate_cjk_checkpoints.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def char_meta():
    chars = ["\n", "a", "b"]
    return dict(vocab_size=len(chars), chars=chars,
                stoi={char: index for index, char in enumerate(chars)},
                itos=dict(enumerate(chars)))


def bpe_meta():
    pieces = [bytes([index]) for index in range(256)] + ["\n", "a", "b", "ab", "aba"]
    return dict(vocab_size=len(pieces), tokenizer="char_bpe", char_tokens=pieces[256:],
                stoi={piece: index for index, piece in enumerate(pieces)}, itos=dict(enumerate(pieces)))


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(block_size=4, vocab_size=3)
        self.embedding = nn.Embedding(3, 3)
        self.lm_head = nn.Linear(3, 3, bias=False)
        self.transformer = nn.ModuleDict(dict(ln_f=nn.Identity()))
        with torch.no_grad():
            self.embedding.weight.copy_(torch.eye(3))
            self.lm_head.weight.copy_(torch.tensor([[1., 0., 0.], [0., 2., 0.], [0., 0., 3.]]))

    def forward(self, x, y, iter_num=0):
        return self.lm_head(self.transformer.ln_f(self.embedding(x))), None


class FullEvalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        torch.set_num_threads(1)

    def tearDown(self):
        self.temporary.cleanup()

    def test_windows_cover_each_target_once_including_short_tail(self):
        for size in range(40):
            for block in (1, 4, 8):
                for stride in range(1, block + 1):
                    positions = []
                    for window in iter_windows(size, block, stride):
                        self.assertLessEqual(window.end-window.start, block)
                        self.assertGreaterEqual(window.score_from, 0)
                        positions.extend(range(window.start+1+window.score_from, window.end+1))
                    self.assertEqual(positions, list(range(1, size)))
        for stride in (0, 5):
            with self.assertRaises(ValueError):
                list(iter_windows(10, 4, stride))

    def test_histogram_quantiles_match_explicit_linear_quantile(self):
        values = np.array([1, 1, 2, 3, 3, 3, 9])
        for probability in (0., .5, .95, 1.):
            self.assertAlmostEqual(histogram_quantile(np.bincount(values, minlength=10), probability),
                                   float(np.quantile(values, probability)))

    def test_rank_ties_left_mass_masking_and_exact_quantiles(self):
        probabilities = torch.tensor([[.4, .4, .2], [.1, .6, .3], [.25, .25, .5], [.8, .1, .1]])
        targets = torch.tensor([1, 1, 0, -1])
        accumulator = MetricAccumulator(3, 3, self.root)
        try:
            accumulator.add(probabilities.log(), targets, torch.eye(3)[targets.clamp_min(0)], torch.eye(3), chunk_size=1)
            summary = accumulator.summary([1, 2, 1])
            self.assertEqual(summary["scored_tokens"], 3)
            self.assertEqual(summary["scored_utf8_bytes"], 5)
            self.assertAlmostEqual(summary["target_rank"], 4 / 3)
            self.assertAlmostEqual(summary["target_left_prob"], 1 / 6)
            self.assertAlmostEqual(summary["target_prob"], (0.4+0.6+0.25)/3, places=6)
            # Tied rank=1 does not imply argmax chooses the target's ID.
            self.assertAlmostEqual(summary["top1_correct"], 1/3)
            self.assertAlmostEqual(summary["target_rank_95"], 1.9)
            self.assertAlmostEqual(summary["left_prob_95"], float(np.quantile([.4, .6, .75], .95)), places=6)
            self.assertAlmostEqual(summary["loss"], -sum(math.log(x) for x in (.4, .6, .25))/3, places=6)
            self.assertAlmostEqual(summary["bits_per_byte"], summary["nll_sum"]/math.log(2)/5)
            self.assertEqual(accumulator.counts.tolist(), [1, 2, 0])
            # Keep legacy readers working even when a vocabulary ID never occurs.
            from utils.cjk_comparison_metrics import snapshot_metrics
            path = self.root / "tokens.csv"
            export_tokens(path, char_meta(), accumulator, [10, 20, 30], None, 5, "fixture", "val")
            actual = snapshot_metrics(path, 5)
            np.testing.assert_allclose(actual, [summary["target_prob"], summary["target_rank"],
                                               summary["target_left_prob"]])
        finally:
            accumulator.close()

    def test_frozen_bpe_longest_match_and_unseen_utf8_bytes(self):
        tokenizer = FrozenTokenizer(bpe_meta())
        text = "abaab🙂\n"
        ids = tokenizer.encode(text)
        self.assertEqual(ids[:2], [260, 259])
        self.assertEqual(tokenizer.decode_bytes(ids), text.encode())
        self.assertEqual(sum(len(tokenizer.pieces[index]) for index in ids), len(text.encode()))
        self.assertEqual(tokenizer.unknown_characters(text), {})
        char = FrozenTokenizer(char_meta())
        self.assertEqual(char.unknown_characters("axxx"), {"x": 3})
        with self.assertRaises(KeyError):
            char.encode("x")

    def test_evaluation_matches_direct_forward_and_is_batch_invariant(self):
        ids = np.array([0, 1, 1, 2, 0, 1, 2, 2, 1, 0, 2], dtype=np.uint16)
        model = TinyModel()
        before = state_fingerprint(model)
        direct, _ = model(torch.tensor(ids[:-1].astype(np.int64))[None], None)
        expected = torch.nn.functional.cross_entropy(direct[0], torch.tensor(ids[1:].astype(np.int64))).detach()
        summaries = []
        with patch("torch.optim.AdamW", side_effect=AssertionError("An optimizer must not be created")):
            for batch_size in (1, 3):
                summary, accumulator = evaluate_stream(model, ids, char_meta(), self.root,
                                                        stride=2, batch_size=batch_size)
                try:
                    summaries.append(summary)
                    self.assertAlmostEqual(summary["loss"], float(expected), places=6)
                    self.assertEqual(summary["scored_tokens"], len(ids)-1)
                    self.assertEqual(int(accumulator.counts.sum()), len(ids)-1)
                finally:
                    accumulator.close()
        for metric in ("loss", "target_rank", "target_prob", "target_left_prob", "left_prob_95", "ln_f_cosine_95"):
            self.assertAlmostEqual(summaries[0][metric], summaries[1][metric], places=7)
        self.assertEqual(state_fingerprint(model), before)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertFalse(model.training)
        self.assertEqual(len(model.transformer.ln_f._forward_hooks), 0)

    def test_no_target_stream_and_invalid_ids(self):
        summary, accumulator = evaluate_stream(TinyModel(), np.array([1]), char_meta(), self.root, stride=2)
        accumulator.close()
        self.assertEqual(summary["scored_tokens"], 0)
        self.assertTrue(math.isnan(summary["loss"]))
        with self.assertRaisesRegex(ValueError, "outside"):
            evaluate_stream(TinyModel(), np.array([0, 3]), char_meta(), self.root, stride=2)

    def test_artifact_identity_detects_replaced_data(self):
        path = self.root / "data.bin"
        path.write_bytes(b"abcd")
        identity = file_identity(path)
        self.assertEqual(verify_identity(identity), path)
        path.write_bytes(b"abce")
        with self.assertRaisesRegex(ValueError, "changed"):
            verify_identity(identity)

    def test_flores_oov_does_not_silently_replace_or_remove_rows(self):
        data = self.root / "data"
        for name in RUN_NAMES:
            path = data / name
            path.mkdir(parents=True)
            with (path / "meta.pkl").open("wb") as handle:
                pickle.dump(bpe_meta() if name.endswith("_bpe") else char_meta(), handle)
        records = [dict(id="first", split="dev", texts=dict.fromkeys(flores.LANGUAGES, "a")),
                   dict(id="second", split="dev", texts=dict.fromkeys(flores.LANGUAGES, "x"))]
        output = self.root / "flores"
        output.mkdir()
        manifest = flores.prepare_records(records, data, output,
                                          converter=lambda lang, texts, dest, workers: (["a"]*len(texts), {}))
        char_jobs, bpe_jobs = manifest["runs"]["ja_original_char"], manifest["runs"]["ja_original_char_bpe"]
        self.assertEqual(char_jobs[0]["status"], "unencodable_oov")
        self.assertEqual(char_jobs[0]["unknown_occurrences"], 1)
        self.assertEqual(bpe_jobs[0]["status"], "ready")
        self.assertEqual(char_jobs[1]["scope"], "jointly_encodable")
        self.assertEqual(json.loads(Path(char_jobs[1]["selection"]["path"]).read_text()), ["first"])
        self.assertEqual(char_jobs[1]["selection"]["sha256"], bpe_jobs[1]["selection"]["sha256"])
        self.assertEqual(manifest["coverage"]["dev"]["ja_original"]["supported_sentence_fraction"], .5)
        self.assertEqual(len(manifest["runs"]["ja_ipa_char"]), 1)

    def test_ipa_flags_match_ted_instead_of_flores_defaults(self):
        ja = flores.ipa_command("ja", "src", "dest", "stats", 4)
        self.assertIn("unspaced_ipa", ja)
        self.assertNotIn("--use_spacy", ja)
        ko = flores.ipa_command("ko", "src", "dest", "stats", 4)
        self.assertTrue(ko[1].endswith("espeak2ipa.py"))
        self.assertIn("--no-wrapper", ko)
        self.assertEqual(ko[ko.index("--lang")+1], "ko")

    def test_flores_download_pins_revision_and_preserves_source_order(self):
        from huggingface_hub import DatasetCardData
        import pyarrow as pa
        import pyarrow.parquet as pq
        parquet = self.root / "fixture.parquet"
        rows = [dict(id=index, **{column: text for column in flores.COLUMNS.values()})
                for index, text in ((2, "a"), (1, "b"))]
        pq.write_table(pa.Table.from_pylist(rows), parquet)
        with patch("huggingface_hub.HfApi") as api_class, \
                patch("huggingface_hub.hf_hub_download", return_value=str(parquet)) as download:
            api = api_class.return_value
            api.dataset_info.return_value = SimpleNamespace(sha="a"*40,
                card_data=DatasetCardData(license="cc-by-sa-4.0"))
            api.list_repo_files.return_value = ["README.md", "data/dev-00000-of-00001.parquet"]
            records, provenance = flores.download_records("fixture/repo", "main", self.root)
            self.assertEqual(provenance["revision"], "a"*40)
            self.assertEqual(provenance["card_data"]["license"], "cc-by-sa-4.0")
            self.assertEqual([record["id"] for record in records], ["2", "1"])
            self.assertEqual([record["split"] for record in records], ["dev", "dev"])
            self.assertEqual(download.call_args.kwargs["revision"], "a"*40)
            self.assertEqual(api.list_repo_files.call_args.kwargs["revision"], "a"*40)

    def test_record_alignment_validation(self):
        row = dict(id="a", split="dev", texts=dict.fromkeys(flores.LANGUAGES, "a"))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            flores.validate_records([row, row])
        row["texts"]["ja"] = "a\nb"
        with self.assertRaisesRegex(ValueError, "multiline"):
            flores.validate_records([row])

    def test_metrics_match_original_per_token_accumulator(self):
        from utils.per_token_metrics import PerTokenMetrics
        torch.manual_seed(13)
        logits = torch.randn(2, 7, 3)
        targets = torch.randint(3, (2, 7))
        targets[0, -1] = -1
        old = PerTokenMetrics(self.root / "old", {"test": 3})
        old.begin_evaluation()
        old.add_evaluation_batch("test", "val", logits, targets)
        new = MetricAccumulator(3, 13, self.root)
        try:
            new.add(logits, targets, torch.randn(2, 7, 3), torch.eye(3))
            for old_name, new_name in (("loss", "loss"), ("target_probability", "target_prob"),
                                       ("target_rank", "target_rank"), ("left_probability", "target_left_prob")):
                sums, counts = old.pending[("test", "val", old_name)]
                np.testing.assert_allclose(new.sums[new_name], sums.numpy(), rtol=1e-7, atol=1e-7)
                np.testing.assert_array_equal(new.counts, counts.numpy())
        finally:
            new.close()

    def test_state_mutation_is_detected_and_hooks_are_removed(self):
        class MutatingModel(TinyModel):
            def __init__(self):
                super().__init__()
                self.register_buffer("counter", torch.tensor(0))

            def forward(self, x, y, iter_num=0):
                self.counter.add_(1)
                return super().forward(x, y, iter_num)

        model = MutatingModel()
        with self.assertRaisesRegex(RuntimeError, "changed"):
            evaluate_stream(model, np.array([0, 1, 2]), char_meta(), self.root, stride=2)
        self.assertEqual(len(model.transformer.ln_f._forward_hooks), 0)
        self.assertFalse(list(self.root.glob("quantiles-*")))

    def test_real_gpt_checkpoint_loading_and_batch_padding(self):
        from dataclasses import asdict
        from gpt_conf import GPTConfig
        from model import GPT
        config = GPTConfig(vocab_size=3, n_layer=1, n_head=2, n_kv_group=2,
                           n_embd=8, block_size=4, dropout=0.0)
        torch.manual_seed(123)
        model = GPT(config).eval()
        path = self.root / "ckpt.pt"
        torch.save(dict(model={"_orig_mod."+key: value for key, value in model.state_dict().items()},
                        config=dict(dataset="tiny"), model_args=asdict(config), iter_num=19,
                        optimizer={"sentinel": "not restored"}), path)
        original_hash = file_identity(path)
        with patch("torch.optim.AdamW", side_effect=AssertionError("optimizer constructed")):
            restored, info = load_checkpoint(path)
        self.assertEqual(info["iteration"], 19)
        ids = np.array([0, 1, 2, 1, 0, 2, 1, 1, 2, 0], dtype=np.uint16)
        summaries = []
        for batch_size in (1, 3):
            summary, accumulator = evaluate_stream(restored, ids, char_meta(), self.root,
                                                    stride=2, batch_size=batch_size, iteration=19)
            accumulator.close()
            summaries.append(summary)
        self.assertAlmostEqual(summaries[0]["loss"], summaries[1]["loss"], places=6)
        self.assertEqual(original_hash, file_identity(path))
        # Incomplete model state must fail rather than be silently ignored.
        damaged = torch.load(path, weights_only=False)
        del damaged["model"][next(iter(damaged["model"]))]
        torch.save(damaged, self.root / "bad.pt")
        with self.assertRaises(RuntimeError):
            load_checkpoint(self.root / "bad.pt")

    def test_cli_full_run_reports_and_verified_resume(self):
        from dataclasses import asdict
        from gpt_conf import GPTConfig
        from model import GPT
        config = GPTConfig(vocab_size=3, n_layer=1, n_head=2, n_kv_group=2,
                           n_embd=8, block_size=4, dropout=0.)
        model = GPT(config).eval()
        checkpoint = self.root / "fixture.pt"
        torch.save(dict(model=model.state_dict(), config=dict(dataset="fixture"),
                        model_args=asdict(config), iter_num=7), checkpoint)
        meta_path = self.root / "meta.pkl"
        with meta_path.open("wb") as handle:
            pickle.dump(char_meta(), handle)
        jobs = []
        for split, values in (("train", [0, 1, 2, 1, 1, 2, 0]), ("val", [1, 1, 0, 2])):
            path = self.root / f"{split}.bin"
            np.array(values, dtype=np.uint16).tofile(path)
            jobs.append(dict(id=f"ted_{split}", corpus="ted", split=split, scope="full", status="ready",
                             dtype="uint16", tokens=file_identity(path), source_tokens=len(values),
                             source_utf8_bytes=len(values)))
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps(dict(schema_version=1, runs=[dict(run="ja_original_char",
            checkpoint=file_identity(checkpoint), meta=file_identity(meta_path), training_counts=None,
            iteration=7, jobs=jobs)])))
        args = SimpleNamespace(manifest=manifest, output=self.root / "output", runs=None,
                               device="cpu", dtype="float32", stride=2, batch_size=3,
                               max_tokens=None, geometry=True, corpus="all")
        cli.evaluate(args)
        directory = args.output / "ja_original_char"
        self.assertTrue((directory / "ted_combined.json").is_file())
        self.assertTrue((args.output / "bits_per_byte.png").is_file())
        with (directory / "ted_val/per_token_metrics.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(sum(int(row["val_eval_count"]) for row in rows), 3)
        self.assertTrue(all(row["vector_magnitude"] for row in rows))
        self.assertTrue(all(row["training_seen_count"] == "" for row in rows))
        with patch.object(cli, "load_checkpoint", side_effect=AssertionError("Completed jobs must not rerun")):
            cli.evaluate(args)
        args.stride = 1
        with self.assertRaisesRegex(ValueError, "different evaluation"):
            cli.evaluate(args)

    def test_combined_ted_uses_token_byte_weights_and_global_quantiles(self):
        model, meta = TinyModel(), char_meta()
        run = self.root / "ja_original_char"
        run.mkdir()
        summaries = []
        for split, ids in (("train", [0, 1, 2, 2, 1, 0]), ("val", [2, 1, 2])):
            directory = run / f"ted_{split}"
            directory.mkdir()
            summary, accumulator = evaluate_stream(model, np.array(ids), meta, directory, stride=2)
            try:
                cli.persist_distribution(directory, accumulator)
            finally:
                accumulator.close()
            summary.update(run="ja_original_char", partial=False, source_utf8_bytes=len(ids))
            (directory / "result.json").write_text(json.dumps(summary))
            (directory / "per_token_metrics.csv").write_text("fixture\n")
            cli.mark_complete(directory)
            summaries.append(summary)
        cli.combine_ted(run, meta)
        combined = json.loads((run / "ted_combined.json").read_text())
        count = sum(item["scored_tokens"] for item in summaries)
        self.assertEqual(combined["scored_tokens"], count)
        self.assertEqual(combined["excluded_initial_tokens"], 2)
        self.assertAlmostEqual(combined["loss"], sum(item["loss"]*item["scored_tokens"] for item in summaries)/count)
        self.assertAlmostEqual(combined["bits_per_byte"], sum(item["nll_sum"] for item in summaries)/math.log(2)/count)
        # A completion flag alone is insufficient: corrupt saved distributions
        # must never be silently reused for a resumed or aggregate result.
        (run / "ted_val" / "cosine.f32").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "changed"):
            cli.combine_ted(run, meta)


if __name__ == "__main__":
    unittest.main()
