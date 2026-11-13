"""Check the offline dashboard's resource closure and all bundled snapshots."""

import csv
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import unittest
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1] / "report/pr926-dashboard"


def read_assignment(path, prefix):
    source = path.read_text(encoding="utf-8").strip()
    if not source.startswith(prefix) or not source.endswith(";"):
        raise ValueError(f"Unexpected data wrapper: {path}")
    return json.loads(source[len(prefix):-1])


class References(HTMLParser):
    def __init__(self):
        super().__init__()
        self.files, self.ids, self.scripts = [], set(), []

    def handle_starttag(self, tag, attributes):
        attributes = dict(attributes)
        if "id" in attributes:
            self.ids.add(attributes["id"])
        if tag in ("script", "img") and "src" in attributes:
            self.files.append(attributes["src"])
        if tag in ("link", "a") and attributes.get("href"):
            self.files.append(attributes["href"])
        if tag == "script":
            self.scripts.append(attributes)


class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = read_assignment(ROOT / "data/summary.js", "window.PR926=")
        cls.tokens = {
            run["id"]: read_assignment(ROOT / "data" / f"{run['id']}.js",
                "window.PR926_TOKENS[" + json.dumps(run["id"]) + "]=")
            for run in cls.summary["runs"]
        }

    def test_entrypoint_and_lazy_data_work_with_local_classic_scripts(self):
        references = References()
        references.feed((ROOT / "index.html").read_text(encoding="utf-8"))
        for value in references.files:
            if value.startswith("#"):
                continue
            parsed = urlsplit(value)
            self.assertFalse(parsed.scheme or parsed.netloc, value)
            path = (ROOT / parsed.path).resolve()
            self.assertTrue(path.is_relative_to(ROOT.resolve()), value)
            self.assertTrue(path.is_file(), value)
        self.assertTrue(all(script.get("type") != "module" for script in references.scripts))
        sources = [script["src"] for script in references.scripts]
        self.assertLess(sources.index("assets/plotly.min.js"), sources.index("assets/dashboard.js"))
        self.assertLess(sources.index("data/summary.js"), sources.index("assets/dashboard.js"))
        self.assertLess(sources.index("data/token-evidence.js"), sources.index("assets/evidence.js"))
        for filename in ("dashboard.js", "evidence.js"):
            code = (ROOT / "assets" / filename).read_text(encoding="utf-8")
            self.assertIsNone(re.search(r"\b(fetch|XMLHttpRequest|WebSocket)\s*\(|\bimport\s*\(", code))
            for identifier in re.findall(r"\$\(['\"]([^'\"]+)['\"]\)", code):
                self.assertIn(identifier, references.ids, filename)
        css = (ROOT / "assets/style.css").read_text(encoding="utf-8")
        self.assertNotIn("@import", css)
        self.assertNotRegex(css, r"url\(\s*['\"]?(?:https?:)?//")

    def test_all_12_runs_include_every_checkpoint_and_complete_token_arrays(self):
        expected = {f"{language}_{representation}_{tokenizer}"
                    for language in ("ja", "ko", "zh_cn")
                    for representation in ("original", "ipa") for tokenizer in ("char", "char_bpe")}
        self.assertEqual(set(self.tokens), expected)
        self.assertEqual(len(self.summary["runs"]), 12)
        self.assertEqual(sum(len(run["steps"]) for run in self.summary["runs"]), 101)
        metric_count = len(self.summary["metrics"])
        self.assertEqual(metric_count, 10)
        for run in self.summary["runs"]:
            data = self.tokens[run["id"]]
            self.assertEqual(len(data["tokens"]), run["vocab"])
            self.assertEqual({int(step) for step in data["snapshots"]}, set(run["steps"]))
            self.assertIn(run["endpoint"], run["steps"])
            for step, snapshot in data["snapshots"].items():
                self.assertEqual(len(snapshot), metric_count)
                self.assertTrue(all(len(values) == run["vocab"] for values in snapshot))
                self.assertTrue(all(value is None or math.isfinite(value)
                                    for values in snapshot for value in values))
                self.assertEqual(sum(snapshot[0]), int(step) * 16 * 1024)
                self.assertEqual(sum(snapshot[8]), 50 * 16 * 1024)
                self.assertEqual(sum(snapshot[9]), 50 * 16 * 1024)
                for index in (0, 8, 9):
                    self.assertTrue(all(value >= 0 and value == int(value) for value in snapshot[index]))
                for index in (5, 7):
                    # Preserve the supplied FP32 sums: a few left masses are
                    # 1.00000012, rather than clipping historical observations.
                    self.assertTrue(all(value is None or 0 <= value <= 1 + 5e-7 for value in snapshot[index]))
                self.assertTrue(all(value is None or 1 <= value <= run["vocab"] for value in snapshot[6]))

    def test_overview_downloads_and_token_losses_describe_the_same_final_runs(self):
        with (ROOT / "data/results_all_12_runs.tsv").open(newline="", encoding="utf-8") as handle:
            table = {row["run"]: row for row in csv.DictReader(handle, delimiter="\t")}
        self.assertEqual(set(table), set(self.tokens))
        for run in self.summary["runs"]:
            row = table[run["id"]]
            self.assertEqual(run["endpoint"], int(row["final_iteration"]))
            self.assertAlmostEqual(run["finalBpb"], float(row["final_bpb"]), places=12)
            self.assertAlmostEqual(run["finalLoss"], float(row["final_val_loss"]), places=12)
            self.assertAlmostEqual(run["finalEpoch"], float(row["final_epochs"]), places=12)
            snapshot = self.tokens[run["id"]]["snapshots"][str(run["endpoint"])]
            count = sum(snapshot[8])
            loss = sum(value * weight for value, weight in zip(snapshot[1], snapshot[8]) if weight) / count
            self.assertAlmostEqual(loss, run["finalLoss"], delta=1e-6)
        evidence = read_assignment(ROOT / "data/token-evidence.js", "window.PR926_TOKEN_EVIDENCE=")
        self.assertEqual(set(evidence["summary"]), set(self.tokens))
        for run in self.summary["runs"]:
            self.assertEqual(evidence["summary"][run["id"]]["endpoint"]["step"], run["endpoint"])

    def test_original_assets_data_and_plotly_license_are_preserved(self):
        provenance = json.loads((ROOT / "provenance.json").read_text(encoding="utf-8"))
        files = provenance["unchanged_source_files"]
        self.assertEqual(len(files), 21)
        for relative, identity in files.items():
            data = (ROOT / relative).read_bytes()
            self.assertEqual(len(data), identity["bytes"], relative)
            self.assertEqual(hashlib.sha256(data).hexdigest(), identity["sha256"], relative)
        self.assertIn("assets/plotly.LICENSE.txt", files)
        license_text = (ROOT / "assets/plotly.LICENSE.txt").read_text(encoding="utf-8")
        self.assertIn("Plotly, Inc", license_text)
        self.assertIn("Permission is hereby granted", license_text)

    @unittest.skipUnless(shutil.which("node"), "Node.js is only needed for the syntax check")
    def test_bundled_javascript_parses(self):
        for path in sorted(ROOT.rglob("*.js")):
            result = subprocess.run([shutil.which("node"), "--check", str(path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, f"{path}: {result.stderr}")


if __name__ == "__main__":
    unittest.main()
