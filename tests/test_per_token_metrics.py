import csv
import json
import math
from pathlib import Path
import re
import shutil
import subprocess

import pytest
import torch

from utils.per_token_metrics import PerTokenMetrics
from train_args import parse_args


def test_counter_checkpoint_round_trip_is_weights_only_safe_and_independent(tmp_path):
    tracker = PerTokenMetrics(tmp_path / 'original', {'a': 3, 'b': 5}, counting_started_at=7)
    tracker.count_training_batch('a', torch.tensor([[0, 1, 1, -1]]))
    tracker.count_training_batch('b', torch.tensor([[4, 4, 2]]))
    state = tracker.state_dict()
    tracker.count_training_batch('a', torch.tensor([[2]]))
    assert state['seen']['a'].tolist() == [1, 2, 0]
    path = tmp_path / 'counts.pt'
    torch.save(state, path)
    loaded = torch.load(path, weights_only=True)
    resumed = PerTokenMetrics(tmp_path / 'resumed', {'a': 3, 'b': 5})
    resumed.load_state_dict(loaded)
    resumed.count_training_batch('a', torch.tensor([[1, 2]]))
    assert resumed.seen['a'].tolist() == [1, 3, 1]
    assert resumed.seen['b'].tolist() == [0, 0, 1, 0, 2]
    assert loaded['seen']['a'].tolist() == [1, 2, 0]
    assert resumed.counting_started_at == 7


@pytest.mark.parametrize('bad_state', [
    {'version': 2},
    {'vocab_sizes': {'other': 3}},
    {'seen': {}},
    {'seen': {'tiny': torch.tensor([1, 2])}},
    {'seen': {'tiny': torch.tensor([1.0, 2.0, 3.0])}},
    {'seen': {'tiny': torch.tensor([1, -1, 3])}},
    {'counting_started_at': -1},
])
def test_invalid_counter_checkpoint_does_not_change_live_counts(tmp_path, bad_state):
    tracker = PerTokenMetrics(tmp_path, {'tiny': 3})
    tracker.count_training_batch('tiny', torch.tensor([2]))
    state = tracker.state_dict()
    state.update(bad_state)
    with pytest.raises(ValueError, match='[Pp]er-token'):
        tracker.load_state_dict(state)
    assert tracker.seen['tiny'].tolist() == [0, 0, 1]


def test_tensorboard_default_and_eval_interval(monkeypatch):
    monkeypatch.setattr("sys.argv", ["train.py", "--eval_interval", "50"])
    args, *_ = parse_args()
    assert args.tensorboard_log is True
    assert args.eval_interval == 50


def test_per_token_metrics_exports_counts_losses_summaries_and_plot(tmp_path):
    tracker = PerTokenMetrics(
        tmp_path, {"tiny": 3}, {"tiny": {0: "\\n", 1: "a", 2: "\\t"}}
    )
    tracker.count_training_batch("tiny", torch.tensor([[0, 1, 1, 2]]))
    tracker.set_token_geometry(
        "tiny", torch.tensor([[3.0, 4.0], [0.0, 2.0], [1.0, 0.0]])
    )
    tracker.begin_evaluation()
    targets = torch.tensor([[0, 1, 1]])
    logits = torch.tensor([[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]])
    tracker.add_evaluation_batch("tiny", "train", logits, targets)
    tracker.add_evaluation_batch("tiny", "val", logits, targets)
    tracker.export(10)
    tracker.begin_evaluation()
    tracker.add_evaluation_batch("tiny", "train", logits, targets)
    tracker.add_evaluation_batch("tiny", "val", logits, targets)
    tracker.export(20)

    with open(tracker.detail_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert [int(row["training_seen_count"]) for row in rows[:3]] == [1, 2, 1]
    assert [row["token_text_escaped"] for row in rows[:3]] == ["\\n", "a", "\\t"]
    assert math.isclose(float(rows[0]["val_loss"]), 0.0949229, rel_tol=1e-5)
    assert int(rows[1]["val_eval_count"]) == 2
    assert [float(row["vector_magnitude"]) for row in rows[:3]] == [5.0, 2.0, 1.0]
    assert all(math.isfinite(float(row["min_pairwise_angle_deg"])) for row in rows[:3])
    for metric in ("avg_target_probability", "avg_target_rank", "avg_left_probability"):
        assert all(math.isfinite(float(row[metric])) for row in rows[:2])
        assert math.isnan(float(rows[2][metric]))

    with open(tracker.summary_path, newline="", encoding="utf-8") as handle:
        summaries = list(csv.DictReader(handle))
    assert {row["metric"] for row in summaries} == {
        "train_loss", "val_loss", "training_seen_count", "vector_magnitude",
        "min_pairwise_angle_deg",
        "avg_target_probability", "avg_target_rank", "avg_left_probability",
    }
    assert "skew" in summaries[0] and "excess_kurtosis" in summaries[0]
    html = Path(tracker.plot_path).read_text(encoding="utf-8")
    assert "Summary statistics" in html
    graph_files = {
        "per_token_validation_loss.html",
        "per_token_training_loss.html",
        "per_token_training_occurrences.html",
        "per_token_vector_magnitude.html",
        "per_token_min_pairwise_angle.html",
        "per_token_target_probability.html",
        "per_token_target_rank.html",
        "per_token_left_probability.html",
        "per_token_loss_by_iteration.html",
        "per_token_loss_by_appearances.html",
        "per_token_vector_magnitude_by_iteration.html",
        "per_token_min_pairwise_angle_by_iteration.html",
        "per_token_target_probability_by_iteration.html",
        "per_token_target_rank_by_iteration.html",
        "per_token_left_probability_by_iteration.html",
    }
    for filename in graph_files:
        assert filename in html
        graph_html = (tmp_path / filename).read_text(encoding="utf-8")
        assert "Plotly.newPlot" in graph_html
        assert "per_token_metrics.html" in graph_html
    assert "right logarithmic" in (tmp_path / "per_token_loss_by_iteration.html").read_text(encoding="utf-8")
    assert len(list(tmp_path.glob("per_token_static_tiny_iter_*_by_*.png"))) == 16
    slideshow = (tmp_path / "per_token_static_slideshow.html").read_text(encoding="utf-8")
    assert "per_token_static_slideshow.html" in html
    assert "per_token_metrics.html" in slideshow
    assert "ArrowLeft" in slideshow and "ArrowRight" in slideshow
    assert "Previous" in slideshow and "Next" in slideshow


def test_per_token_metrics_migrates_legacy_detail_csv(tmp_path):
    detail_path = tmp_path / "per_token_metrics.csv"
    detail_path.write_text(
        "iteration,dataset,token_id,train_loss,train_eval_count,val_loss,val_eval_count,training_seen_count\n"
        "10,tiny,0,1.5,2,2.5,3,4\n"
        "20,tiny,0,\\n,1.25,2,2.25,3,8\n",
        encoding="utf-8",
    )

    tracker = PerTokenMetrics(tmp_path, {"tiny": 1}, {"tiny": {0: "\\n"}})

    with detail_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["token_text_escaped"] == "\\n"
    assert rows[0]["train_loss"] == "1.5"
    assert rows[0]["training_seen_count"] == "4"
    assert rows[0]["vector_magnitude"] == "nan"
    assert rows[0]["min_pairwise_angle_deg"] == "nan"
    assert rows[0]["avg_target_probability"] == "nan"
    assert rows[0]["avg_target_rank"] == "nan"
    assert rows[0]["avg_left_probability"] == "nan"
    assert rows[1]["token_text_escaped"] == "\\n"
    assert rows[1]["train_loss"] == "1.25"
    assert rows[1]["training_seen_count"] == "8"


def test_probability_metrics_have_exact_validation_semantics(tmp_path, monkeypatch):
    tracker = PerTokenMetrics(tmp_path, {"tiny": 3})
    tracker.EVALUATION_CHUNK_SIZE = 1
    monkeypatch.setattr(tracker, "_write_plot", lambda *args: None)
    probabilities = torch.tensor([
        [0.4, 0.4, 0.2], [0.1, 0.6, 0.3],
        [0.25, 0.25, 0.5], [0.8, 0.1, 0.1],
    ])
    targets = torch.tensor([1, 1, 0, -1])
    tracker.count_training_batch("tiny", targets)
    tracker.begin_evaluation()
    # A different training distribution must not leak into validation metrics.
    tracker.add_evaluation_batch("tiny", "train", torch.zeros(4, 3), targets)
    tracker.add_evaluation_batch("tiny", "val", probabilities[:2].log(), targets[:2])
    tracker.add_evaluation_batch("tiny", "val", probabilities[2:].log(), targets[2:])
    tracker.export(7)
    with open(tracker.detail_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(r["training_seen_count"]) for r in rows] == [1, 2, 0]
    assert [int(r["val_eval_count"]) for r in rows] == [1, 2, 0]
    for index, expected in enumerate(((0.25, 2.0, 0.5), (0.5, 1.0, 0.0))):
        actual = [float(rows[index][key]) for key in (
            "avg_target_probability", "avg_target_rank", "avg_left_probability"
        )]
        assert actual == pytest.approx(expected)
    assert float(rows[1]["val_loss"]) == pytest.approx(
        (-math.log(0.4) - math.log(0.6)) / 2
    )
    assert float(rows[1]["train_loss"]) == pytest.approx(math.log(3))
    assert math.isnan(float(rows[2]["avg_target_probability"]))
    tracker.begin_evaluation()
    tracker.add_evaluation_batch("tiny", "val", torch.zeros(1, 3), torch.tensor([-1]))
    assert not tracker.pending


def test_probability_metrics_extreme_logits_and_invalid_targets(tmp_path):
    tracker = PerTokenMetrics(tmp_path, {"tiny": 3})
    logits = torch.tensor([[10000.0, -10000.0, 10000.0]])
    tracker.add_evaluation_batch("tiny", "val", logits, torch.tensor([1]))
    assert tracker.pending[("tiny", "val", "target_rank")][0][1] == 3
    assert tracker.pending[("tiny", "val", "target_probability")][0][1] == 0
    assert tracker.pending[("tiny", "val", "left_probability")][0][1] == 1
    with pytest.raises(ValueError, match="outside the vocabulary"):
        tracker.add_evaluation_batch("tiny", "val", logits, torch.tensor([3]))
    with pytest.raises(ValueError, match="outside the vocabulary"):
        tracker.count_training_batch("tiny", torch.tensor([-2]))


@pytest.mark.parametrize("header_width", [8, 9, 10, 11, 14])
def test_migration_preserves_all_published_and_mixed_row_layouts(tmp_path, header_width):
    full = PerTokenMetrics.DETAIL_FIELDS
    previous = tuple(name for name in full if not name.startswith("avg_"))
    schemas = {14: full, 11: previous}
    for field in ("min_pairwise_angle_deg", "vector_magnitude", "token_text_escaped"):
        previous = tuple(name for name in previous if name != field)
        schemas[len(previous)] = previous
    sample = dict(iteration=10, dataset="tiny", token_id=0, token_text_escaped="\\n",
                  vector_magnitude=4.0, min_pairwise_angle_deg=90.0,
                  train_loss=1.25, train_eval_count=2, val_loss=2.25,
                  val_eval_count=3, training_seen_count=8,
                  avg_target_probability=0.3, avg_target_rank=2,
                  avg_left_probability=0.7)
    path = tmp_path / "per_token_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(schemas[header_width])
        for width in (8, 9, 10, 11, 14):
            writer.writerow([sample[name] for name in schemas[width]])
    PerTokenMetrics(tmp_path, {"tiny": 1}, {"tiny": {0: "\\n"}})
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        assert row["train_loss"] == "1.25"
        assert row["val_loss"] == "2.25"
        assert row["training_seen_count"] == "8"
        assert row["token_text_escaped"] == "\\n"
    assert all(row["avg_target_probability"] == "nan" for row in rows[:-1])
    assert rows[-1]["avg_target_probability"] == "0.3"


def test_migration_rejects_unknown_row_width_without_replacing_source(tmp_path):
    path = tmp_path / "per_token_metrics.csv"
    original = (
        "iteration,dataset,token_id,train_loss,train_eval_count,val_loss,val_eval_count,training_seen_count\n"
        "10,tiny,0,1.0,2,2.0,3,4,unexpected,extra,fields,here\n"
    )
    path.write_text(original, encoding="utf-8")
    with pytest.raises(ValueError, match="row width"):
        PerTokenMetrics(tmp_path, {"tiny": 1})
    assert path.read_text(encoding="utf-8") == original


def test_report_escapes_summary_cells(tmp_path, monkeypatch):
    from utils.per_token_html import write_per_token_pages
    import utils.per_token_static as static
    monkeypatch.setattr(static, "write_static_dashboards", lambda *args: [])
    dataset = '<img src=x onerror="alert(1)">'
    row = {field: 0 for field in PerTokenMetrics.DETAIL_FIELDS}
    row.update(dataset=dataset, token_text_escaped="test")
    write_per_token_pages(tmp_path, [row], [{"dataset": dataset}], 0)
    page = (tmp_path / "per_token_metrics.html").read_text(encoding="utf-8")
    assert dataset not in page
    assert "&lt;img" in page


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is needed for JS runtime test")
def test_token_text_is_runtime_data_and_cannot_execute_template_expressions(tmp_path):
    from utils.per_token_html import _overview
    token = "` ${globalThis.injected = true} </script><script>"
    row = dict(dataset="tiny", token_id=0, token_text_escaped=token,
               val_loss=1.0, train_loss=1.0, training_seen_count=1)
    _overview(tmp_path, "test.html", "Test", [row], "val_loss", True, False)
    page = (tmp_path / "test.html").read_text(encoding="utf-8")
    script = re.search(r"<script>(.*?)</script>", page, re.S).group(1)
    runner = r'''
const vm = require('node:vm');
const fs = require('node:fs');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
let captured;
const select = {value: '', add(option) {this.value ||= option.value;}};
const sandbox = {
  document: {getElementById(id) {return id === 'dataset' ? select : {checked: false};}},
  Option: function(text, value) {this.text = text; this.value = value;},
  Plotly: {newPlot(id, traces) {captured = traces; return Promise.resolve();}},
  error: {}, console,
};
vm.runInNewContext(input.script, sandbox);
if (sandbox.injected !== undefined) throw Error('Token text executed');
if (!captured[0].x[0].includes(input.token)) throw Error('Token text changed');
'''
    subprocess.run(["node", "-e", runner], input=json.dumps({"script": script, "token": token}),
                   text=True, check=True, capture_output=True)
