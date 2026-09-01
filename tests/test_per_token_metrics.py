import csv
import math

import torch

from utils.per_token_metrics import PerTokenMetrics
from train_args import parse_args


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

    with open(tracker.summary_path, newline="", encoding="utf-8") as handle:
        summaries = list(csv.DictReader(handle))
    assert {row["metric"] for row in summaries} == {
        "train_loss", "val_loss", "training_seen_count", "vector_magnitude",
        "min_pairwise_angle_deg",
    }
    assert "skew" in summaries[0] and "excess_kurtosis" in summaries[0]
    html = tracker.plot_path.read_text(encoding="utf-8")
    assert "Summary statistics" in html
    graph_files = {
        "per_token_validation_loss.html",
        "per_token_training_loss.html",
        "per_token_training_occurrences.html",
        "per_token_vector_magnitude.html",
        "per_token_min_pairwise_angle.html",
        "per_token_loss_by_iteration.html",
        "per_token_loss_by_appearances.html",
        "per_token_vector_magnitude_by_iteration.html",
        "per_token_min_pairwise_angle_by_iteration.html",
    }
    for filename in graph_files:
        assert filename in html
        graph_html = (tmp_path / filename).read_text(encoding="utf-8")
        assert "Plotly.newPlot" in graph_html
        assert "per_token_metrics.html" in graph_html
    assert "right logarithmic" in (tmp_path / "per_token_loss_by_iteration.html").read_text(encoding="utf-8")
    assert len(list(tmp_path.glob("per_token_static_tiny_iter_*_by_*.png"))) == 10
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
    assert rows[1]["token_text_escaped"] == "\\n"
    assert rows[1]["train_loss"] == "1.25"
    assert rows[1]["training_seen_count"] == "8"
