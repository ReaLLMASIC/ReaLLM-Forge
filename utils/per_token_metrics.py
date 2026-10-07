"""Per-token loss/count reporting kept deliberately separate from TensorBoard."""

import csv
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from utils.per_token_html import write_per_token_pages
from utils.min_angle_graph_export import compute_min_angle_graph


class PerTokenMetrics:
    """Accumulate training-token exposure and export evaluation snapshots."""

    DETAIL_FIELDS = (
        "iteration", "dataset", "token_id", "token_text_escaped", "vector_magnitude",
        "min_pairwise_angle_deg", "train_loss", "train_eval_count", "val_loss",
        "val_eval_count", "avg_target_probability", "avg_target_rank",
        "avg_left_probability", "training_seen_count",
    )
    EVALUATION_CHUNK_SIZE = 256

    def __init__(self, output_dir, vocab_sizes, token_texts=None, counting_started_at=0):
        self.output_dir = output_dir
        self.vocab_sizes = dict(vocab_sizes)
        self.token_texts = token_texts or {}
        self.counting_started_at = counting_started_at
        self.seen = {
            name: np.zeros(size, dtype=np.int64) for name, size in self.vocab_sizes.items()
        }
        self.pending = {}
        self.vector_magnitudes = {}
        self.min_pairwise_angles = {}
        os.makedirs(output_dir, exist_ok=True)
        self.detail_path = os.path.join(output_dir, "per_token_metrics.csv")
        self.summary_path = os.path.join(output_dir, "per_token_summary.csv")
        self.plot_path = os.path.join(output_dir, "per_token_metrics.html")
        self._ensure_detail_schema()

    def state_dict(self):
        """Save counters at the same training boundary as the model checkpoint."""
        return {
            'version': 1,
            'vocab_sizes': dict(self.vocab_sizes),
            'counting_started_at': self.counting_started_at,
            # CPU tensors remain compatible with torch.load(weights_only=True).
            'seen': {name: torch.from_numpy(counts.copy())
                     for name, counts in self.seen.items()},
        }

    def load_state_dict(self, state):
        """Restore validated counters without retaining aliases to the checkpoint."""
        if state.get('version') != 1:
            raise ValueError('Unsupported per-token checkpoint state version')
        if state.get('vocab_sizes') != self.vocab_sizes:
            raise ValueError('Per-token checkpoint datasets/vocabulary sizes do not match')
        if set(state.get('seen', {})) != set(self.vocab_sizes):
            raise ValueError('Per-token checkpoint counters do not match datasets')
        started_at = state.get('counting_started_at')
        if type(started_at) is not int or started_at < 0:
            raise ValueError('Invalid per-token counting start iteration')
        restored = {}
        for name, size in self.vocab_sizes.items():
            counts = state['seen'][name]
            if (not isinstance(counts, torch.Tensor) or counts.dtype != torch.int64
                    or tuple(counts.shape) != (size,) or torch.any(counts < 0)):
                raise ValueError(f'Invalid per-token checkpoint counters for {name}')
            restored[name] = counts.detach().cpu().numpy().copy()
        self.seen = restored
        self.counting_started_at = started_at

    def _ensure_detail_schema(self):
        """Upgrade detail CSVs written before escaped token text was added."""
        if not os.path.exists(self.detail_path) or os.path.getsize(self.detail_path) == 0:
            return
        with open(self.detail_path, newline="", encoding="utf-8") as handle:
            raw_rows = list(csv.reader(handle))
        if not raw_rows:
            return

        header = tuple(raw_rows[0])
        required_legacy_fields = {
            "iteration", "dataset", "token_id", "train_loss", "train_eval_count",
            "val_loss", "val_eval_count", "training_seen_count",
        }
        if (not required_legacy_fields.issubset(header)
                or not set(header).issubset(self.DETAIL_FIELDS)
                or len(set(header)) != len(header)):
            raise ValueError(
                f"Unsupported per-token metrics CSV schema in {self.detail_path}: "
                f"{raw_rows[0]}"
            )

        if header == self.DETAIL_FIELDS and all(
            len(values) == len(header) for values in raw_rows[1:]
        ):
            return

        # Preserve all earlier published row layouts, including rows appended
        # beneath a stale header by a partially migrated run.
        previous = tuple(field for field in self.DETAIL_FIELDS
                         if not field.startswith("avg_"))
        schemas = {len(self.DETAIL_FIELDS): self.DETAIL_FIELDS,
                   len(previous): previous}
        for field in ("min_pairwise_angle_deg", "vector_magnitude", "token_text_escaped"):
            previous = tuple(name for name in previous if name != field)
            schemas[len(previous)] = previous

        migrated = []
        for line_number, values in enumerate(raw_rows[1:], start=2):
            fields = header if len(values) == len(header) else schemas.get(len(values))
            if fields is None:
                raise ValueError(
                    f"Unsupported per-token CSV row width at line {line_number}: "
                    f"{len(values)} fields; original file was not changed"
                )
            row = dict(zip(fields, values))
            dataset = row.get("dataset", "")
            token_id = int(row["token_id"])
            row["token_text_escaped"] = (
                row.get("token_text_escaped")
                or self.token_texts.get(dataset, {}).get(token_id, "")
            )
            row.setdefault("vector_magnitude", "nan")
            row.setdefault("min_pairwise_angle_deg", "nan")
            row.setdefault("avg_target_probability", "nan")
            row.setdefault("avg_target_rank", "nan")
            row.setdefault("avg_left_probability", "nan")
            migrated.append(row)

        temporary_path = self.detail_path + ".tmp"
        with open(temporary_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.DETAIL_FIELDS)
            writer.writeheader()
            writer.writerows(migrated)
        os.replace(temporary_path, self.detail_path)

    def count_training_batch(self, dataset, targets):
        values = targets.detach().reshape(-1).to("cpu", dtype=torch.long)
        values = values[values != -1]  # Match the repository's loss ignore_index.
        if torch.any((values < 0) | (values >= self.vocab_sizes[dataset])):
            raise ValueError(f"Training target outside the vocabulary for {dataset}")
        counts = torch.bincount(values, minlength=self.vocab_sizes[dataset]).numpy()
        self.seen[dataset] += counts[: self.vocab_sizes[dataset]]

    def begin_evaluation(self):
        self.pending = {}

    def set_vector_magnitudes(self, dataset, weight):
        """Capture the current output-token vector L2 norms for an evaluation."""
        self.vector_magnitudes[dataset] = (
            weight.detach().float().norm(dim=-1).cpu().numpy()
        )

    def set_token_geometry(self, dataset, weight, block_size=2048, compute_device="auto"):
        """Capture vector lengths and each token's closest non-self angle."""
        graph = compute_min_angle_graph(
            weight, block_size=block_size, compute_device=compute_device
        )
        self.vector_magnitudes[dataset] = graph["norms"].numpy()
        self.min_pairwise_angles[dataset] = graph["min_angles"].numpy()

    def add_evaluation_batch(self, dataset, split, logits, targets):
        """Collect CE on both splits and categorical probability metrics on val.

        Rank is 1 + the number of strictly larger logits. Left probability is
        the softmax mass of those same competitors; ties are excluded.
        Temporary vocabulary-sized tensors are bounded to a chunk of positions.
        """
        vocab_size = self.vocab_sizes[dataset]
        flat_logits = logits.detach().reshape(-1, logits.size(-1))
        flat_targets = targets.detach().reshape(-1).to(logits.device, dtype=torch.long)
        if len(flat_logits) != len(flat_targets):
            raise ValueError("Logits and targets must contain the same number of positions")
        for start in range(0, len(flat_targets), self.EVALUATION_CHUNK_SIZE):
            target_chunk = flat_targets[start:start + self.EVALUATION_CHUNK_SIZE]
            valid = target_chunk != -1
            if not valid.any():
                continue
            ids = target_chunk[valid]
            if torch.any((ids < 0) | (ids >= vocab_size) | (ids >= logits.size(-1))):
                raise ValueError(f"Evaluation target outside the vocabulary for {dataset}")
            chunk = flat_logits[start:start + self.EVALUATION_CHUNK_SIZE][valid].float()
            values_by_metric = {
                "loss": F.cross_entropy(chunk, ids, reduction="none"),
            }
            if split == "val":
                probabilities = F.softmax(chunk, dim=-1)
                target_logits = chunk.gather(1, ids.unsqueeze(1))
                ahead = chunk > target_logits
                values_by_metric.update(
                    target_probability=probabilities.gather(1, ids.unsqueeze(1)).squeeze(1),
                    target_rank=ahead.sum(dim=1).add(1),
                    left_probability=probabilities.masked_fill(~ahead, 0).sum(dim=1),
                )
            cpu_ids = ids.cpu()
            batch_counts = torch.bincount(cpu_ids, minlength=vocab_size)
            for metric, values in values_by_metric.items():
                key = (dataset, split, metric)
                if key not in self.pending:
                    self.pending[key] = (
                        torch.zeros(vocab_size, dtype=torch.float64),
                        torch.zeros(vocab_size, dtype=torch.int64),
                    )
                sums, counts = self.pending[key]
                sums.scatter_add_(0, cpu_ids, values.to("cpu", dtype=torch.float64))
                counts += batch_counts

    @staticmethod
    def _summary(values):
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if not values.size:
            return {key: math.nan for key in ("mean", "median", "std", "skew", "excess_kurtosis", "min", "max", "p10", "p90", "coefficient_of_variation")}
        mean, std = values.mean(), values.std()
        centered = values - mean
        skew = np.mean(centered ** 3) / std ** 3 if std else 0.0
        kurtosis = np.mean(centered ** 4) / std ** 4 - 3 if std else 0.0
        return {
            "mean": mean, "median": np.median(values), "std": std, "skew": skew,
            "excess_kurtosis": kurtosis, "min": values.min(), "max": values.max(),
            "p10": np.percentile(values, 10), "p90": np.percentile(values, 90),
            "coefficient_of_variation": std / mean if mean else math.nan,
        }

    def export(self, iteration):
        rows, summaries = [], []
        for dataset, vocab_size in self.vocab_sizes.items():
            split_data = {}
            for split in ("train", "val"):
                sums, counts = self.pending.get(
                    (dataset, split, "loss"),
                    (torch.zeros(vocab_size), torch.zeros(vocab_size, dtype=torch.long)),
                )
                sums, counts = sums.numpy(), counts.numpy()
                split_data[split] = np.divide(
                    sums, counts, out=np.full(vocab_size, np.nan), where=counts != 0
                )
                split_data[split + "_count"] = counts
            for metric in ("target_probability", "target_rank", "left_probability"):
                sums, counts = self.pending.get(
                    (dataset, "val", metric),
                    (torch.zeros(vocab_size), torch.zeros(vocab_size, dtype=torch.long)),
                )
                sums, counts = sums.numpy(), counts.numpy()
                split_data[metric] = np.divide(
                    sums, counts, out=np.full(vocab_size, np.nan), where=counts != 0
                )
            for token_id in range(vocab_size):
                rows.append({
                    "iteration": iteration, "dataset": dataset, "token_id": token_id,
                    "token_text_escaped": self.token_texts.get(dataset, {}).get(token_id, ""),
                    "vector_magnitude": float(self.vector_magnitudes.get(
                        dataset, np.full(vocab_size, np.nan)
                    )[token_id]),
                    "min_pairwise_angle_deg": float(self.min_pairwise_angles.get(
                        dataset, np.full(vocab_size, np.nan)
                    )[token_id]),
                    "train_loss": float(split_data["train"][token_id]),
                    "train_eval_count": int(split_data["train_count"][token_id]),
                    "val_loss": float(split_data["val"][token_id]),
                    "val_eval_count": int(split_data["val_count"][token_id]),
                    "avg_target_probability": float(split_data["target_probability"][token_id]),
                    "avg_target_rank": float(split_data["target_rank"][token_id]),
                    "avg_left_probability": float(split_data["left_probability"][token_id]),
                    "training_seen_count": int(self.seen[dataset][token_id]),
                })
            for metric, values in (
                ("train_loss", split_data["train"]), ("val_loss", split_data["val"]),
                ("training_seen_count", self.seen[dataset]),
                ("vector_magnitude", self.vector_magnitudes.get(
                    dataset, np.full(vocab_size, np.nan)
                )),
                ("min_pairwise_angle_deg", self.min_pairwise_angles.get(
                    dataset, np.full(vocab_size, np.nan)
                )),
                ("avg_target_probability", split_data["target_probability"]),
                ("avg_target_rank", split_data["target_rank"]),
                ("avg_left_probability", split_data["left_probability"]),
            ):
                summary = self._summary(values)
                summary.update(iteration=iteration, dataset=dataset, metric=metric,
                               populated_tokens=int(np.isfinite(values).sum()), vocab_size=vocab_size)
                summaries.append(summary)
        self._append_csv(self.detail_path, rows, self.DETAIL_FIELDS)
        summary_fields = ("iteration", "dataset", "metric", "populated_tokens", "vocab_size",
                          "mean", "median", "std", "skew", "excess_kurtosis", "min", "max",
                          "p10", "p90", "coefficient_of_variation")
        self._append_csv(self.summary_path, summaries, summary_fields)
        self._write_plot(self._read_detail_rows(), summaries, iteration)

    @staticmethod
    def _append_csv(path, rows, fields):
        new_file = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if new_file:
                writer.writeheader()
            writer.writerows(rows)

    def _read_detail_rows(self):
        """Load all snapshots so the HTML can plot a token's history."""
        rows = []
        with open(self.detail_path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append({
                    "iteration": int(row["iteration"]),
                    "dataset": row["dataset"],
                    "token_id": int(row["token_id"]),
                    "token_text_escaped": row.get("token_text_escaped", ""),
                    "vector_magnitude": float(row.get("vector_magnitude", "nan")),
                    "min_pairwise_angle_deg": float(row.get("min_pairwise_angle_deg", "nan")),
                    "train_loss": float(row["train_loss"]),
                    "train_eval_count": int(row["train_eval_count"]),
                    "val_loss": float(row["val_loss"]),
                    "val_eval_count": int(row["val_eval_count"]),
                    "avg_target_probability": float(row.get("avg_target_probability", "nan")),
                    "avg_target_rank": float(row.get("avg_target_rank", "nan")),
                    "avg_left_probability": float(row.get("avg_left_probability", "nan")),
                    "training_seen_count": int(row["training_seen_count"]),
                })
        return rows

    def _write_plot(self, latest_rows, summaries, iteration):
        """Write a lightweight index and isolated graph pages."""
        write_per_token_pages(
            self.output_dir, latest_rows, summaries, iteration,
            counting_started_at=self.counting_started_at,
        )
