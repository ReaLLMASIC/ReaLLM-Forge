#!/usr/bin/env bash
set -euo pipefail

# Run sweep with longest-bytes trimming strategy
python latin_punct_router_eval.py \
  --model_name google/gemma-3-270m-it \
  --route_mode latin_punct_only \
  --latin_trim_sweep \
  --latin_trim_strategy longest_bytes \
  --latin_trim_sweep_max 80 \
  --latin_trim_sweep_step 10 \
  --split "validation[:1%]" \
  --max_samples 100 \
  --max_target_tokens 64 \
  --example_split "validation[:20]" \
  --sweep_examples 10 \
  --example_max_new_tokens 64 \
  --report_dir latin_trim_reports_longest_bytes \
  --byte_fallback

# Run sweep with highest-id trimming strategy
python latin_punct_router_eval.py \
  --model_name google/gemma-3-270m-it \
  --route_mode latin_punct_only \
  --latin_trim_sweep \
  --latin_trim_strategy highest_id \
  --latin_trim_sweep_max 80 \
  --latin_trim_sweep_step 10 \
  --split "validation[:1%]" \
  --max_samples 100 \
  --max_target_tokens 64 \
  --example_split "validation[:20]" \
  --sweep_examples 10 \
  --example_max_new_tokens 64 \
  --report_dir latin_trim_reports_highest_id \
  --byte_fallback

# Run quantization sweep on 100% latin+punct(+byte) candidate set
python latin_punct_router_eval.py \
  --model_name google/gemma-3-270m-it \
  --route_mode latin_punct_only \
  --quantization_sweep \
  --quant_group_size 32 \
  --split "validation[:1%]" \
  --max_samples 100 \
  --max_target_tokens 64 \
  --quant_report_dir quantization_reports \
  --byte_fallback

# Run trimmed-vocab two-pass sweep:
# first pass low-precision shortlist, second pass rerank budget (top-N settable).
python latin_punct_router_eval.py \
  --model_name google/gemma-3-270m-it \
  --two_pass_trim_sweep \
  --two_pass_first_bits 4 \
  --two_pass_first_mode group32_asymmetric \
  --two_pass_first_configs group32_asymmetric:4,group32_symmetric:4,vector_asymmetric:4,vector_symmetric:4,group32_asymmetric:3,group32_symmetric:3,vector_asymmetric:3,vector_symmetric:3 \
  --two_pass_first_group_size 32 \
  --two_pass_second_topn_values 1,10,100,1000,10000 \
  --two_pass_second_dtype float16 \
  --latin_trim_strategy longest_bytes \
  --latin_trim_sweep_max 80 \
  --latin_trim_sweep_step 10 \
  --split "validation[:1%]" \
  --max_samples 100 \
  --max_target_tokens 64 \
  --two_pass_report_dir two_pass_trim_reports \
  --byte_fallback

# Build combined graph: routed(longest-bytes) + routed(highest-id) + full LM head
python - <<'PY'
import csv
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def read_csv(path):
    xs, full, routed = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(int(float(row["latin_trim_percent"])))
            full.append(float(row["top1_full_percent"]))
            routed.append(float(row["top1_routed_percent"]))
    return xs, full, routed

xs_a, full_a, routed_a = read_csv("latin_trim_reports_longest_bytes/latin_trim_sweep.csv")
xs_b, full_b, routed_b = read_csv("latin_trim_reports_highest_id/latin_trim_sweep.csv")

if xs_a != xs_b:
    raise RuntimeError(f"Mismatched sweep x-values: {xs_a} vs {xs_b}")

quant_labels, quant_vals = [], []
with open("quantization_reports/quantization_sweep.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        bits = int(row["bits"])
        mode = row["mode"]
        label = "baseline" if bits == 0 else f"{mode}-{bits}b"
        quant_labels.append(label)
        quant_vals.append(float(row["top1_percent"]))

two_pass_series = {}
with open("two_pass_trim_reports/two_pass_trim_sweep.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (
            row["candidate_variant"],
            row.get("first_pass_mode", "unknown"),
            int(row.get("first_pass_bits", 0)),
            int(row["second_pass_topn"]),
        )
        two_pass_series.setdefault(key, ([], []))
        two_pass_series[key][0].append(int(float(row["latin_trim_percent"])))
        two_pass_series[key][1].append(float(row["top1_two_pass_percent"]))

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
ax0, ax1 = axes
ax0.plot(xs_a, full_a, marker="o", label="Full LM head")
ax0.plot(xs_a, routed_a, marker="o", label="Routed (longest_bytes)")
ax0.plot(xs_b, routed_b, marker="o", label="Routed (highest_id)")
for (variant, mode, bits, topn), (xs, ys) in sorted(two_pass_series.items(), key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3])):
    ax0.plot(xs, ys, marker="o", label=f"Two-pass {variant} ({mode}/{bits}b, top{topn})")
ax0.set_xlabel("Latin trim percent")
ax0.set_ylabel("Top-1 accuracy (%)")
ax0.set_title("Trim strategies vs Full LM head")
ax0.grid(True, alpha=0.3)
ax0.legend()

ax1.bar(range(len(quant_labels)), quant_vals)
ax1.set_xticks(range(len(quant_labels)))
ax1.set_xticklabels(quant_labels, rotation=70, ha="right")
ax1.set_ylabel("Top-1 accuracy (%)")
ax1.set_title("Quantization sweep (latin+punct+byte)")

plt.tight_layout()
out_path = "latin_trim_reports_combined_accuracy.png"
plt.savefig(out_path, dpi=180)
print(f"Wrote combined graph: {out_path}")

# Write combined raw-score CSV for easier inspection/spreadsheet use.
combined_rows = []
for pct, full, routed in zip(xs_a, full_a, routed_a):
    combined_rows.append({
        "series": "full_lm_head",
        "latin_trim_percent": pct,
        "top1_percent": full,
        "extra": "from longest_bytes sweep",
    })
    combined_rows.append({
        "series": "routed_longest_bytes",
        "latin_trim_percent": pct,
        "top1_percent": routed,
        "extra": "",
    })
for pct, routed in zip(xs_b, routed_b):
    combined_rows.append({
        "series": "routed_highest_id",
        "latin_trim_percent": pct,
        "top1_percent": routed,
        "extra": "",
    })
for (variant, mode, bits, topn), (xs, ys) in sorted(two_pass_series.items(), key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3])):
    for pct, score in zip(xs, ys):
        combined_rows.append({
            "series": f"two_pass_{variant}_{mode}_{bits}b_top{topn}",
            "latin_trim_percent": pct,
            "top1_percent": score,
            "extra": f"candidate_variant={variant};first_pass_mode={mode};first_pass_bits={bits};second_pass_topn={topn}",
        })
for label, val in zip(quant_labels, quant_vals):
    combined_rows.append({
        "series": "quantization_bar",
        "latin_trim_percent": "",
        "top1_percent": val,
        "extra": label,
    })

combined_csv_path = "latin_trim_reports_combined_scores.csv"
with open(combined_csv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["series", "latin_trim_percent", "top1_percent", "extra"])
    writer.writeheader()
    writer.writerows(combined_rows)
print(f"Wrote combined CSV: {combined_csv_path}")

# Also emit an interactive Plotly HTML view.
fig = go.Figure()
fig.add_trace(go.Scatter(x=xs_a, y=full_a, mode="lines+markers", name="Full LM head"))
fig.add_trace(go.Scatter(x=xs_a, y=routed_a, mode="lines+markers", name="Routed (longest_bytes)"))
fig.add_trace(go.Scatter(x=xs_b, y=routed_b, mode="lines+markers", name="Routed (highest_id)"))
for (variant, mode, bits, topn), (xs, ys) in sorted(two_pass_series.items(), key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3])):
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=f"Two-pass {variant} ({mode}/{bits}b, top{topn})"))
fig.update_layout(
    title="Trim strategies + two-pass variants (interactive)",
    xaxis_title="Latin trim percent",
    yaxis_title="Top-1 accuracy (%)",
)
plotly_html_path = "latin_trim_reports_combined_accuracy.html"
fig.write_html(plotly_html_path, include_plotlyjs="cdn")
print(f"Wrote interactive HTML: {plotly_html_path}")
PY
