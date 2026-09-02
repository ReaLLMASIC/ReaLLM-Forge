#!/usr/bin/env python3
"""Sweep LayerNorm gain pruning and folded LM-head symmetric quantization."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .core import discover_norms, effective_gain, merge_gain_into_head, set_effective_gain, threshold_gain


def float_range(start: float, stop: float, step: float) -> list[float]:
	if step <= 0 or stop < start:
		raise ValueError("threshold stop must be >= start and step must be positive")
	values, current = [], start
	while current <= stop + step * 1e-9:
		values.append(round(current, 12))
		current += step
	return values


def evaluate(model, tokenizer, tasks: list[str], batch_size: int) -> dict[str, float]:
	"""Evaluate the already modified model through lm-evaluation-harness."""
	try:
		from lm_eval import simple_evaluate
		from lm_eval.models.huggingface import HFLM
	except ImportError as exc:
		raise RuntimeError("Benchmarking requires `pip install lm-eval`") from exc
	lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
	result = simple_evaluate(model=lm, tasks=tasks, batch_size=batch_size)
	flat = {}
	for task, metrics in result["results"].items():
		for name, value in metrics.items():
			if isinstance(value, (int, float)) and not name.endswith("_stderr"):
				flat[f"{task}/{name}"] = float(value)
	return flat


def write_plot(rows: list[dict], path: Path) -> None:
	data = json.dumps(rows).replace("</", "<\\/")
	path.write_text(f"""<!doctype html><meta charset=utf-8><title>LayerNorm surgery sweep</title>
<script src=https://cdn.plot.ly/plotly-2.35.2.min.js></script><h1>LayerNorm surgery sweep</h1>
<label>Metric <select id=m></select></label><div id=p style='height:75vh'></div><script>
const rows={data}; const fixed=new Set(['threshold','bits','zero_channels','nonzero_channels','shaved_parameters','shaved_fraction']);
const metrics=[...new Set(rows.flatMap(Object.keys))].filter(k=>!fixed.has(k)); m.innerHTML=metrics.map(x=>`<option>${{x}}</option>`).join('');
function draw(){{const metric=m.value;const groups=Object.groupBy?Object.groupBy(rows,r=>r.bits):rows.reduce((a,r)=>((a[r.bits]??=[]).push(r),a),{{}});Plotly.newPlot('p',Object.entries(groups).map(([bits,v])=>({{x:v.map(r=>r.threshold),y:v.map(r=>r[metric]),mode:'lines+markers',name:`${{bits}} bit`}})),{{xaxis:{{title:'gain threshold'}},yaxis:{{title:metric}},title:'Benchmark vs pruning threshold and precision'}})}}m.onchange=draw;draw();</script>""", encoding="utf-8")


def main() -> None:
	p = argparse.ArgumentParser(description=__doc__)
	p.add_argument("--model", required=True, help="Hugging Face model id or local directory")
	p.add_argument("--targets", default="final", help="comma-separated exact norm names, 'all', or final")
	p.add_argument("--threshold-start", type=float, default=0.0)
	p.add_argument("--threshold-stop", type=float, default=0.1)
	p.add_argument("--threshold-step", type=float, default=0.01)
	p.add_argument("--bits", default="2,3,4,5,6,7,8", help="comma-separated integer precisions")
	p.add_argument("--tasks", default="hellaswag,arc_easy,arc_challenge,piqa,winogrande,boolq")
	p.add_argument("--batch-size", type=int, default=1)
	p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
	p.add_argument("--output-dir", type=Path, default=Path("layernorm_sweep_results"))
	p.add_argument("--local-files-only", action="store_true")
	args = p.parse_args()
	bits = [int(x) for x in args.bits.split(",")]
	if any(x < 2 or x > 8 for x in bits):
		p.error("--bits values must be from 2 through 8")
	model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=args.local_files_only).to(args.device).eval()
	tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
	norms = discover_norms(model)
	if not norms:
		raise RuntimeError("No LayerNorm/RMSNorm gain modules were found")
	if args.targets == "final":
		target_names = [list(norms)[-1]]
	elif args.targets == "all":
		target_names = list(norms)
	else:
		target_names = [x.strip() for x in args.targets.split(",") if x.strip()]
	missing = set(target_names) - set(norms)
	if missing:
		raise ValueError(f"Unknown target norms: {sorted(missing)}; available: {list(norms)}")
	final_name = list(norms)[-1]
	head_module = model.get_output_embeddings()
	original_head = head_module.weight.detach().clone()
	# Some architectures tie input embeddings and the LM head. Surgery is an
	# output projection optimization and must not silently alter token lookup.
	head_module.weight = torch.nn.Parameter(original_head.clone(), requires_grad=False)
	original_gains = {name: effective_gain(norms[name]).clone() for name in target_names}
	original_final_gain = effective_gain(norms[final_name]).clone()
	rows = []
	for threshold in float_range(args.threshold_start, args.threshold_stop, args.threshold_step):
		for precision in bits:
			with torch.no_grad():
				head_module.weight.copy_(original_head)
				set_effective_gain(norms[final_name], original_final_gain)
				for name, gain in original_gains.items():
					set_effective_gain(norms[name], gain)
					if name != final_name:
						set_effective_gain(norms[name], threshold_gain(gain, threshold))
			final_gain = original_gains.get(final_name, original_final_gain)
			# A non-final target cannot remove LM-head columns; fold its unpruned
			# final gain only so quantization remains available without overstating savings.
			final_threshold = threshold if final_name in target_names else 0.0
			# Quantization happens inside this operation, after thresholding and
			# selecting only the LM-head columns whose folded gain remains non-zero.
			stats = merge_gain_into_head(original_head, final_gain, final_threshold, precision)
			# Dense equivalent used for standard kernels/evaluators. A deployment kernel can
			# consume merged_nonzero + permutation and skip the zero columns entirely.
			full_folded = torch.zeros_like(original_head)
			nonzero_ids = stats.permutation[:stats.nonzero_channels]
			full_folded[:, nonzero_ids] = stats.merged_nonzero
			with torch.no_grad():
				head_module.weight.copy_(full_folded)
				# The final gain is now represented exactly once, in the folded head.
				set_effective_gain(norms[final_name], torch.ones_like(final_gain))
			metrics = evaluate(model, tokenizer, [x.strip() for x in args.tasks.split(",") if x.strip()], args.batch_size)
			rows.append({"threshold": threshold, "bits": precision, "zero_channels": stats.zero_channels,
				"nonzero_channels": stats.nonzero_channels, "shaved_parameters": stats.shaved_parameters,
				"shaved_fraction": stats.shaved_fraction, **metrics})
			print(rows[-1], flush=True)
	args.output_dir.mkdir(parents=True, exist_ok=True)
	columns = list(dict.fromkeys(k for row in rows for k in row))
	with (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader(); writer.writerows(rows)
	with (args.output_dir / "settings.csv").open("w", newline="", encoding="utf-8") as handle:
		writer = csv.writer(handle); writer.writerow(("setting", "value"))
		for key, value in vars(args).items(): writer.writerow((key, value))
	write_plot(rows, args.output_dir / "results.html")


if __name__ == "__main__":
	main()
