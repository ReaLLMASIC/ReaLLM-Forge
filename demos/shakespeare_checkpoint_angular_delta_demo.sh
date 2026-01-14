#!/bin/bash
# demos/shakespeare_checkpoint_angular_delta_demo.sh
# Demonstrates training two Shakespeare character models and comparing their checkpoints.

set -euo pipefail

bash ./demos/fake_ptq_uniform_eval_demo.sh

DATA_DIR="data/shakespeare_char"
OUT_DIR_A="out_fake_ptq_shakespeare"

for i in `seq 3 8`; do
	OUT_DIR_B="out_fake_ptq_shakespeare_uniform_sweep/${i}bit"
	CKPT_PATH_A="${OUT_DIR_A}/ckpt.pt"
	CKPT_PATH_B="${OUT_DIR_B}/ckpt.pt"
	HIST_DIR="comparison_demo_histograms_fp32_${i}bit"
	COMPARISON_DIR="comparison_demo_reports_fp32_${i}bit"
	REGEX_ATTN="transformer\\.h\\.[0-4]+\\.attn\\.(c_attn_(q|k|v)|c_proj)\\.weight"
	REGEX_MLP="transformer\\.h\\.[0-4]+\\.mlp\\.(c_fc|c_proj)\\.weight"

	# Ensure the dataset is available.
	pushd "${DATA_DIR}" > /dev/null
	bash get_dataset.sh
	popd > /dev/null

	if [[ ! -f "${CKPT_PATH_A}" ]]; then
		echo "Expected checkpoint not found at ${CKPT_PATH_A}" >&2
		exit 1
	fi

	if [[ ! -f "${CKPT_PATH_B}" ]]; then
		echo "Expected checkpoint not found at ${CKPT_PATH_B}" >&2
		exit 1
	fi

	echo "\nRunning checkpoint regex explorer to compare attention projection weights..."
	python3 analysis/checkpoint_analysis/checkpoint_regex_explorer.py \
		"${CKPT_PATH_A}" \
		--compare-ckpt "${CKPT_PATH_B}" \
		--comparison-csv "attn.csv" \
		"${REGEX_ATTN}" \
		--histogram-dir "${HIST_DIR}" \
		--histogram-bins 40

	echo "\nRunning checkpoint regex explorer to compare MLP projection weights..."
	python3 analysis/checkpoint_analysis/checkpoint_regex_explorer.py \
		"${CKPT_PATH_A}" \
		--compare-ckpt "${CKPT_PATH_B}" \
		--comparison-csv "mlp.csv" \
		"${REGEX_MLP}" \
		--histogram-dir "${HIST_DIR}" \
		--histogram-bins 40

	echo "\nVector angle CSVs and histograms saved under ${COMPARISON_DIR}, with per-checkpoint histograms in ${HIST_DIR}."
done
