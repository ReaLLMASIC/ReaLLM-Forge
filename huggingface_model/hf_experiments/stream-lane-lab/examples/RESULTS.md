# CPU prototype results — v0.1.0

Measured on 2026-10-20. The default viewer checkpoints are small two-layer transformers: width 48, two attention heads, MLP width 144, context 64, and 128 amplitude levels per column. Each column is a sum of three sine waves. No analytic sine predictor, future samples, or phase labels are fed to the model.

## Main 1 → 2 → 3 channel checkpoints

These use five-knot tied SLERP, followed by a short rollout-focused fine-tuning stage. Each stage selects its checkpoint on validation rollout RMSE. They demonstrate usable short forecasts; longer continuations remain imperfect. The same curriculum is shown for all channel counts, rather than choosing a different architecture using the test set.

| Channels | Parameters | Next-step RMSE ↓ | 8-step RMSE ↓ | 32-step RMSE ↓ | Persistence 32-step ↓ | CPU 8-step median |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 49,633 | 0.00545 | 0.05015 | 0.17189 | 0.46095 | 9.2 ms |
| 2 | 49,874 | 0.00817 | 0.09887 | 0.36612 | 0.46373 | 14.2 ms |
| 3 | 50,115 | 0.01190 | 0.11909 | 0.37452 | 0.44780 | 19.5 ms |

RMSE is in the signal units, whose generator is bounded to ±0.9; model range is [-1,1]. “8-step RMSE” pools the first eight recursively generated samples, not only the eighth endpoint. The growing error is real forecast drift, not a display or reconstruction effect. The viewer defaults to eight samples and lets you extend the horizon. Its one-step history uses predictions saved before the actual sample arrived.

Latency is a warm, single-stream measurement with one CPU thread, seven repetitions per horizon, and no concurrent training. It excludes HTTP, rendering, and checkpoint loading. See `benchmark.json` for p95 values; this is a shared virtual CPU, so absolute times vary. Source training-stage timings in raw logs include concurrent experiments and are not clean performance comparisons. No KV cache is used.

## What is included

- Six loadable viewer checkpoints: the three main models above, a one-channel independent table, a one-channel two-knot untied model, and a three-column mixed-geometry model.
- Three original warm-start checkpoints for reproducing the rollout-focused stage.
- Twelve complete experiment records in `results/`, with every-step JSONL losses, run configurations, selected-checkpoint test metrics, and one example trajectory.
- [Comparison report](report/index.html), [loss curves](report/curves.png), CSV/JSON summary, and [environment details](environment.json).
- Forty passing CPU checks. Thirteen CUDA checks are skipped here; these are a required parity gate before relying on Triton on the 4090.

## Interpretation and limits

The categorical geometry is a working implementation, not evidence of a universal best variant. In this small study, tied five-knot SLERP learned more accurate smooth one-channel forecasts than the tested independent and untied two-knot runs, but the exploratory runs differ in initialization/training objective and are not a controlled ranking. The five `tuned-` runs share matched-v1 backbone initialization and example order within the same channel count and dimensions; they still use one training seed. Tied versus relaxed untied also changes the output family.

Training uses 512 phase/amplitude seeds (1000–1511), checkpoint selection uses eight validation seeds (20000–20007), and final evaluation uses 24 distinct seeds (30000–30023). These are data seeds, not 24 independent training runs. The small validation set can select checkpoints that generalize unevenly, visible in the raw metrics. Frequencies are fixed per column: [1/128, 1/47, 1/23] cycles/sample times (1+0.07c). The results test unseen phases/amplitudes in this family, not new frequencies or arbitrary real streams.

Next-step accuracy is substantially stronger than long free-running accuracy. The three-channel 32-step forecast still has substantial error; more knots alone are not established as a remedy. For stronger continuations, expand validation coverage and test longer rollout supervision, model width, and multiple training seeds before making a quality claim. Bounded outputs prevent value explosions but do not ensure correct dynamics.

## Reproduce the rollout stage

The original warm starts predate matched-v1 initialization; their exact files are included. Starting from them reproduces the fine-tuning setup without pretending a fresh current training command regenerates their old initialization. The new stage has a fresh optimizer, not a resumed optimizer state.

```bash
for C in 1 2 3; do
  python -m lane_lab train \
    --init-model "examples/warmstarts/c${C}-slerp5-tied" \
    --steps 600 --batch-size 12 --lr 0.0003 --eval-every 100 \
    --target-mode linear --input-noise 0 \
    --rollout-steps 24 --rollout-gradient full \
    --mse-weight 100 --rollout-weight 300 \
    --output "runs/reproduce-c${C}"
done
python -m lane_lab report --runs runs --output runs/report
```

For new matched architecture sweeps, use the README sweep command and preserve its entire configuration. The supplied full sweep covers the independent tied baseline and tied/untied SLERP at 2, 3, 5, and 9 knots across all three channel counts. That 27-run sweep has not been completed here; do not infer its results from the twelve exploratory runs.

On CUDA, use `python -m pytest tests/test_kernels.py -q` first. The kernels have real forward/backward implementations, but no CUDA hardware was available to compile, execute, or benchmark them in this session. CPU checkpoints use `backend: auto` and therefore use the Torch reference.
