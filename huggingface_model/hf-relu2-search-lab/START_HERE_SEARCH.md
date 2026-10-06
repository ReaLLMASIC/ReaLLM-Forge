# Start here: quick depth sweep and HF attention search

Extract this archive into a **fresh folder**. Continue older runs with their
original lab: their source fingerprints intentionally differ from this release.
No training data, model checkpoints, Python environment or measured GPU results
are bundled. Source, tests, preparation scripts and reporting are included.

## Environment

Either reuse the lab's pinned environment by setting `LAB_PYTHON` to its absolute
Python path, or install a separate one:

```bash
cd hf-relu2-search-lab
bash scripts/setup.sh
```

The tested API baseline is Transformers 4.44.2 / tokenizers 0.19.1 and PyTorch
2.9.1. Setup installs the CUDA build for the 4090. Reusing a compatible environment
and prepared token data avoids extra disk consumption on your 30 GB-free drive.

## First: the requested 512-wide, 2048-context depth sweep

```bash
bash run_quick_depth.sh plan
bash run_quick_depth.sh all --layers 16 24 32
```

This is six runs: softmax Flash SDPA and fused ReLU² at each depth, four heads of
128 channels, FFN width 2048, QK norm/learned scale/RoPE, Muon decay zero and the
audited AdamW auxiliary groups. Each run trains for 100,007,936 tokens with no
time deadline. Layer choices are configurable; they are not pre-certified fits
on a 4090. The command checks all requested GPU shapes before training.

Model counts at 16/24/32 layers are 76.10M/101.28M/126.45M. The estimated peak
additional data/checkpoint/report storage is 9.48 GB plus a 2 GB free-space
reserve; environment installation is separate. Full checkpoints are retained
for interrupted runs, and completed optimizer states are retired after final
model verification. See [QUICK_DEPTH_512_2048.md](QUICK_DEPTH_512_2048.md).

```bash
# Read while training; no localhost connection is required.
watch -n 10 cat runs/quick_d512_ctx2048/live/report.md
# Resume an interrupted run with the same settings.
bash run_quick_depth.sh train --layers 16 24 32 --resume
```

Open `runs/quick_d512_ctx2048/live/index.html` directly for the live plots.
PNG/PDF figures show validation versus training tokens and best/final validation
versus layers. Partial progress stays separate from completed-budget comparisons.

## Then: the HF hyperparameter-search adaptation

```bash
bash run_hf_search.sh plan
bash run_hf_search.sh all --data data/fineweb_100m
```

The bounded default is one 384-wide, 16-layer architecture at context 1024, with
three heads and 128-wide QK/V, comparing softmax, ReLU² and linear16. It uses the
same approximately 100M-token budget for every arm. `data/fineweb_100m` reuses the
quick sweep's stream if already prepared; otherwise `all` prepares it.

The new model allows arbitrary head count independent of residual width and
separate `n_qk_head_dim` / `n_v_head_dim`. For example, `512/3/128/64` is valid.
Equal widths reuse the direct fused path; unequal widths use an explicitly
labeled padding bridge around the same kernel. GPU overhead is not measured here.

Use `configs/hf_search_random_example.json` for a bounded random sample of two
architectures (six attention runs), or edit a copied JSON preset for a finite grid:

```bash
bash run_hf_search.sh plan --preset configs/hf_search_random_example.json \
  --output runs/hf_random2
bash run_hf_search.sh all --preset configs/hf_search_random_example.json \
  --output runs/hf_random2 --data data/fineweb_100m
```

Read `runs/hf_search/report.md` or open its `report.html` directly. Reports include
partial validation curves, completed-run comparisons, loss versus training time,
CSV results, backend labels and memory peaks. The HF search retains completed
optimizer state by default; `--retire-completed` explicitly trades completed-job
optimizer resume for storage savings. Both launchers enforce disk and VRAM checks.

## Inspect the implementation and study

| File | What to inspect |
|---|---|
| `docs/REALLM_FORGE_REVIEW.md` | Current repository search behavior, identified issues, exact pinned source links |
| `docs/ADAPTATION_GUIDE.md` | Projection shapes, attention diagram, HF API examples, optimizer routing, feature priorities |
| `docs/HF_SEARCH.md` | Search-space format, run/resume commands, reports and limitations |
| `hf_model/configuration_search.py` | Independent residual/head dimensions and explicit supported settings |
| `hf_model/modeling_search.py` | Projections, RoPE/QK normalization, padding, fused dispatch and HF model |
| `hf_search/hf_trainer.py` | Optional native Muon/AdamW bridge for HF Trainer |
| `tests/test_search_model.py` | Outputs/gradients, masks, cache, HF export and CUDA tests |
| `tests/test_hf_search.py` | Matched budgets, exact resume, failures and reports |
| `SEARCH_VALIDATION.md` | Validation actually executed for this release |

The finite-search runner is a concrete adaptation, not a wholesale port of
upstream greedy growth, Vizier, per-layer configuration lists or recurrent memory.
Native unequal-width kernels and summed/GQA heads are documented next steps.
