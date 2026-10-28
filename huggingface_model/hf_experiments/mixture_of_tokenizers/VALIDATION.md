# Validation record

Recorded 2026-10-21 UTC. These checks ran on CPU. The RTX 4090 configuration is supplied for execution on the target GPU; GPU memory use, throughput, and real-corpus language quality have not been measured.

| Check | Result |
| --- | --- |
| Automated suite | 27 tests passed in 5.02 seconds |
| Dependency consistency | `pip check` passed |
| Full-size tokenizer export | All four languages exported exact 10,000 and 20,000 private lexical tokens |
| Global vocabulary sizes | 40,271 and 80,271 |
| Tokenizer invariants | Nested lexical vocabularies, allowed scripts, singleton native digits, lossless round trips, exactly 256 shared IDs |
| CPU training smoke | 40 optimizer steps; 35,295 predicted tokens; checkpoint save/load and validation completed |
| Training resume | Six continuous steps exactly matched three steps plus save/reload plus three steps: maximum parameter difference 0.0; both consumed 5,275 predicted tokens |
| Sweep integration | Sequential runner completed training, checkpointing, and evaluation on the smoke configuration |
| Interrupted sweep | Regression test confirms an interrupted child run stops the sweep before evaluation or the next vocabulary size |
| Evaluation | Native and byte routes completed on held-out synthetic documents; analytic metric tests passed |
| Generation | Cached, manually routed generation completed from a saved smoke checkpoint |
| Data access | Two documents streamed successfully from each configured language source at its pinned revision |
| Attention implementation | Runtime inspection confirmed GPT-NeoX SDPA attention |

The full-size tokenizer audit used 15,000 artificial documents per language and a 50,000-token candidate stream. It tests vocabulary construction, not the quality of real-language segmentation. Its machine-readable results are in `validation/full-vocab-verification.json`.

The 40-step synthetic training run reduced mean validation route NLL from 4.5500 at step 20 to 3.6059 at step 40. This is only an integration check. It is not evidence of useful Chinese, Korean, Japanese, or English ability, and it must not be compared with benchmark results.

Core versions were PyTorch 2.5.1, Transformers 4.46.3, Tokenizers 0.20.3, and Datasets 3.1.0. The test machine had no available CUDA device. The repository includes no trained models or downloaded corpora.

## Reproduce the checks

Follow the setup and CPU smoke commands in `README.md`, then run `python -m pytest -q`. The full-size and exact-resume JSON files record additional one-off checks performed while implementing this package; they are not automatically rerun by pytest.

## First GPU run

1. Prepare the pinned corpora and train both real-language tokenizer sizes.
2. Run a short 4090 smoke with a separate output root and `--max-steps 20`; inspect loss, peak VRAM, and checkpoint reload.
3. Run the default 20M-token pilots in a fresh output root. Keep the prepared corpus and tokenizer manifests with the results.
4. Compare held-out bits per UTF-8 byte under matched-text and fixed-token context settings, plus fallback rates, exposure, GPU time, and parameter counts.
5. Freeze task-specific benchmark adapters and prompts before reporting official task scores. The included benchmark manifest is a plan, and the normalized JSONL scorer is a generic interface.

Use separate output directories for pilots and final schedules. Changing the total token budget when resuming changes the learning-rate decay horizon; it does not recreate a fresh run at the larger budget.
