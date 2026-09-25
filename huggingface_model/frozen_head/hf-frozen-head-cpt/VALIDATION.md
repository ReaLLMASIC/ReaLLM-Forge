# Validation performed

Environment: Python 3.12; PyTorch 2.8.0+cpu; Transformers 4.57.6; Hugging Face Hub 0.36.2. No GPU was available in the authoring environment.

## Passed in the initial CPT version

- 14 pytest cases, including eight parameterized comparisons of chunked CE and all trainable gradients against the standard Hugging Face Qwen2/GPT-2 forward.
- All four primary tying/freezing configurations in the objective/gradient comparisons.
- Head-only freezing leaves input embeddings trainable; full frozen-tensor hashes remain identical after AdamW.
- Packed next-token targets, overlap, and frequency counts.
- Repository-level hash assignment and exact replay stream counters.
- Tiny end-to-end A consolidation and both full/frozen-head B runs using local Hugging Face models and a local tokenizer.
- Four head-swap losses agree at the initial checkpoint.
- Interrupted/resumed training produces bit-for-bit identical CPU model weights and test metrics to uninterrupted training.
- Matched-gain interpolation uses the first chronological crossing and does not extrapolate unreachable gains.
- CUDA device-index resolution and allocation-ceiling arithmetic under mocks; this tests control logic, not CUDA execution or memory usage.
- A separate offline CLI invocation ran `preflight`, `run`, child-process training, and PNG/SVG/CSV report generation successfully.
- Python compilation and shell syntax checks.

## Not measured here

- Real H100 peak VRAM, throughput, kernel behavior, or training duration.
- Full Qwen2.5-1.5B dataset download/training runs through Hugging Face.
- Whether freezing the head mitigates forgetting on the proposed real corpora.
- Compatibility with every supported model family at full size. Objective tests cover tiny Qwen2 and GPT-2; the other supported families use their documented standard decoder/linear-head structure.

Run `python cpt.py preflight --config configs/h100_pilot.json` on the H100 to measure the configured training footprint. It performs actual optimizer updates and original-head evaluation on synthetic input. Follow with the pilot before scaling the sweep. The package contains no fabricated experiment results.

## Task fine-tuning extension

The complete suite now has **23 passing tests**. Added coverage:

- Response-only loss and gradients match the standard Hugging Face forward with prompt/padding masks, for all four primary freezing conditions.
- Numerical answer scoring distinguishes correct answers from format compliance and supports comma/decimal normalization.
- Overlong reference solutions are rejected instead of silently truncating the final answer.
- A complete tiny local math-SFT experiment exercises pretrained baseline evaluation, full/frozen-head training, generated-answer scoring, retention metrics, and reports.
- Interrupted/resumed SFT reproduces uninterrupted CPU weights and final test scores exactly.
- Accuracy matching selects actual first-crossing checkpoints rather than interpolating a discrete metric.
- Standard benchmark result aggregation preserves paired records and rejects changed benchmark samples/protocols.
- A separate offline task CLI run completed data preparation, pretrained baseline evaluation, child-process full/frozen-head SFT, generated-answer scoring, and CSV/PNG/SVG reports.

The **external evaluation harness and remote benchmark datasets were not installed or run here**. Its wrapper uses the documented `HFLM` / `simple_evaluate` API, records dependency and task-code versions, and checks evaluated-record fingerprints. Install `requirements-benchmarks.txt` and run `benchmarks.py suite` on the H100 machine to execute those evaluations. No real GSM8K/HellaSwag/ARC/PIQA/BoolQ results are included.
