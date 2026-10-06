# Depth sweep v1.1: 16-layer start / 3 depths / 30 GB additional disk budget

Read **DEPTH_SWEEP.md**. Use **run_depth.sh**, not run.sh:

```bash
bash run_depth.sh plan
bash run_depth.sh all
```

Reuse the existing venv and data. For a new environment only:
`PIP_NO_CACHE_DIR=1 bash scripts/setup.sh`.

Default output: `runs/depth_16_3_disk30`. Up to 96 runs. Final model weights and
reports are kept; completed optimizer states are retired after export verification.
Interrupted jobs remain exactly resumable. Disk capacity may reduce the endpoint
below 70% of the VRAM layer limit. A v1 five-depth output cannot be resumed as v1.1.
