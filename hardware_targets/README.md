# Hardware profiling targets

This directory contains automation helpers for running exported ExecuTorch programs on specific devices.

## Android

Use `android/profile_pte.py` to stage a runner and `.pte` file onto an attached device via `adb`, invoke the runner, and parse energy/latency metrics emitted between `EXECUTORCH_METRICS_BEGIN` and `EXECUTORCH_METRICS_END` markers.
