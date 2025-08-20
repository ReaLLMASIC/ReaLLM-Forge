# `explorations/`

This directory contains configuration files for experiments and hyperparameter sweeps. These files are used by the `optimization_and_search/run_experiments.py` and `hyperparam_search.py` scripts.

## Key Files and Their Purpose

| File/Folder | Description |
|---|---|
| `...` | Each `.json` or `.yaml` file contains a list of configurations for `run_experiments.py`, or a single configuration for `train.py`. These files are the primary way to define and manage experiments in the repository. |