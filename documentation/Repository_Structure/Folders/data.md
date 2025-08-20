# `data/`

This directory contains datasets and scripts for preparing them for training.

## Key Files and Their Purpose

| File/Folder | Description |
|---|---|
| `combine_datasets.py` | A script for combining multiple datasets into a single dataset. |
| `create_new_dataset.sh` | A script for creating a new dataset directory with a template structure. This is the recommended way to add a new dataset to the repository. |
| `README.md` | A README file with information about the datasets. |
| `whisper_install.sh` | A script for installing the Whisper ASR model, which can be used for speech-related tasks. |
| `...` | Each subdirectory contains a specific dataset and a `get_dataset.sh` script for downloading and preparing it. The `prepare.py` script in each subdirectory contains the tokenization logic for that dataset. |