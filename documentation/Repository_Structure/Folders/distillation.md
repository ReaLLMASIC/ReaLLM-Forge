# `distillation/`

This directory contains scripts for model distillation, a technique for transferring knowledge from a large "teacher" model to a smaller "student" model.

## Key Files and Their Purpose

| File/Folder | Description |
|---|---|
| `angle_optimization.py` | A script for optimizing the angle between feature vectors. This can be used to encourage the student model's feature vectors to align with the teacher model's feature vectors. |
| `get_feature_vectors.py` | A script for extracting feature vectors from a model. These feature vectors can then be used as targets for the student model during distillation. |