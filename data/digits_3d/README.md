# Sequential digits with held-out vocabulary characters

This synthetic dataset repeats a configurable sequence of digit-like symbols.
The first ten are `0`–`9`; larger `--num-digits` values add renderable
punctuation. A configurable number of letters are included in the vocabulary
but occur in neither split, making them useful controls when inspecting how
training moves token vectors. `EMBEDDING_DIM` selects the model width; widths
above three are projected into the viewer with a shared three-component PCA.

The demo enables fixed-norm embeddings by default: initialization and every
optimizer update project each vector to radius `sqrt(EMBEDDING_DIM)`. Set
`WTE_FIXED_NORM=false` to retain the original unconstrained training mode, or
set `WTE_FIXED_NORM_VALUE` to choose another radius.

To compare unconstrained, square-root-dimension-radius, and unit-radius
initialization with tied and untied WTE/LM-head weights, run
`demos/digits_3d_trajectory_sweep.sh`. The default is 3 dimensions, 10 trained
symbols, and 10 held-out letters. Its `EMBEDDING_DIMS`, `DIGIT_COUNTS`,
`LETTER_COUNTS`, and `WTE_TYING_MODES` variables accept whitespace-separated
values, so PCA widths such as `8 16 64` remain selectable. Sweep runs train for
10,000 iterations by default; override `SWEEP_MAX_ITERS` as needed.

Set `DROPOUT_PERCENT` in the single demo to regenerate the training split and
resume training without the trailing `DROPOUT_COUNT` trained symbols at that
percentage of the run. The symbols stay in the vocabulary and checkpointed
model, which makes their post-dropout motion directly comparable.

`SCHEDULE_MODE=add` runs the complementary experiment: affected symbols start
absent and enter the dataset at `DROPOUT_PERCENT`. `SCHEDULE_MODE=duty_cycle`
repeatedly includes and excludes them; `DUTY_CYCLE_PERCENT` controls their
included fraction (20–80) within each `DUTY_PERIOD_PERCENT` period.

`OPTIMIZER_MODE` selects `full_muon`, `adam`, `adagrad`, `sgd`, or `rmsprop`.
Full Muon routes every matrix-shaped parameter (including WTE and LM head) to
Muon with zero weight decay, while vector parameters use its auxiliary Adam.
Adam uses `ADAM_WEIGHT_DECAY` (0.1 by default); the other comparison optimizers
use zero weight decay.

The data is generated locally and is released under the repository's license;
there is no external source or additional dataset license.

```bash
python3 data/digits_3d/prepare.py --num-digits 10 --num-letters 10
```

The command creates the standard nanoGPT `train.bin`, `val.bin`, and `meta.pkl`
files. Generated binaries are intentionally ignored by git.
