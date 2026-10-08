# `avg_top1_correct` and top-1 loss variations

`avg_top1_correct` is logged during validation as the mean fraction of tokens
whose highest-probability prediction equals the target token. In `Trainer`, the
validation pass computes probabilities, takes the top-1 index with `probs.max`,
and averages `(top1_idx == Y).float()` into `top1_correct`. The TensorBoard tag
is written as `<dataset>/avg_top1_correct`.

Because the metric uses `argmax`, it is not directly differentiable. Losses that
optimize for this metric therefore need to use one of two patterns:

1. **Discrete routing, continuous gradient**: use top-1 correctness only under
   `torch.no_grad()` to decide which examples should receive more or less normal
   cross-entropy gradient.
2. **Differentiable surrogate**: replace the hard top-1 flip with a smooth logit
   gap penalty that rewards the target logit for crossing the strongest
   competing logit.

## Existing losses that already follow this idea

- `top1_focus`: adds a batch-level penalty proportional to
  `1 - batch_top1_correct`.
- `skip_correct_top1`: drops tokens that are already top-1 correct.
- `attenuated_correct_top1`: keeps all tokens, but down-weights top-1-correct
  tokens by `--correct_top1_attenuation`.
- `distance_attenuated_top1`: attenuates cross entropy according to the logit
  distance between the current top prediction and the target.
- `top1_margin` and `top1_ratio`: differentiable gap-style objectives that push
  the target above the best non-target token.

## New variations added here

### `top1_corrective_ce`

This loss uses a batch-local analogue of `avg_top1_correct`:

```text
batch_error_rate = 1 - mean(argmax(logits) == targets)
loss = mean(CE(token) * (1 + boost * batch_error_rate) for top1-wrong tokens,
            CE(token) for top1-correct tokens)
```

Use it when you want training to concentrate more on batches where the current
model is making many top-1 mistakes without completely removing the gradient
from already-correct tokens.

Example:

```bash
python train.py --loss_fn top1_corrective_ce --top1_corrective_boost 1.0
```

### `top1_confidence_gap`

This loss keeps standard cross entropy and adds a smooth top-1 surrogate:

```text
loss = CE + beta * mean(softplus(best_non_target_logit - target_logit))
```

The penalty is small when the target already beats the strongest competitor and
large when another token is still ahead. Unlike `avg_top1_correct`, it provides a
usable gradient before the top-1 decision flips.

Example:

```bash
python train.py --loss_fn top1_confidence_gap --top1_confidence_gap_beta 0.5
```

The confidence-gap weight defaults to `0.5`.

### `top1_corrective_confidence_gap`

This merged objective applies both mechanisms at once: its cross-entropy term
up-weights top-1-incorrect tokens using `--top1_corrective_boost`, and it adds
the smooth strongest-competitor penalty using `--top1_confidence_gap_beta`.

```text
loss = corrective_CE(boost)
     + beta * mean(softplus(best_non_target_logit - target_logit))
```

Example using the default confidence-gap weight of `0.5`:

```bash
python train.py --loss_fn top1_corrective_confidence_gap \
  --top1_corrective_boost 1.0
```

The dedicated `explorations/top1_corrective_confidence_gap_sweep.yaml` runs a
cross-entropy baseline and the full `0.25`, `0.5`, `0.75` Cartesian grid for
both merged-loss hyperparameters.

## Practical sweep suggestion

Start with cross entropy as a baseline, then compare one discrete-weighting loss
and one differentiable-surrogate loss:

```yaml
- loss_fn: ["cross_entropy"]
- loss_fn: ["top1_corrective_ce"]
  top1_corrective_boost: [0.5, 1.0, 2.0]
- loss_fn: ["top1_confidence_gap"]
  top1_confidence_gap_beta: [0.25, 0.5, 1.0]
- loss_schedule: ["0:cross_entropy,10000:top1_confidence_gap"]
```

Track both validation loss and `<dataset>/avg_top1_correct`; top-1-oriented
losses can improve discrete accuracy while sometimes worsening calibration or
cross-entropy.
