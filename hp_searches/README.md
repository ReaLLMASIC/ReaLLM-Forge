# Hyperparameter Searches

This is a folder for hyperparameter searches, currenlty we have a greedy search
algorithm which allows us to inspect the balance of parameters when growing a
model from a baseline.

## Usage

1. Create a baseline.yaml file and store in the hp_searches/ dir

```yaml
# static
dataset: "minipile"
use_rotary_embeddings: True
use_abs_pos_embeddings: False
use_flash_lobo: true
use_flash_lobo_per_head: true
use_concat_heads: False
attention_variant: "infinite"
max_iters: 5000
eval_interval: 5000
compile: True
batch_size: 64
block_size: 256
n_cproj: 1
# changeable things
n_layer: 1
n_head: 1
n_embd: 32
mlp_size: 32
n_qk_head_dim: 32
n_v_head_dim: 32
flash_lobo_log_const: 0.1
```

2. Create a bash script for running this and store in the hp_searches/ dir

Note: list parameter names in the same order as their min step size:
```bash
#!/bin/bash
# lobo_attnhead_search.sh

python3 hyperparam_search.py \
  --orig_settings ./hp_searches/lobo_attnhead_search.yaml \
  --param_names \
        n_layer \
        n_head \
        n_cproj \
        n_embd \
        mlp_size \
        n_qk_head_dim \
        n_v_head_dim \
  --increments \
        1 \
        1 \
        1 \
        32 \
        32 \
        32 \
        32 \
  --random_iterations 1 \
  --iterations 1 \
  --num_iterations 20000 \
  --efficiency_target params \
  --override max_iters=20000 batch_size=64 \
  --results_file results.yaml
```

- `random_iterations` is the number of trials to average for per hp, e.g. n_layer
maybe we can try with 3 random seeds, and get the average to try to fish through
noise.
- `iterations` is the depth of the search per parameter (e.g. try adding 1
    n_layer then trying adding a second n_layer, and see which is best)
- `num_iterations` is the number of growth steps
- `override` don't use this at first, but allows you to manually override
    settings for any hp_search already started (just stop and resume with these
    overrides to for example increase the max_iters, and to unblock the model
    when delta score gets too close to noise levels)
 - `results_file` where to store results for viewing with `view_hp_log.py`
  - `efficiency_target` chooses the cost metric for efficiency: `params`
  (default), `vram` for peak GPU memory, or `iter` for average iteration
  latency.


1. Run bash script from main directory

```bash
bash ./hp_searches/lobo_attnhead_search.sh
```

To quickly exercise the three efficiency metrics, see
`hp_searches/test_efficiency_targets.sh` which runs a tiny search using
the accompanying `efficiency_targets_demo.yaml` baseline.

1. View with `view_hp_log.py`

```bash
python view_hp_log.py results.yaml
```

Note, this will auto-refresh.

3. Monitor via the above, and update max_iters as necessary.

The hyperparameter_search.py has override features, useful for changing training
settings needed as the model grows, e.g. max_iters, learning rate, batch size,
etc.

## Notes And Observations

#### Mitigating Step Noise:

There are a couple ways to improve step noise:

1. Increase the training data, e.g. max_iters (stay less than 1 epoch)
2. Increase the # random iterations to average (e.g. if data is limited)

### Step Settings Considerations:

1. Too large a step size and we can miss sub-optimizations
2. Too small a step size and we encounter too much noise
3. Increase iteration number -- try multiples of the step size, hoping to get
   past noise. Too night an iteration number is wasteful due to reduction in the
   % correct naturally for parameter even in the right direction.

### Training Step Considerations

1. too many training steps, too minimal delta param.
2. too few training steps, too much noise

There are some "humps" for learned parameters, e.g. absolute position encodings,
which typically resolve after 10,000 iterations or so. Currently we try to set a
at 5000 iteration count (with eval_interval at 5000) then move to 20_000 once we
hit a snag.

## Background

###  Why Grow a Model?

We find that we have a large number of hyperparameters, and the hyperparameters
have a strong interdependency on each other.

This means that we really should continually test compatbility of different
techniques during the search.

One way the team is exploring to do this, is to start a small model, and
continually take a step in the most parameter efficient direction.

In this case, we can ensure compatbility of the different parameters (as they
are continually co-tested), and each step will be the greatest increase in
capability per parameter.

### Limitations of Target Optimization

Our prior attempts at optimization include:

1. Optimizing hyperparameters for set # of total parameters:
    a. the hyperparameter space was still too large
    b. we had a danger that the model would simply increase its number of parameters to fit in the maximum space, making it hard to interpret the ultimate shape.

2. Optimizing for parameter efficiency
    a. the most efficient parameters for % of next token correctness were the initial parameters, so this would tend our models to go to zero-parameters.

### How Growing the Model Addresses 1 and 2

*Continual Pressure for Parameter Efficiency*

In this way we provide, "pressure" for the model to balance its parameters at
each stage of growth.

Essentially we have a gradient at each of the steps for how to grow the model,
and try to normalize by the number of additional parameters required to obtain
that improvement (the most cost efficient step).

There are other types of cost efficiency we can also explore, as well as
specific task targets, that make this an interesting framework to develop.

### Supported Metrics and Planned Extensions

Currently the only supported metric is validation loss or rather 1/exp(val_loss)
which is equal to the probability that the next token is correct.

Later we hope to also include summarization tasks, translation tasks, etc. so
and see what model shapes result from each of these.

Also the "Cost" can be altered as well.

## Next Steps

### Means to increase maxiters automatically upon no feasible direction
We might program a means to have this occur automatically (step size for
max_iters specifically) if we have negative or zero for each of the parameters
directions

### Curve Fitting on Ratios

With the resulting data, we can try to find the curve of parametesr per mlp
size, nad vest val loss vs different characteristics, and ratios of differnet
ones via curve fitting. This could yield insights on balancing parameters
especially towards small language models.
