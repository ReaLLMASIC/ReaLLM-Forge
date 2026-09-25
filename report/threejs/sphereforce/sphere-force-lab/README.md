# Sphere Force Lab

An interactive, entirely client-side experiment for the native geometry of tied token rows in a tiny causal Transformer.

## Experiment

- configurable residual/model dimension `2–128`; default `3`
- active master row radius `sqrt(modelDim)` in the full model space
- `1–8` pre-RMSNorm decoder blocks; attention + MLP, attention only, or MLP only
- `1–16` heads; independent Q/K and V/O widths per head (`1–128`)
- attention hidden width is the concatenated value width `heads × valueHeadDim`; linked controls edit the same quantity
- MLP width `1–1024`; GELU (tanh approximation), ReLU, or squared ReLU
- learned absolute position embeddings or full adjacent-pair RoPE on Q/K (base 10,000, even Q/K width required)
- configurable maximum context `1–256`, also the actual length of every nonempty training sequence
- configurable `1–100` targeted tokens in a repeating cyclic sequence
- optional seeded first-order Markov transition matrix over the original numeric target IDs
- configurable batch size (`1–100`) of shifted cyclic sequences
- configurable `0–100` randomly initialized untargeted rows; more can be placed on the sphere during training, up to 100 total
- one shared tied WTE / LM-head table with every active row constrained to the sphere
- optimizer: manual AdamW or RMSProp
- every active tied row is reprojected after every optimizer update
- TensorFlow.js uses its CPU backend only

The selected iteration is defined **after** its spherical projection and **before** the next optimizer update. Every displayed frame stores row positions, raw tied-row gradients, constrained forces, realized optimizer displacements, all batch hidden states, and the active softmax log partition.

The iteration limit is independent of model setup. Raising it with **Set + resume** continues the current weights, optimizer moments, manually placed rows, and complete timeline. The timeline can replay all collected frames at `0.25×–4×` speed without changing the model.

## Architecture and higher-dimensional views

Architecture controls are drafts until **Apply + reset timeline**. Reset uses the seed and all architecture/experiment/QAT settings, recreates parameters and optimizer state, and clears the timeline. Invalid dimensions and oversized configurations retain the old run. Defaults retain D=3, one head/block, Q/K=3, V=3, MLP=12, absolute positions, and context=10. Direct simulator callers that omit architecture use context equal to their original target count.

Each attention block projects D→H·dQK independently for queries and keys and D→H·dV for values. Causal attention is scaled by sqrt(dQK); concatenated head outputs pass through Wₒ of shape [H·dV,D]. Model dimension need not divide by head count. Each enabled sublayer uses pre-RMSNorm and a residual connection; final RMSNorm feeds the shared tied output table. MLP biases remain enabled. RoPE rotates projected Q/K, not values, and allocates no learned positional table. MLP-only blocks have no cross-position mixing and require learned absolute embeddings. Inactive sublayer parameters are not allocated. Initialization scales vary with input width to preserve the original default scales while increasing dimensions.

Training, normalization, tangent projection, hidden-state means, pair distances, and numerical diagnostics use every model dimension. The 3D viewer crops to coordinates 1–3 (z=0 for D=2) without renormalizing. High-dimensional rows and projected hidden means may therefore lie inside the reference sphere. The probe is a coordinate-sphere slice with coordinates 4…D fixed at zero, and its tangential arrows omit forces outside this slice. Existing-row arrows show the first three components of the full-D tangent vector. The 2D model displays the XY circle and disables the spherical probe surface. Insertion uses the displayed coordinate slice, initializing other coordinates to zero; all coordinates can evolve on later updates.

Parameter counts exclude inactive reserved vocabulary capacity. The CPU viewer checks combined compute limits (2 million allocated parameters, 4 million attention-score cells across layers, and an approximate 100 million forward multiply-accumulates) before allocation. History estimates include full hidden states, full row vectors, and display copies; reset/limit changes reject estimates above 1 GiB. These are interactive safeguards, not hardware-independent performance guarantees. AdamW retains beta1=.9 and beta2=.99; RMSProp decay=.99. Inactive reserved vocabulary rows no longer receive weight decay; insertion initializes a new constrained row and zero moments. The displayed move/force cosine compares the prior observed move to the current force, in full dimension.

## Dataset source: direct cycle or Markov matrix

**Dataset source → Training dataset** defaults to **Direct cycle · fast**. It retains the existing repeated-cycle training data and cached input/label tensors until membership changes. It does not allocate or traverse a transition matrix, build sampling tables, or draw random numbers on the direct path. Select **Markov transition matrix** to use probabilities instead; dataset edits are drafts until **Apply + reset timeline**, which also resets the model, optimizer, schedules, and history. Invalid matrices leave the live run intact.

For N original targeted tokens, supply an N×N matrix P. Row i is the current token, column j the next token, and P[i,j] is the conditional transition probability. Entries must be finite and nonnegative; every row must sum to 1 within 1e-6 (rounding-sized discrepancies are normalized). Edit a selected row, choose the cycle/uniform preset, or paste a JSON nested array or numeric rows separated by commas/whitespace. **Load into editor** parses the pasted text; **Normalize rows** explicitly converts positive row weights into probabilities. All-zero rows are invalid. Changing N does not silently resize or discard an existing matrix; load or choose a new N×N matrix. Added untargeted letters are not matrix states.

For batch size B, context length T, and ordered included IDs C, sequence b starts at C[b mod |C|]. Sample x[t+1] from P[x[t],:] and train on input x[t] with the sampled next-token label x[t+1], for t=0…T−1. These are contiguous first-order Markov sequences with round-robin starting states, **not draws from a stationary distribution**, and there is no burn-in. The objective is ordinary sampled next-token CE, not soft-label CE against a row of P. Reducible chains and self-loops are permitted; some included tokens may have little or zero target exposure. Diagnostics distinguish allowed membership from actual sampled targets.

- **Fresh batch each iteration** (Markov default) uses the dataset seed and iteration index. Snapshot t stores gradients, hidden means, fields, and a first-sequence preview for the same batch that drives update t→t+1. After the update, a new batch is drawn for snapshot t+1. Loss can fluctuate because both model and sample change. It is not a fixed-validation loss or the exact expected Markov CE.
- **Fixed seeded batch** uses sampling index zero and caches those sequences until membership changes. The model may fit that finite realization rather than the full transition distribution.
- The dataset RNG is independent of model initialization. Same model settings, dataset seed, and target schedule reproduce the run. QAT changes, insertion, inspection, and same-step reevaluation do not advance sampling. Membership changes deterministically rebuild the batch for that step; they do not preserve the old sample.
- A row with one allowed nonzero destination uses direct integer lookup without an RNG draw. If all included rows are deterministic, the entire batch is cached even in fresh-batch mode. A cycle matrix with all targets included exactly matches direct-cycle data and updates. Stochastic rows use precompiled sparse cumulative distributions and binary search; these tables rebuild only when membership changes. The causal mask is reused while context length stays constant.

Target dropout masks excluded IDs from starts and destinations. For each included source i, remaining probabilities become P[i,j] / sum(P[i,k] for included k). If that sum is zero, the source temporarily self-loops; the UI names the affected rows. This explicit fallback avoids reintroducing an excluded token. It is not equivalent to bypassing a removed state: a cycle matrix can acquire a self-loop after dropout, whereas direct-cycle mode reconnects its remaining sequence. Restoration uses the unmodified original matrix. If no targets remain, both modes preserve the existing idle-clock behavior. Replay uses saved batch metadata and hidden states without resampling.

## Target dropout, restoration, and duty cycles

In **Target inclusion**, select any original numeric target and choose **Exclude target**, **Restore target**, or **Repeating duty cycle**. **Apply policy now** acts at the current live iteration; **Schedule policy** uses the entered start iteration. These actions pause training and preserve all weights, optimizer moments, QAT settings, and earlier frames. Press **Run** to continue. Scheduled rules beyond the training limit remain pending until that limit is extended.

Policies are independent per token. The latest start at or before the current iteration wins; at identical starts, the last-created rule wins. A policy persists until a later policy replaces it. Schedule an exclusion followed by a restoration to define a temporary dropout. A restore or exclusion also ends an active duty cycle. Future rules can be cancelled; past/current rules remain recorded and can be overridden by a new policy. Editing while viewing history still edits the live run and returns the view there. Reset restores all original targets and clears their rules.

For a duty cycle starting at iteration `s`, period `P`, and requested included percentage `p`, define `K = round(P × p / 100)` and phase `r = (iteration − s) mod P`. The token is included when `r < K`: first `K` iterations on, then `P−K` off. The UI shows both these integer counts and the realized fraction `K/P`. A 25% cycle with period 100 is on for 25 iterations and off for 75. Percentages 0 and 100 are supported. Short/truncated periods may not realize the requested percentage exactly. Scheduling uses iteration numbers, not wall-clock time.

Dropping a token removes it from both inputs and target labels, keeping its vocabulary row in the softmax denominator. Row IDs and labels do not change. In direct-cycle mode, if the ordered included IDs are `C` with count `M`, every batch sequence has the configured context length `T`: `input[b,t] = C[(b+t) mod M]`, `target[b,t] = C[(b+t+1) mod M]`. Dropping 2 from 0,1,2,3 produces the repeating cycle 0→1→3→0, while keeping `B×T` training positions. Absolute positions range from 0 to T−1; RoPE uses the same position indices. Empty membership produces zero positions. Markov mode instead conditions the matrix as described above.

The direct-cycle batch does not rotate phases between updates. Context shorter than the cycle is supported, but a small batch may not observe every included target; the UI reports observed IDs and coverage. Included-cycle membership is distinct from per-batch prediction exposure. Target means use actual labels and their actual counts, which can be unequal. In direct-cycle mode, increasing context to at least the included count guarantees every included target appears; this is not guaranteed for Markov samples. This fixed-window convention differs from older versions, whose context shrank to M on exclusion; new dropout comparisons therefore also change the exposure normalization relative to those historical runs.

Snapshot `t` contains membership and gradients for update `t→t+1`. A policy starting at `t` is applied before evaluating that snapshot. An empty dataset advances the iteration/QAT/duty-cycle clocks but performs no optimizer step, weight decay, moment update, or sphere projection. Its loss/accuracy are unavailable and probe field hidden. The displayed optimizer-update count can therefore differ from the iteration count. Automatic restorations still occur on schedule.

Snapshots retain their own membership masks and policy lists. Dropped numeric tokens become amber untargeted markers; restoration gives them their original target colors. Untargeted probability mass, pair distances and the denominator-only gradient check include dropped numeric rows. Timeline markers indicate policy edits and membership transitions. Loss values on different datasets are not directly comparable, and empty phases appear as gaps in the loss trace. The separate capacity of 100 added untargeted rows does not change when numeric targets are excluded.

## Average hidden-state overlays

Under **View → Average hidden states**, independently enable **Projected on sphere** and **In free space**. Both can be shown together and remain synchronized with the selected timeline frame in the Three.js and compatibility Canvas views. Colors identify the predicted target; filled diamonds labeled `μj` show raw means, hollow diamonds labeled `μjˢ` show projected means. Dashed segments connect the two positions when both views are enabled. Free-space display expands the view to include longer vectors without clamping their coordinates.

Let `h_i` be the actual post-final-RMSNorm hidden vector supplied to the LM head, `y_i` its next-token target, and `n_j` the number of positions with target `j` in the frame. This simulator gives all valid positions equal CE weight. It records `μ_j = sum(h_i for y_i=j) / n_j`. Free space draws `μ_j`; the sphere draws `R μ_j / ||μ_j||`, with `R=sqrt(modelDim)`. **Average first, then project**; averaging individually normalized hidden states would be a different quantity. These points are hidden-state means, not trained vocabulary rows or force minima.

Targets without observed labels in the selected batch have no samples and no marker. A zero mean can appear at the origin in free space, but its undefined sphere direction is omitted. Each immutable snapshot stores its own means and sample counts, including frames created by target restoration or QAT changes. Averages use the same post-projection/pre-next-update model state as that frame's fields. This is a per-frame batch average, not a moving average over training iterations.

## Quantization-aware training

Choose full precision, ternary (`−1,0,1`), signed int3 (`−4…3`), int4 (`−8…7`), int5 (`−16…15`), or symmetric int3 (`−3…3`), int4 (`−7…7`), int5 (`−15…15`). These are scaled integer codebooks, not literal unscaled weights or packed integer execution.

Weight-only fake quantization applies to the shared active token embedding/head, positional embeddings, and attention/MLP matrices. Activations, biases, RMSNorm gains, master weights, and optimizer state stay in full precision. Each tensor uses zero point 0 and its own detached scale `s = max(maxPositive/qmax, minNegative/qmin)`; all-zero tensors use scale 1. Rounding chooses the nearest code, with half steps away from zero, then clips to the codebook. Inactive reserved token rows do not affect calibration.

The forward weight is `W_eff = (1 − α) W + α Q(W)`. The backward pass uses the identity straight-through estimator `dW_eff/dW := 1`, including clipping; gradients do not pass through calibration. Sphere projection applies only to master token rows. Effective rows are not renormalized and may leave the sphere. The row inspector displays both master and forward coordinates, and the diagnostics report embedding RMSE.

- **Linear blend** (default): `α(t) = clamp((t − start)/duration, 0, 1)`, default duration 200 iterations.
- **Cosine blend**: `α = (1 − cos(π × progress))/2` for the same clamped progress.
- **Full QAT at start**: `α = 0` before start and `1` from start onward; duration is ignored.
- **Apply QAT to current run** pauses and applies the entered absolute start, including a past start. It appends a configuration frame at the same optimizer iteration and refreshes cached gradients.
- **Resume into QAT** sets start to the current live iteration, applies the selected precision/schedule, and resumes toward the entered maximum. If that maximum was reached, it extends through the blend plus one full-QAT update (or 200 iterations for immediate QAT), capped at 50,000. A maximum inside the schedule can be raised later with **Set + resume**.
- **Apply + reset timeline** initializes a fresh model using all entered experiment and QAT settings.

The gradient in snapshot `t` drives update `t → t+1` using `α(t)`. At the endpoint of a blend, the snapshot is fully quantized; the following update is the first using 100% QAT. Applying QAT retains weights, optimizer moments and bias-correction step, inserted rows, and old snapshots. Replay uses each frame's saved precision, blend and scale. Runs remain in browser memory and are not persisted across reloads.

## Force conventions

For an active row `w` and ambient cross-entropy gradient `g`, the constrained negative-gradient force is

```text
F_tangent = -(I - w wᵀ / R²) g.
```

The probe heatmap freezes the selected frame. At each point `u` on the sphere it counterfactually inserts one new untargeted output row:

```text
p_u(s) = sigmoid(uᵀ h_s - log Z_s)
g(u)   = mean_s p_u(s) h_s
F(u)   = -(I - u uᵀ / R²) g(u)
```

At full precision this is an exact instantaneous force field, not a future-trajectory prediction. Existing targeted-row gradients additionally include tied input-side effects. Existing untargeted-row gradients equal the denominator-only expression because those rows never occur in the cyclic training batch.

With QAT, tangent arrows are STE surrogate forces on master rows, not derivatives of the discontinuous rounded loss. The probe uses `u_eff = (1 − α)u + αQ(u)` inside the logit, with the saved embedding scale frozen. Its probabilities and insertion potential use that effective logit; force arrows use the identity STE. Actual insertion may recalibrate the shared embedding scale and change existing logits. The unused-gradient check uses effective row logits and compares the analytic surrogate with autodiff.

## Views

1. token rows and trails;
2. actual tied-row CE tangent-force arrows;
3. observed optimizer displacement after reprojection;
4. probe force / potential / probability / radial heatmaps;
5. ambient, tangent, and removed-radial decomposition.

The timeline retains every iteration up to the user-selected limit (maximum `50,000`). The interface estimates the in-memory history size before reset because larger batches and token counts retain more hidden-state data per frame. A compatibility Canvas renderer preserves the full experiment when WebGL is unavailable; browsers with WebGL use the Three.js scene.

## Run locally

```sh
pnpm install
pnpm dev
```

Quality checks:

```sh
pnpm lint
pnpm build
node scripts/verify-qat.mjs
node scripts/verify-architecture.mjs
node scripts/verify-dataset.mjs
```

The UI continuously reports maximum row-norm error, normalized tangency residual, and—after an untargeted row is active—the discrepancy between TensorFlow.js autodiff and the analytic denominator-only gradient.
