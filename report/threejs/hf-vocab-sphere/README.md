# Hugging Face Vocabulary Sphere

A polished FastAPI + Three.js application for studying token-vector geometry from an open-access Hugging Face model. Load a vocabulary matrix, search tokens with regular expressions or ID expressions, build a hand-picked set or an automatic vicinity around an anchor, and project the selected vectors either onto an interactive three-dimensional sphere or into unconstrained regular 3-D space.

The interface deliberately treats every projection as an approximation. It reports angular rank correlation, normalized stress, angular error, local-neighbor recall, and anchor-order preservation so that a visually attractive embedding is not mistaken for an exact picture of the original model space.

For formulas, optimization objectives, failure modes, and a method-selection workflow, see [`PROJECTION_GUIDE.md`](PROJECTION_GUIDE.md).

## Features

- **Model-agnostic vocabulary loading.** Enter a Hugging Face model ID, revision, or local path. Choose the input embedding, output/LM-head matrix, or automatic selection.
- **Low-overhead loading.** The preferred path reads only the relevant tensor from safetensors, including sharded checkpoints. A full `AutoModelForCausalLM` load is a compatibility fallback.
- **Regex, literal, and ID search.** ID mode accepts expressions such as `42,90-110`.
- **Text-to-token selection.** Paste arbitrary text, tokenize it with the active model tokenizer, inspect every occurrence in sequence order, add individual tokens, add all projectable tokens, or replace the current set. Repeated occurrences remain visible while the selected map set is deduplicated by token ID.
- **Four neighborhood/selection workflows.** Add individual search results, replace the set with visible matches, request semantic nearest neighbors, or take a contiguous token-ID window.
- **Interactive Three.js geometry studio.** Orbit, zoom, pan, raycast/hover, pin a token, export JSON, and capture a PNG in either `S²` or regular `R³` mode.
- **Scalable, composable labels.** Node and edge-angle labels use one screen-space canvas rather than one WebGL texture per label. Independent pixel-size sliders are provided, node labels have no fixed display cap, and A/B/C alias letters and token IDs can be hidden independently with checkboxes or the `T` / `I` hotkeys. The edge-label cap defaults to `0 = all` for very dense graphs.
- **Portable settings files.** Save model fields, token selection, arithmetic resultants, projection controls, appearance, and camera position to a versioned JSON file. Loading applies safe bounded values immediately and restores the saved token workspace once the matching model/revision/vector source is active; it never initiates a model download on its own.
- **Angle-aware edge rendering.** Neighbor edges use their exact original-space angle for color, support adjustable pixel thickness, and can display degree labels. Dynamic interleaved buffers are updated in place so animated edge lines and their labels stay synchronized.
- **Magnitude shell.** Keep projected directions near the reference sphere while encoding real relative vector norms with `r = R (||v|| / median ||v||)^α`; `α=0` is a common surface and `α=1` preserves norm ratios.
- **Vector arithmetic and SLERP.** Add labelled expressions such as `(A-B)+C`, `(A+B+C)/3+D`, `mean(A,B,C)+D`, or `slerp(A,B,0.5)`. Results are evaluated safely in the full model dimension and drawn as vectors from the origin.
- **Projection laboratory.** Eleven entries are exposed in the method catalog, including raw-vector PCA/SVD, an adaptive default, and a random baseline.
- **Method benchmark.** Run eligible methods on the same token set and compare their distortion metrics in one table.
- **Blockwise full-vocabulary neighbor search.** Semantic vicinity does not materialize a vocabulary-by-vocabulary matrix.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

Open `http://127.0.0.1:8000`.

UMAP is optional:

```bash
pip install -r requirements-optional.txt
```

The browser imports a pinned Three.js ES module from `esm.sh`, so the initial page load needs internet access unless those modules are vendored locally.

## Settings files

Use **Save settings** to export a versioned JSON workspace. The file records:

- model ID/path, revision, vector source, compute field, and download preference;
- current token IDs and display metadata, anchor, arithmetic expressions, and SLERP editor values;
- projection method and numerical controls;
- point/edge/label appearance, independent alias-letter and token-ID visibility, and the camera pose.

Loading a file validates its schema and clamps all numeric values to the same ranges as the UI. Visual and control settings apply immediately. Token selections are restored only when the active model identity, revision, and concrete vector source match the saved workspace. When they do not match, the model form is populated and the selection is kept pending until that model is loaded manually. This prevents an imported file from unexpectedly downloading or replacing a large model. Projection coordinates are not serialized; after restoration, click **Project selected tokens** (or press **Enter**) to recompute them from the active model vectors.

## Model requirements and loading behavior

The selected repository needs a tokenizer plus a two-dimensional vocabulary matrix. The loader recognizes common names such as `lm_head.weight`, `model.embed_tokens.weight`, `transformer.wte.weight`, `gpt_neox.embed_in.weight`, and related architecture-qualified suffixes.

`MODEL_LOAD_STRATEGY` controls loading:

- `auto` — try a safetensors-only matrix load, then fall back to a full Transformers model.
- `weight_only` or `safetensors` — never instantiate the full model.
- `full_model` — use `AutoModelForCausalLM` directly.

Other useful environment variables are shown in `.env.example`. `TRUST_REMOTE_CODE` is false by default. Hugging Face authentication can be supplied through the normal `HF_TOKEN` environment variable when needed.

A vocabulary matrix can still be large. A 250,000 × 4,096 float16 matrix is roughly 1.9 GiB before auxiliary data. The weight-only path avoids unrelated layers but cannot make the requested matrix itself smaller.

## Input embedding versus LM head

The **input embedding** maps token IDs into the residual stream. The **LM head/output embedding** maps hidden states to vocabulary logits. Some language models tie these matrices, but others do not. For an untied model they answer different geometric questions, so the active source is always displayed in the UI and included in exports.

## Spherical and regular 3-D coordinate modes

The normalized model vectors live on a high-dimensional unit sphere, often with thousands of ambient dimensions. The displayed surface `S²` has only two intrinsic degrees of freedom. No projection can preserve every pairwise angle or every local neighborhood unless the selected data are already effectively two-dimensional. In **Spherical surface** mode the app uses this pipeline:

1. Cast the selected rows to float32 and L2-normalize them.
2. Compute a three-coordinate embedding with the chosen method.
3. Radially normalize each three-coordinate point to the unit sphere.
4. Optionally rotate the result so the anchor is at the north pole. This last rotation changes orientation only, not displayed pairwise angles.

The radial step makes every method comparable on the same visual surface, but it can add distortion to a method that originally produced meaningful radii.

In **Regular 3-D** mode the final pointwise radial normalization is omitted. The server applies at most one optional translation for viewing and one global positive scale, so the projection's relative Euclidean radii are retained. Angular diagnostics still compare directions from the origin, while the visible 3-D radii remain free.

## Projection methods

### Auto / fidelity-aware

Uses direct spherical-stress optimization for at most 140 tokens, tangent-space PCA for an anchored selection up to 2,000 tokens, and spherical PCA otherwise. It is a practical default, not an assertion that one method is universally best.

### Spherical PCA

Runs PCA on L2-normalized vectors after mean, anchor, or no centering. It is fast, deterministic, and useful for broad global variation. PCA maximizes captured variance; it does not directly minimize angular error. Radial normalization discards the projected radius.

### Raw-vector PCA / SVD

Applies a single three-coordinate linear map to the unnormalized vocabulary rows. With **Regular 3-D**, **Centering: None**, and the ordinary projection-coordinate display, this is the arithmetic-friendly mode: linear relations such as `P((A-B)+C) = P(A)-P(B)+P(C)` remain valid up to floating-point error and the app's global scale/rotation. Mean or anchor centering adds a translation, and either spherical normalization or magnitude-shell remapping changes the visible algebra.

### Tangent-space PCA

Chooses the anchor or an extrinsic mean direction, applies the spherical logarithmic map into that point’s tangent space, and runs PCA there. This often works well for a tight semantic neighborhood because small great-circle distances become approximately Euclidean. Points near the base point’s antipode are problematic, and the app emits a warning when the log-map angles become extreme.

### Cosine Gram eigenmap

Builds the selected-token cosine Gram matrix and keeps its three leading positive eigen-directions. This is a direct low-rank approximation to pairwise dot products before the final sphere normalization. It is compelling for small or medium sets, but its eigendecomposition is cubic in token count.

### Classical angular MDS

Computes the full matrix of original angular distances, double-centers the squared distances, and takes the top positive eigenvectors. It targets global angular distances more explicitly than PCA. Classical MDS assumes a Euclidean target; forcing its result onto `S²` adds a second source of stress. It is capped at 800 points.

### Direct spherical stress

Initializes from angular MDS or spherical PCA, then directly optimizes unit vectors in three dimensions so their displayed great-circle angles minimize normalized squared error against the original angles. This is the most sphere-specific method in the app and is often a strong choice for small sets. The objective is non-convex, the seed can matter, and no solution can preserve an arbitrary high-dimensional angle matrix exactly. It is capped at 500 points.

### Cosine Isomap

Builds a cosine-neighbor graph, approximates manifold geodesics with shortest paths, then performs a spectral embedding. It can reveal a curved local manifold that PCA flattens poorly. Results can fail or change dramatically when the graph is disconnected, too sparse, or contains neighborhood shortcuts.

### 3-D cosine t-SNE

Optimizes local similarity probabilities and can separate visually coherent clusters. Its global distances, cluster sizes, and empty spaces should not be read metrically. The objective is non-convex, so the seed matters. It is intentionally excluded from the default benchmark because it is comparatively slow.

### 3-D cosine UMAP

Uses an approximate cosine-neighbor graph and stochastic low-dimensional optimization. It is often a useful nonlinear choice for larger selections, with behavior controlled by `n_neighbors`, `min_dist`, and the seed. It requires the optional `umap-learn` package.

### Gaussian random baseline

Multiplies the normalized vectors by a seeded Gaussian matrix with three output columns. Three dimensions are far too few for strong general distance-preservation guarantees on a large set, which makes this a useful negative control: a sophisticated method should normally beat it on the reported metrics.

## Vector arithmetic and result vectors

The arithmetic editor assigns Excel-style aliases `A`, `B`, …, `Z`, `AA`, … in selected-token order. The safe expression language supports:

- vector addition and subtraction;
- scalar multiplication and division;
- parentheses and unary signs;
- `mean(...)`, `avg(...)`, and `average(...)` for vector averages;
- `slerp(A, B, t)` for `0 ≤ t ≤ 1`.

SLERP follows the shortest geodesic between the two unit directions. Because vocabulary rows can have different lengths, the implementation linearly interpolates their L2 magnitudes and multiplies that magnitude by the interpolated direction. Thus `t=0` and `t=1` reproduce the original vectors exactly; equal-norm inputs reduce to ordinary spherical linear interpolation.

Vector-vector multiplication, arbitrary Python, attribute access, indexing, keyword arguments, and unknown function calls are rejected. Zero-vector results are rejected because they have no direction.

Each result is appended to the high-dimensional projection input, returned with its original magnitude and nearest selected tokens, and drawn as a labelled arrow from the origin. Nonlinear and unit-direction projections still show the exact high-dimensional result, but they do not preserve addition visually; the UI warns when the selected display is not algebraically faithful.

## Edge geometry

Neighbor edges are selected by original-space cosine similarity. Their `angle_deg` value is computed before projection and therefore remains exact even when the displayed points are distorted. Angle coloring maps `0° → 180°` through the same cyan–violet–gold–coral scale used elsewhere. Thick edges use Three.js `LineSegments2`, so the width control is measured in screen pixels and works consistently across WebGL implementations where ordinary line widths are otherwise fixed at one pixel.

The line geometry is allocated only when the rendered edge count changes; animation updates the existing interleaved position/color buffers in place. Edge labels are generated from that same validated edge list, preventing the former state in which an angle annotation could survive while its line was missing. Node and angle annotations are drawn together on a high-DPI overlay canvas, avoiding browser/GPU limits associated with hundreds of individual canvas-texture sprites.

## Diagnostics

- **Angular Spearman ρ:** rank correlation between sampled original and displayed pairwise angles. High values mean relative angular ordering is preserved.
- **Stress-1:** square root of squared angular residuals divided by squared original angles. Lower is better.
- **Angle MAE / p95:** mean and 95th-percentile absolute error in degrees.
- **kNN recall@k:** overlap between original-space and displayed nearest neighbors, averaged over up to 256 probe tokens.
- **Anchor ρ:** rank correlation of all angles from the anchor before and after projection.
- **Runtime:** server-side projection and metric computation time, excluding network transfer and rendering.

Pairwise diagnostics use all pairs for small selections and a deterministic sample of at most 80,000 pairs for larger selections.

## API overview

- `POST /api/model/load`
- `DELETE /api/model`
- `GET /api/status`
- `GET /api/models/local`
- `GET /api/tokens/search`
- `POST /api/tokens/tokenize`
- `GET /api/tokens/{token_id}`
- `GET /api/tokens/neighbors`
- `GET /api/tokens/window`
- `GET /api/projection/methods`
- `POST /api/projection`
- `POST /api/projection/compare`

FastAPI’s interactive schema is available at `/docs`.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

The included tests exercise deterministic core projections, spherical and regular-3-D coordinate modes, linear arithmetic preservation, SLERP and averaging semantics, the safe vector-expression evaluator, text tokenization, edge deduplication, and model-independent API routes without downloading a model.
