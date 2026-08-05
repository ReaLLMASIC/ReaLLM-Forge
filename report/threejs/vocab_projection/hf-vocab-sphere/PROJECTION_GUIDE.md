# Projection Guide: From Vocabulary Vectors to a 3-D Sphere

## 1. The object being visualized

For a selected token set, let the active vocabulary matrix contribute row vectors

\[
w_i \in \mathbb{R}^{d}, \qquad i=1,\ldots,n.
\]

The app first removes magnitude from the geometric comparison:

\[
x_i = \frac{w_i}{\lVert w_i\rVert_2}.
\]

Original-space cosine similarity and signed angular distance are therefore

\[
s_{ij}=x_i^\top x_j, \qquad
\theta_{ij}=\arccos(\operatorname{clip}(s_{ij},-1,1)).
\]

The browser displays unit vectors

\[
y_i\in S^2\subset\mathbb{R}^3,
\]

whose displayed great-circle distances are

\[
\widehat{\theta}_{ij}=\arccos(y_i^\top y_j).
\]

A general high-dimensional angle matrix cannot be represented exactly on `S²`. The practical question is which relationships should receive priority when distortion is unavoidable.

## 2. The common sphere-mapping stage

Every method first produces ordinary three-coordinate values `zᵢ ∈ R³`. The app then applies

\[
y_i = \frac{z_i}{\lVert z_i\rVert_2}.
\]

This gives every method the same visual support and makes displayed angles directly comparable. It also discards projected radius. For PCA, t-SNE, UMAP, Isomap, and classical MDS, that radius may contain information; the sphere view intentionally chooses directional readability over retaining it.

When an anchor is selected, a rigid 3-D rotation moves its displayed point to `(0,1,0)`. A second deterministic rotation fixes the remaining spin around that axis. These rotations do not change any displayed pairwise angle.

## 3. Method-by-method analysis

### Spherical PCA

The input is the matrix of normalized vectors. Depending on the UI setting, it is mean-centered, anchor-centered, or left uncentered. PCA finds three orthogonal directions maximizing projected variance.

**Optimizes:** Euclidean reconstruction variance before radial normalization.

**Strengths:** deterministic, fast, scalable, useful for global axes, easy to compare across nearby selections.

**Failure mode:** variance is not angular stress. A dominant anisotropic direction can consume a component, while the final radial normalization can move low-radius points substantially.

**Use it for:** hundreds to thousands of tokens, broad vocabulary slices, and a stable first view.

### Tangent-space PCA

Choose a unit base direction `μ`, either the anchor or the normalized extrinsic mean. The spherical logarithmic map is

\[
\log_\mu(x_i)=
\frac{\theta_i}{\sin\theta_i}
\left(x_i-(\mu^\top x_i)\mu\right),
\qquad
\theta_i=\arccos(\mu^\top x_i).
\]

PCA is then applied to these tangent vectors.

**Optimizes:** variance in a local linearization of the original high-dimensional sphere.

**Strengths:** respects local spherical geometry better than subtracting vectors directly; particularly natural for an anchor neighborhood.

**Failure mode:** the logarithmic map is ill-conditioned near the antipode `θ≈π`, and one tangent chart cannot faithfully cover a widely dispersed set.

**Use it for:** semantic nearest-neighbor sets around a target token.

### Cosine Gram eigenmap

Construct the selected-token Gram matrix

\[
K_{ij}=x_i^\top x_j.
\]

Its leading positive eigensystem gives a rank-three approximation

\[
K \approx U_3\Lambda_3U_3^\top,
\qquad
z_i=(U_3\Lambda_3^{1/2})_i.
\]

**Optimizes:** the best low-rank approximation to the cosine Gram matrix in Frobenius norm before sphere normalization.

**Strengths:** directly tied to pairwise dot products; deterministic; informative spectrum diagnostics.

**Failure mode:** cubic eigendecomposition in token count and unavoidable loss when the Gram matrix has substantial rank beyond three.

**Use it for:** carefully chosen sets up to roughly one thousand tokens.

### Classical angular MDS

Build the full angular-distance matrix `D=(θᵢⱼ)`, square it, and double-center:

\[
B=-\tfrac12 J D^{\circ 2}J,
\qquad
J=I-\tfrac1n\mathbf{1}\mathbf{1}^\top.
\]

The leading positive eigenvectors of `B` form a classical multidimensional-scaling embedding.

**Optimizes:** a Euclidean representation of the angular distance matrix.

**Strengths:** targets global angular distance more explicitly than PCA.

**Failure mode:** arbitrary angular distance matrices are not generally Euclidean in three dimensions. Negative eigenvalues quantify that incompatibility, and radial normalization adds another distortion stage.

**Use it for:** small sets where global pairwise relations matter more than runtime.

### Direct spherical stress

Initialize on `S²` with angular MDS or PCA, then optimize the displayed unit vectors directly:

\[
\min_{y_1,\ldots,y_n\in S^2}
\frac{\sum_{i<j}(\widehat\theta_{ij}-\theta_{ij})^2}
{\sum_{i<j}\theta_{ij}^2}.
\]

The implementation uses Adam, re-normalizes every point after each step, reduces the learning rate during the run, and retains the best iterate.

**Optimizes:** exactly the normalized squared angular stress reported by the app, subject to the `S²` constraint.

**Strengths:** the objective matches the displayed geometry; often the strongest small-set angular fit.

**Failure mode:** non-convex, iterative, seed-dependent, and quadratic per iteration. A better stress score does not guarantee better semantic cluster readability.

**Use it for:** small presentation sets, hand-curated comparisons, and validating whether a linear projection is leaving substantial angular fidelity on the table.

### Cosine Isomap

Build a nearest-neighbor graph under cosine distance, compute graph shortest paths, and spectrally embed those geodesic estimates.

**Optimizes:** preservation of graph-geodesic distances rather than direct ambient angles.

**Strengths:** can unfold curved local structure that linear methods compress.

**Failure mode:** too few neighbors disconnect the graph; too many create shortcuts. The result can change abruptly with the neighborhood parameter.

**Use it for:** medium-sized selections with a plausible connected manifold.

### 3-D cosine t-SNE

Convert high-dimensional local similarities into probability distributions and minimize KL divergence against low-dimensional similarities.

**Optimizes:** local probability-neighborhood agreement, with asymmetric penalties that strongly discourage missing close neighbors.

**Strengths:** frequently produces visually separated local groups.

**Failure mode:** global distances, cluster area, empty space, and orientation are not metrically trustworthy. The result is stochastic and can vary across seeds.

**Use it for:** exploratory cluster discovery, followed by verification in original-space angles.

### 3-D cosine UMAP

Construct a fuzzy cosine-neighbor graph and optimize a low-dimensional fuzzy-set cross-entropy.

**Optimizes:** a balance of local connectivity and configurable global organization.

**Strengths:** nonlinear, generally faster than t-SNE on larger selections, and tunable through `n_neighbors` and `min_dist`.

**Failure mode:** the parameters define the visual story. Small `n_neighbors` emphasizes local microstructure; large values smooth toward global structure. It is stochastic.

**Use it for:** hundreds to several thousand tokens when local structure is central.

### Gaussian random baseline

Draw `R∈R^{d×3}` with entries from `N(0,1/3)` and set

\[
z_i=x_i^\top R.
\]

**Optimizes:** nothing about the observed token set.

**Strengths:** fast, reproducible with a seed, and valuable as a control.

**Failure mode:** three coordinates are much too few for strong general distance-preservation guarantees on a large set.

**Use it for:** checking whether a sophisticated projection actually improves the measured geometry.

## 4. Reading the diagnostics

### Angular Spearman correlation

The app correlates ranks of sampled `θᵢⱼ` and `θ̂ᵢⱼ`. This ignores absolute calibration and asks whether nearer pairs remain nearer. A high value can coexist with sizeable degree error.

### Stress-1

\[
\operatorname{stress}_1 =
\sqrt{
\frac{\sum(\widehat\theta_{ij}-\theta_{ij})^2}
{\sum\theta_{ij}^2}
}.
\]

It measures absolute angular fit relative to the energy in the original distances. Lower is better.

### Mean and 95th-percentile angular error

These reveal calibration in degrees. The p95 statistic is useful when a method looks good on average but badly misplaces a minority of tokens.

### k-nearest-neighbor recall

For each probe token, the app compares its original top-`k` cosine neighbors with its top-`k` displayed spherical neighbors. It emphasizes local topology and can favor t-SNE or UMAP even when global stress is weaker.

### Anchor-angle rank correlation

This compares the original and displayed ordering of tokens by angle from the anchor. It is especially useful when the visualization is intended to communicate rings or shells around one target.

## 5. Recommended workflow

1. **Start with the semantic vicinity selector.** This prevents a rare target from being visually overwhelmed by unrelated vocabulary regions.
2. **Use Auto, then benchmark.** Auto supplies a reasonable first view; the benchmark reveals whether another method materially improves the metrics that matter for the task.
3. **Match the metric to the question.** Choose angular stress for quantitative pairwise comparison, kNN recall for cluster/neighborhood exploration, or anchor correlation for radial stories.
4. **Inspect outliers.** Pin suspicious points and compare their original angle, norm, and nearest-neighbor edges.
5. **Change the seed for stochastic methods.** A cluster that disappears across seeds is weak evidence.
6. **Compare input and output matrices.** Untied models can encode materially different token relationships in the two spaces.
7. **Export the JSON.** The export contains token metadata, coordinates, original-space edges, method details, warnings, and diagnostics for reproducible analysis.

## 6. Claims the sphere does not support

- Screen distance is not the original model distance.
- A gap on the sphere does not prove a semantic boundary.
- Cluster area is not token frequency or probability.
- The orientation of PCA, MDS, t-SNE, or UMAP axes has no inherent semantic meaning.
- A token vector is not a context-dependent hidden state; it is one row of an embedding or output matrix.
- Similar LM-head rows indicate similar output directions, not necessarily interchangeability in generated text.
- Token-ID proximity is tokenizer construction metadata, not semantic proximity; that is why the app separates ID windows from cosine vicinity.
