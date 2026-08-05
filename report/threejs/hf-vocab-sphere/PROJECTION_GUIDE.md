# Projection Guide: Vocabulary Vectors on S² and in Regular 3-D

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

In spherical mode the browser displays unit vectors

\[
y_i\in S^2\subset\mathbb{R}^3,
\]

whose displayed great-circle distances are

\[
\widehat{\theta}_{ij}=\arccos(y_i^\top y_j).
\]

A general high-dimensional angle matrix cannot be represented exactly on `S²`. Regular 3-D mode instead displays unconstrained coordinates `zᵢ∈R³`; it retains projected radius but still cannot preserve a general high-dimensional geometry exactly. The practical question is which relationships should receive priority when distortion is unavoidable.

## 2. Coordinate modes and radial display

Every method first produces ordinary three-coordinate values `zᵢ ∈ R³`.

### Spherical surface

The app applies

\[
y_i = \frac{z_i}{\lVert z_i\rVert_2}.
\]

This gives every method the same visual support and makes displayed angles directly comparable. It also discards projected radius. For PCA, t-SNE, UMAP, Isomap, and classical MDS, that radius may contain information; the sphere view intentionally chooses directional readability over retaining it.

### Regular 3-D

The pointwise normalization is omitted. For a stable camera, the implementation may subtract one global viewing center and always divides by one global positive scale. Those operations preserve pairwise Euclidean differences up to a common scale. For the raw-vector method with centering set to `None`, the viewing translation is also omitted so the origin and vector addition remain meaningful.

Angular diagnostics in regular 3-D use the directions `zᵢ/||zᵢ||`; visible radii are not silently folded into the angular score.

### Magnitude shell

The browser can override either coordinate mode's visible radius while keeping its projected direction:

\[
r_i = R\left(\frac{\lVert w_i\rVert_2}{\operatorname{median}_j\lVert w_j\rVert_2}\right)^\alpha.
\]

`α=0` places every point on the same reference sphere. `α=1` preserves true relative vector-norm ratios after one global median normalization. Intermediate values compress log-radius differences and keep the points near the surface. This is a display transform, so it intentionally changes projected Euclidean distances and should not be used when exact visible vector arithmetic is the goal.

When an anchor is selected, a rigid 3-D rotation moves its displayed point to `(0,1,0)`. A second deterministic rotation fixes the remaining spin around that axis. These rotations do not change any displayed pairwise angle.

### Screen-space annotations and edges

Node names and edge-angle values are annotations, not projected geometry. They are drawn in a high-DPI two-dimensional overlay after the Three.js scene renders. This avoids one texture allocation per label and allows all requested labels to be shown; it also means labels deliberately remain readable even when their associated point is visually behind another point. Separate node-label and edge-label size controls affect only typography.

Edges and edge labels share one validated list of original-space neighbor pairs. Each edge color is a function of its original signed angle, while its endpoints follow the current animated display coordinates. Changing line width, label size, or sphere opacity therefore does not change any projection or fidelity statistic.

## 3. Method-by-method analysis

### Spherical PCA

The input is the matrix of normalized vectors. Depending on the UI setting, it is mean-centered, anchor-centered, or left uncentered. PCA finds three orthogonal directions maximizing projected variance.

**Optimizes:** Euclidean reconstruction variance before radial normalization.

**Strengths:** deterministic, fast, scalable, useful for global axes, easy to compare across nearby selections.

**Failure mode:** variance is not angular stress. A dominant anisotropic direction can consume a component, while the final radial normalization can move low-radius points substantially.

**Use it for:** hundreds to thousands of tokens, broad vocabulary slices, and a stable first view.

### Raw-vector PCA / SVD

Let `W` be the matrix of unnormalized selected vectors. After optional mean or anchor subtraction, a rank-three truncated SVD supplies a single linear map `P:Rᵈ→R³`.

With no centering,

\[
P((A-B)+C)=P(A)-P(B)+P(C)
\]

exactly up to numerical precision. The app's deterministic axis signs, rigid anchor rotation, and global viewing scale are also linear, so they preserve this identity. Pointwise sphere normalization and magnitude-shell remapping are nonlinear and therefore do not.

**Optimizes:** captured variance/energy of the raw vocabulary rows under a rank-three linear map.

**Strengths:** retains magnitude information; gives the most interpretable view for analogies, averages, and result vectors; scales to large selections.

**Failure mode:** large-norm rows can dominate the basis; three components may still omit the dimensions carrying a particular analogy; mean or anchor centering changes absolute vector addition by a translation.

**Use it for:** regular 3-D vector arithmetic, norm-sensitive studies, and comparing raw input/output vocabulary matrices.

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

The reported edge angle is not a displayed measurement. It is computed directly from the original high-dimensional rows before projection:

\[
\theta_{ij}=\arccos\frac{w_i^\top w_j}{\lVert w_i\rVert\lVert w_j\rVert}.
\]

Consequently, edge colors and optional edge labels remain exact even when the endpoint placement is distorted. Thick edges are rendered with screen-space `LineSegments2`; changing width has no effect on the underlying graph.

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

For each probe token, the app compares its original top-`k` cosine neighbors with its top-`k` displayed directional neighbors. It emphasizes local topology and can favor t-SNE or UMAP even when global stress is weaker.

### Anchor-angle rank correlation

This compares the original and displayed ordering of tokens by angle from the anchor. It is especially useful when the visualization is intended to communicate rings or shells around one target.

## 5. Vector arithmetic and SLERP

The expression evaluator operates on full `d`-dimensional vectors, not on the displayed 3-D coordinates. It accepts selected-token aliases (`A`, `B`, …, `AA`, …), parentheses, `+`, `-`, scalar `*` and `/`, unary signs, vector averaging through `mean(...)`/`avg(...)`, and spherical interpolation through `slerp(A,B,t)`. Arbitrary Python, indexing, attributes, vector-vector multiplication, keyword arguments, and unknown function calls are rejected.

For two non-zero vectors `a` and `b`, write

\[
u=\frac{a}{\lVert a\rVert},\qquad
v=\frac{b}{\lVert b\rVert},\qquad
\omega=\arccos(\operatorname{clip}(u^\top v,-1,1)).
\]

For ordinary non-degenerate directions, directional SLERP is

\[
d(t)=\frac{\sin((1-t)\omega)}{\sin\omega}u
     +\frac{\sin(t\omega)}{\sin\omega}v,
\qquad 0\le t\le1.
\]

Vocabulary rows are not generally equal-norm, so the app also interpolates magnitude,

\[
m(t)=(1-t)\lVert a\rVert+t\lVert b\rVert,
\qquad q(t)=m(t)\,\frac{d(t)}{\lVert d(t)\rVert}.
\]

Near-parallel vectors use normalized linear interpolation for numerical stability. Near-antipodal vectors have no unique shortest geodesic; the implementation chooses a deterministic orthogonal great-circle direction so the result remains stable and reproducible. Zero vectors are rejected.

For every arithmetic or SLERP result `q`, the server computes `||q||`, its angle to the anchor, and its nearest selected tokens in the original space. It then appends `q` to the projection input so nonlinear methods can place it jointly with the selected tokens. The map draws a labelled arrow from the origin to the result coordinate.

There are two distinct correctness questions:

1. **Was the high-dimensional operation evaluated correctly?** Yes; this is independent of projection choice.
2. **Does the 3-D drawing preserve the algebra or geodesic visually?** Not in general. A linear, origin-preserving display is best for addition; use Raw-vector PCA / SVD, Regular 3-D, Centering `None`, and Projection coordinates. A spherical display is visually natural for SLERP directions, but the projection itself can still distort the original high-dimensional geodesic.

## 6. Recommended workflow

1. **Start with the semantic vicinity selector.** This prevents a rare target from being visually overwhelmed by unrelated vocabulary regions.
2. **Choose the coordinate question.** Use `S²` for pure directional comparison; use regular `R³` when projected radius or arithmetic matters.
3. **Use Auto, then benchmark.** Auto supplies a reasonable first view; the benchmark reveals whether another method materially improves the metrics that matter for the task.
4. **Match the metric to the question.** Choose angular stress for quantitative pairwise comparison, kNN recall for cluster/neighborhood exploration, or anchor correlation for radial stories.
5. **Inspect outliers.** Pin suspicious points and compare their original angle, norm, and nearest-neighbor edges.
6. **For arithmetic, use the dedicated setup button.** Then verify the result's nearest original-space tokens rather than trusting screen proximity alone.
7. **Change the seed for stochastic methods.** A cluster that disappears across seeds is weak evidence.
8. **Compare input and output matrices.** Untied models can encode materially different token relationships in the two spaces.
9. **Export the JSON.** The export contains token metadata, coordinates, arithmetic results, original-space edges, method details, warnings, and diagnostics for reproducible analysis.

## 7. Claims the visualization does not support

- Screen distance is not the original model distance.
- A gap on the sphere does not prove a semantic boundary.
- Cluster area is not token frequency or probability.
- The orientation of PCA, MDS, t-SNE, or UMAP axes has no inherent semantic meaning.
- A token vector is not a context-dependent hidden state; it is one row of an embedding or output matrix.
- Similar LM-head rows indicate similar output directions, not necessarily interchangeability in generated text.
- Token-ID proximity is tokenizer construction metadata, not semantic proximity; that is why the app separates ID windows from cosine vicinity.
