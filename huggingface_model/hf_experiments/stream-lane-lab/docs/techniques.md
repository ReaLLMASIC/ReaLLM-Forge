# Per-column geometry and forecasting

Rendered diagrams: [model architecture](architecture.svg) and [linked scalar geometry](geometry.svg). The corresponding `.mmd` files are editable Mermaid sources.

## Scalar values and independent vectors

For column `c`, declare a numeric range `[l_c,u_c]` and `Q_c` discrete levels. Normalize and quantize:

\[
t_c=\operatorname{clip}\!\left(\frac{x_c-l_c}{u_c-l_c},0,1\right),\qquad
q_c=\operatorname{round}((Q_c-1)t_c),\qquad
v_c(q)=l_c+\frac{q}{Q_c-1}(u_c-l_c).
\]

Clipping prevents out-of-range indices; it also loses the size of an out-of-range excursion. Choose meaningful ranges before training and monitor real inputs for clipping.

Independent encoding learns `E_c` with shape `Q_c × D` and returns row `q_c`. A tied head uses `z_c(q)=hᵀE_c[q]`; an untied head has its own table. Unlike SLERP, neighboring quantization levels have no enforced geometric relationship. The table costs `Q_c D` parameters, doubled for an untied input/output pair.

## Spherical input and tied output

Learn `K` raw anchors `A_j` and a positive common radius `r=exp(log r)`. Let `U_j=A_j/||A_j||` and `W_j=rU_j`. For normalized scalar `t`, select the local segment:

\[
s=(K-1)t,\quad j=\min(\lfloor s\rfloor,K-2),\quad \tau=s-j.
\]

For neighboring directions, the actual geometric angle is mathematically `θ_j=acos(U_jᵀU_{j+1})`. The implementation computes it using `atan2` of the orthogonal component norm and dot product, then clamps it to `[10⁻³,π−10⁻³]`. The interpolated vector is

\[
e(t)=a(\tau,\theta_j)W_j+b(\tau,\theta_j)W_{j+1},\qquad
a=\frac{\sin((1-\tau)\theta_j)}{\sin\theta_j},\quad
b=\frac{\sin(\tau\theta_j)}{\sin\theta_j}.
\]

For an ordinary nondegenerate arc inside the angle bounds, this stays on the radius-`r` sphere. Clamping near coincident or antipodal anchors slightly relaxes that statement, while preserving endpoints. Antipodal anchors do not define a unique great-circle arc. Numerical safeguards should not be interpreted as additional expressivity. The radius initializes to `sqrt(D)`.

At `t=1`, the clamped segment index gives `j=K-2` and `τ=1`, so the result is the last knot without an out-of-bounds gather. Adjacent segments use the same internal knot: the curve is position-continuous (`C⁰`), but its tangent may have a corner. Two, three, five, and nine knots give one, two, four, and eight arcs.

The input uses its scalar directly, so SLERP can encode intermediate values from mean decoding. A tied categorical head evaluates the same curve at `t_q=q/(Q-1)`. Tying means shared parameters, not a copied initialization.

## Untied relaxed output curve

The untied SLERP head learns independent unconstrained knots `V_j` and angles

\[
\phi_j=\epsilon+(\pi-2\epsilon)\sigma(\rho_j).
\]

It uses the same coefficient formula with `φ_j` instead of the geometric angle between its output knots:

\[
e_{\rm out}(t)=a(\tau,\phi_j)V_j+b(\tau,\phi_j)V_{j+1}.
\]

The output knots are not normalized or projected after optimizer steps, and `φ_j` need not equal the angle between `V_j` and `V_{j+1}`. This is a relaxed interpolation family; it need not trace a constant-radius sphere. Input encoding still uses the spherical geometry above. Thus an observed tied/untied difference combines sharing and output-family changes.

At construction, copy the input's scaled knots and initialize `φ_j` from their geometric angles, allocating separate parameter storage. This matches ordinary nondegenerate tied curves up to precision and angle bounds. When converting an existing tied checkpoint, load the trained input first, then initialize the independent output. Merely changing a JSON flag is not checkpoint migration. Loading an already-untied checkpoint must preserve its saved output parameters.

Because `sin φ` becomes small near `π`, bounded angles prevent division by zero but do not prevent amplification. At a segment midpoint,

\[
e_{\rm out}(1/2)=\frac{V_j+V_{j+1}}{2\cos(\phi_j/2)}.
\]

Monitor knot norms, angle extremes, logits, and rollout saturation. Extra knots expand local flexibility but do not guarantee stable autoregression.

## Factorized logits and gradients

Project `h` onto the `K` knots once, then blend those scores for every level:

\[
p_j=h^TV_j,\qquad z(q)=a_qp_{j(q)}+b_qp_{j(q)+1}.
\]

This replaces a dense `Q × D` output table multiplication with `K` dot products plus `Q` blends. The sampled output table has rank at most `min(K,D,Q)`; this rank bound does not bound the complete nonlinear transformer's expressivity. Biases, if introduced later, need separate accounting.

For a two-knot segment, with `δ_q=∂L/∂z(q)`:

\[
\frac{\partial L}{\partial V_0}=\left(\sum_q\delta_q a_q\right)h,\quad
\frac{\partial L}{\partial V_1}=\left(\sum_q\delta_q b_q\right)h,\quad
\frac{\partial L}{\partial h}=\sum_q\delta_qe_{\rm out}(t_q).
\]

Angles receive gradients through the coefficients. Untying does not freeze input embeddings: they still receive gradients through the transformer. Coefficients must remain in the training graph; detached or permanently cached training coefficients would silently remove angle learning.

| Per-column parameters | Input | Additional untied output |
| --- | ---: | ---: |
| Independent | `QD` | `QD` |
| Linked SLERP | `KD + 1` | `KD + K − 1` |

Tied heads add no separate output geometry. The radius adds one parameter to each spherical input curve. At `D=48,Q=128,K=5`, the input geometry uses `241` parameters, compared with `6,144` for the independent table; the untied relaxed head adds `244`.

## Loss, forecasting, and what is measured

Each time step predicts all columns from preceding frames. For aligned next-frame targets, the default categorical loss is

\[
L_{\rm CE}=\frac1C\sum_c\frac1{BT}\sum_{b,t}-\log p_c(q_{b,t,c}\mid x_{b,\le t,:}).
\]

Equal column averaging prevents a high-cardinality column from receiving extra weight merely because it has more levels. Optional numeric supervision adds `λ MSE(μ,target)`, where `μ_c=Σ_q p_c(q)v_c(q)`. The model API normalizes this error by each column's half-range. The sine training CLI instead uses physical-unit MSE explicitly; its common `[-1,1]` ranges make those scales identical. Record `λ`: it changes the objective. Input/target alignment is the caller's responsibility; there is no hidden second shift.

The optional `--target-mode linear` replaces nearest-bin targets with adjacent-bin masses. For `s=(Q−1)t`, let `q₀=floor(s)`, `q₁=min(q₀+1,Q−1)`, and `f=s−q₀`. Train against

\[
L_{\rm linear}=-(1-f)\log p(q_0)-f\log p(q_1).
\]

Its target mean `(1−f)v(q₀)+fv(q₁)` equals the original scalar within its declared range, including the endpoints. Out-of-range values still clip. Only adjacent bins receive target mass; this is not smoothing across every category. The default remains `hard`, and reported evaluation `teacher_nll` always uses hard nearest-bin CE so its definition stays consistent across these training objectives.

CLI defaults add input perturbations and a four-step mean-feedback MSE objective after warm-up. `--rollout-gradient detached` cuts the feedback graph at each step, while each predicted output retains its supervised gradient. This teaches local recovery. `--rollout-gradient full` retains feedback gradients through continuous SLERP inputs and can optimize earlier predictions for their downstream effects. Independent lanes still quantize by rounding and integer lookup, which blocks gradients through their generated scalar values; `full` does not make that quantizer differentiable. In mixed configurations, this distinction applies per lane. Checkpoint selection uses validation rollout RMSE, and final reporting uses different test seeds. Set MSE, rollout, and noise weights to zero for the pure categorical objective.

Fresh experiments record the `matched-v1` initialization protocol. The geometry seed, backbone seed, minibatch seed, and noise seed are isolated. With the same channel count, backbone dimensions, training seed, and batch/noise settings, changing geometry preserves shared backbone initialization and example/noise order. Different input and head parameterizations are intentional. Earlier runs without this marker are exploratory and must not be described as matched. Training flags and exact RNG seeds are saved in `run.json`; comparisons should retain that file alongside checkpoint metrics.

Mean decoding returns `μ`; greedy chooses `argmax p`; sampling draws per-column categorical values. The shared backbone couples their conditioning, but the factorized head does not enforce a joint constraint among sampled columns. Forecasting appends predicted values and repeats, cropping to the most recent configured context. Learned positions are reapplied to that window; this implementation deliberately recomputes the window instead of reusing inconsistent cached positions.

Report separately: (1) teacher-forced NLL/MAE with real history, (2) multi-step rollout MAE/RMSE, (3) horizon-wise errors, (4) persistence scores, and (5) latency. Select checkpoints on validation data and evaluate the selected model on held-out seeds. Longer runs or extra knots are worthwhile only if these measurements improve. A bounded output range prevents numerical value explosions but can still produce persistent saturation or a wrong oscillation.

## Execution boundaries

The Torch implementation defines the reference mathematics. Triton accelerates supported indexed gathers and score blending, including backward propagation; normalization, coefficients, matrix products, and attention remain ordinary PyTorch operations. This keeps learned angles in autograd and avoids constructing the full dense SLERP table during projection. The output geometry disables autocast for its FP32 path; it does not override the caller's global CUDA TF32 setting. Backend parity must check gradients as well as forward values and include endpoint/join cases.

CPU benchmarks use Torch and require no Triton package. Explicit `backend="triton"` refuses unsupported execution; `auto` is the portable choice. CUDA parity and speed measurements must be run on the intended GPU. Training throughput, single-stream latency, and long-horizon latency are distinct quantities and should not be substituted for each other.
