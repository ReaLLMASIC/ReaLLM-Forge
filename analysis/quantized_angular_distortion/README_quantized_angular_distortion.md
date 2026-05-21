# Quantized Angular Distortion

This project studies how low-bit quantization changes the angle between two random normalized vectors.  It overlays a high-dimensional Gaussian transfer prediction with finite-dimensional Monte Carlo simulations for both signed integer quantizers and custom fake-floating-point formats.

The main script assumed by this README is:

```bash
gaussian_transfer_distortion_debugged_compare.py
```

It retains the original integer theory, integer Monte Carlo, dimension sweep, arcsine-law, CSV, and floating-point plotting features, while adding debugged fake-FP behavior, one-PDF-per-FP-width output, and same-width INT-versus-FP comparison plots.

## Quick start

Install the Python dependencies:

```bash
python -m pip install numpy scipy matplotlib
```

Run the built-in checks:

```bash
python gaussian_transfer_distortion_debugged_compare.py --self-test
```

Print the fake-FP formats that will be used:

```bash
python gaussian_transfer_distortion_debugged_compare.py --list-fp-formats
```

Generate the default integer plots:

```bash
python gaussian_transfer_distortion_debugged_compare.py \
  --dim 4096 \
  --trials 300 \
  --bits 3 4 5 \
  --angles-start 0 \
  --angles-stop 90 \
  --angles-step 1 \
  --scale-mode fixed \
  --clip-sigma 3.0 \
  --output outputs/gaussian_transfer_distortion.pdf \
  --csv outputs/gaussian_transfer_distortion.csv
```

Generate one fake-FP PDF/CSV per total FP width:

```bash
python gaussian_transfer_distortion_debugged_compare.py \
  --fp-only \
  --fp-split-by-total-bits \
  --dim 4096 \
  --trials 300 \
  --angles-start 0 \
  --angles-stop 90 \
  --angles-step 1 \
  --fp-theory-samples 500000 \
  --fp-formats e5m2 e4m3 e3m4 e2m3 e2m2 e2m1 \
  --output outputs/base.pdf \
  --fp-output fp_format_angular_distortion.pdf \
  --fp-csv fp_format_angular_distortion.csv
```

This writes files such as:

```text
outputs/fp_format_angular_distortion_fp8.pdf
outputs/fp_format_angular_distortion_fp6.pdf
outputs/fp_format_angular_distortion_fp5.pdf
outputs/fp_format_angular_distortion_fp4.pdf
```

Generate same-width INT-versus-FP overlays:

```bash
python gaussian_transfer_distortion_debugged_compare.py \
  --compare-only \
  --dim 4096 \
  --bits 4 5 6 8 \
  --fp-preset debugged-ieee \
  --angles-start 0 \
  --angles-stop 90 \
  --angles-step 1 \
  --trials 300 \
  --quad-nodes 96 \
  --fp-theory-samples 500000 \
  --output outputs/base.pdf \
  --compare-output int_vs_fp_angular_distortion.pdf \
  --compare-csv int_vs_fp_angular_distortion.csv
```

This writes files such as:

```text
outputs/int_vs_fp_angular_distortion_bit4.pdf
outputs/int_vs_fp_angular_distortion_bit5.pdf
outputs/int_vs_fp_angular_distortion_bit6.pdf
outputs/int_vs_fp_angular_distortion_bit8.pdf
```

For ML-style FP formats, use:

```bash
python gaussian_transfer_distortion_debugged_compare.py \
  --compare-only \
  --dim 4096 \
  --bits 4 8 \
  --fp-preset ml \
  --angles-start 0 \
  --angles-stop 90 \
  --angles-step 1 \
  --trials 300 \
  --fp-theory-samples 500000 \
  --output outputs/base.pdf \
  --compare-output int_vs_fp_ml.pdf \
  --compare-csv int_vs_fp_ml.csv
```

## What the plots mean

The x-axis is the original angle $\theta$ between two unquantized unit vectors.  The y-axis is the angular distortion after quantization:

$$
\widehat\theta - \theta.
$$

Positive values mean the quantized vectors are farther apart than the original vectors.  Negative values mean the quantized vectors are closer together.

Plot conventions:

- **Solid INT curves** are deterministic Gaussian-transfer predictions computed by quadrature.
- **Solid FP curves** are Monte Carlo estimates of the fake-FP Gaussian transfer curve, not exact closed-form curves.  They connect independently estimated angle samples, so small wiggles can be ordinary sampling noise.
- **Markers with error bars** are finite-dimensional Monte Carlo means plus or minus one empirical standard deviation over random vector-pair trials.
- The gray band marks the $20^\circ$--$30^\circ$ regime.

To reduce visible wobble in FP solid curves, increase:

```bash
--fp-theory-samples 5000000
```

To reduce noise in the finite-vector marker means, increase:

```bash
--trials 1000
```

## Theory

### High-dimensional coordinate model

Let $x,y\in\mathbb{R}^d$ be random unit vectors with fixed angle $\theta$.  Write

$$
\rho = \cos\theta.
$$

In high dimension, the scaled coordinate pair

$$
(\sqrt d\,x_i,\sqrt d\,y_i)
$$

is well approximated by a bivariate standard normal pair

$$
(X,Y)\sim\mathcal{N}\left(0,
\begin{bmatrix}
1 & \rho \\
\rho & 1
\end{bmatrix}\right).
$$

For a scalar quantizer $T$, the high-dimensional quantized cosine is modeled by the transfer function

$$
C_T(\rho) =
\frac{\mathbb{E}[T(X)T(Y)]}
{\sqrt{\mathbb{E}[T(X)^2]\,\mathbb{E}[T(Y)^2]}}.
$$

For identical symmetric quantization of both coordinates, the denominator is usually just $\mathbb{E}[T(X)^2]$.  The predicted quantized angle and distortion are

$$
\theta_T = \arccos(C_T(\cos\theta)),
$$

and

$$
\Delta_T(\theta) = \theta_T - \theta.
$$

The plots report $\Delta_T(\theta)$ in degrees.

### Signed integer quantizer

For an integer bit-width `bits`, the script uses a signed symmetric mid-tread codebook

$$
\{-q_{\max},\ldots,-1,0,1,\ldots,q_{\max}\},
$$

where

$$
q_{\max}=2^{\text{bits}-1}-1.
$$

The coordinate code is

$$
Q_b(z)=\operatorname{clip}\left(\operatorname{round}\left(\frac{z}{\delta}\right),-q_{\max},q_{\max}\right).
$$

The script computes cosine similarity using the integer code vectors.  A common dequantization scale cancels in the cosine, so the theory is written directly in terms of codes.

The Gaussian-theory step size is

$$
\delta = \frac{\tau}{q_{\max}}.
$$

The effective clipping threshold $\tau$ depends on `--scale-mode`:

| scale mode | empirical scale | Gaussian-theory threshold |
|---|---|---|
| `fixed` | full scale is `clip_sigma / sqrt(d)` | $\tau=\texttt{clip\_sigma}$ |
| `std` | full scale is `clip_sigma * std(v)` | $\tau=\texttt{clip\_sigma}$ |
| `maxabs` | full scale is `max(abs(v))` | $\tau\approx\Phi^{-1}(1-1/(2d))$ |

The `maxabs` theory is an effective-threshold approximation.  The empirical path still uses the actual maximum absolute coordinate for each sampled vector.

### Integer Gaussian transfer by quadrature

The integer transfer curve is computed by deterministic quadrature, not by random sampling.

For each integer bin $k$, the scalar line is partitioned into intervals

$$
[k-1/2,k+1/2]\delta,
$$

with clipped tails for the extreme codes.  The code integrates over $X$, then uses the conditional law

$$
Y\mid X=x \sim \mathcal{N}(\rho x, 1-\rho^2).
$$

It computes the mixed code moments

$$
M_{pq}=\mathbb{E}[Q_b(X)^p Q_b(Y)^q].
$$

The predicted quantized correlation is

$$
\rho_q = \frac{M_{11}}{M_{20}},
$$

and the predicted distortion is

$$
\arccos(\rho_q)-\theta.
$$

### Finite-dimensional spread by the delta method

For integer quantization, the script also estimates the finite-dimensional standard deviation of the quantized angle using a delta-method approximation.

Define per-coordinate random variables

$$
A=Q_b(X)Q_b(Y),\qquad B=Q_b(X)^2,\qquad C=Q_b(Y)^2.
$$

The sample cosine is approximately

$$
\widehat\rho_q = \frac{\overline A}{\sqrt{\overline B\,\overline C}}.
$$

At symmetry, $\mathbb{E}[B]=\mathbb{E}[C]=M_{20}$ and $\mathbb{E}[A]=M_{11}$.  The gradient of $\overline A/\sqrt{\overline B\overline C}$ at the population moments is

$$
g = \left(\frac{1}{M_{20}},
-\frac{M_{11}}{2M_{20}^2},
-\frac{M_{11}}{2M_{20}^2}\right).
$$

Using the covariance matrix of $(A,B,C)$ from the same mixed moments, the script estimates

$$
\operatorname{Var}(\widehat\rho_q)\approx \frac{g^\top\Sigma g}{d}.
$$

Then it maps correlation uncertainty to angle uncertainty through

$$
\operatorname{Std}(\widehat\theta_q)
\approx
\sqrt{\frac{\operatorname{Var}(\widehat\rho_q)}{1-\rho_q^2}}.
$$

### Arcsine-law reference

For pure sign quantization,

$$
T(z)=\operatorname{sign}(z),
$$

the exact Gaussian transfer is the arcsine law

$$
C_{\operatorname{sign}}(\rho)=\frac{2}{\pi}\arcsin(\rho).
$$

The script can generate an `arcsine_alignment.pdf` sanity-check plot comparing this formula with Gaussian Monte Carlo samples.

### Fake floating-point formats

A fake-FP format is parameterized by explicit exponent bits $E$ and explicit mantissa/fraction bits $M$.  The total width is

$$
1+E+M,
$$

where the extra bit is the sign bit.

The script quantizes to the custom format and returns float64 dequantized values.  It uses nearest-integer rounding through `numpy.rint`, supports subnormals, preserves NaNs, and saturates infinities or out-of-range finite values to the largest finite value of the format.

The default exponent bias is

$$
\operatorname{bias}=2^{E-1}-1.
$$

The minimum normal exponent is

$$
 e_{\min}=1-\operatorname{bias}.
$$

The subnormal grid spacing is

$$
\operatorname{sub\_step}=\frac{2^{e_{\min}}}{2^M}.
$$

The script supports three FP modes.

#### IEEE-like mode

This is the default debugged mode.  Exponent code zero is used for zero/subnormals and exponent all-ones is reserved for special values.  Therefore `E1M*` is invalid in IEEE-like mode because no normal finite exponent code remains.

The maximum finite exponent is

$$
 e_{\max}=2^E-2-\operatorname{bias}.
$$

The largest finite value is

$$
(2-2^{-M})2^{e_{\max}}.
$$

Examples:

| format | bias | $e_{\min}$ | $e_{\max}$ | max finite |
|---|---:|---:|---:|---:|
| FP8 E5M2 IEEE-like | 15 | -14 | 15 | 57344 |
| FP8 E4M3 IEEE-like | 7 | -6 | 7 | 240 |
| FP8 E3M4 IEEE-like | 3 | -2 | 3 | 15.5 |
| FP6 E2M3 IEEE-like | 1 | 0 | 1 | 3.75 |
| FP5 E2M2 IEEE-like | 1 | 0 | 1 | 3.5 |
| FP4 E2M1 IEEE-like | 1 | 0 | 1 | 3 |

#### All-finite mode

All exponent codes are finite.  Exponent zero still provides zero/subnormal behavior, but exponent all-ones is not reserved for infinities or NaNs.

The maximum finite exponent is

$$
 e_{\max}=2^E-1-\operatorname{bias}.
$$

The largest finite value is

$$
(2-2^{-M})2^{e_{\max}}.
$$

Use this mode for explicit all-finite experiments, including `E1M*` formats:

```bash
--fp-formats finite:e1m4 finite:e1m3 finite:e1m2
```

For example, finite `FP4 E2M1` has max finite value $6$, while IEEE-like `FP4 E2M1` has max finite value $3$.

#### FN-style mode

FN-style mode is useful for E4M3FN-like experiments.  It treats the all-ones exponent as finite, but reserves the largest mantissa pattern at the largest exponent as a non-finite or NaN-like top pattern.

The top finite mantissa index is

$$
2^M-2,
$$

so the largest finite value is

$$
\left(1+\frac{2^M-2}{2^M}\right)2^{e_{\max}}.
$$

For `E4M3FN`, this gives max finite value $448$.

Example:

```bash
--fp-formats e5m2 e4m3fn e3m4 finite:e2m1
```

or use the ML preset:

```bash
--fp-preset ml
```

### Debugged fake-FP overflow behavior

The old fake-FP path had a top-exponent mantissa-rounding bug.  If mantissa rounding overflowed at the largest finite exponent, the exponent was incremented and then clipped back to the largest exponent.  This could collapse an out-of-range value to

$$
1.0\cdot 2^{e_{\max}}
$$

instead of saturating to the true largest finite value.

The debugged behavior is:

1. If mantissa rounding overflows and exponent headroom remains, carry into the next exponent and set the mantissa to zero.
2. If mantissa rounding overflows at $e_{\max}$, saturate the mantissa to the largest finite mantissa allowed by the mode.
3. Apply the final maximum-finite clamp.

This makes large values saturate monotonically instead of wrapping or collapsing.

### Fake-FP Gaussian transfer

For FP formats, the script estimates the same Gaussian transfer principle:

$$
C_{\rm fp}(\rho)=
\frac{\mathbb{E}[F(X)F(Y)]}
{\sqrt{\mathbb{E}[F(X)^2]\,\mathbb{E}[F(Y)^2]}},
$$

where $F$ is the fake-FP quantizer and $(X,Y)$ is a correlated standard-normal pair.

The current FP transfer is estimated by Monte Carlo in `fp_theory_mc`, not by deterministic quadrature.  This is why a solid FP curve can look slightly wobbly when `--fp-theory-samples` is modest.  The line is a connected set of per-angle Gaussian-transfer Monte Carlo estimates; it is not an interpolation of the finite-vector empirical averages.

The finite-vector FP empirical path samples actual unit-vector pairs at the requested angle, multiplies coordinates by $\sqrt d$ to match the standard-normal coordinate scale, fake-quantizes those scaled coordinates, and then computes the angle between the quantized vectors.

### INT-versus-FP same-width comparisons

The comparison mode groups selected FP formats by total bit-width and overlays them with the matching integer quantizer:

```text
INT4 versus all selected FP4 formats
INT5 versus all selected FP5 formats
INT6 versus all selected FP6 formats
INT8 versus all selected FP8 formats
```

This is a same-storage-width comparison.  It does not imply that INT and FP have the same dynamic range, clipping rule, or representable-value distribution.

## FP format specification syntax

Accepted examples include:

```text
e4m3
E5M2
fp8:e4m3
ieee:e4m3
finite:e1m4
fn:e4m3
e4m3fn
fp8:e4m3fn
float4e2m1
```

Any explicit total-bit prefix is checked.  For example, `fp8:e4m3` is valid because $1+4+3=8$, while `fp6:e4m3` is rejected.

Useful presets:

| preset | meaning |
|---|---|
| `debugged-ieee` | default IEEE-like formats, with invalid IEEE-like `E1M*` omitted |
| `ml` | compact ML-style set: E5M2, E4M3FN, E3M4, finite FP4 E2M1 |
| `legacy-finite` | all-finite version of the older broad E/M list, including E1M* experiments |

## CSV outputs

Integer CSV files contain, per bit-width and angle:

```text
bits
angle_deg
rho_true
rho_quant_theory
quant_angle_deg_theory
distortion_deg_theory
std_distortion_deg_delta_method
mean_distortion_deg_empirical
std_distortion_deg_empirical
valid_trials
dim
scale_mode
clip_sigma
```

Fake-FP CSV files contain, per format and angle:

```text
format
mode
total_bits
exp_bits
mant_bits
bias
emin
emax
min_normal
min_positive_quantum
max_finite
angle_deg
distortion_deg_theory_mc
mean_distortion_deg_empirical
std_distortion_deg_empirical
valid_trials
dim
```

INT-versus-FP comparison CSV files include both the integer transfer rows and the matching fake-FP transfer rows for each total bit-width.

## Reproducibility and runtime tips

Use `--seed` to make runs reproducible.

For fast debugging:

```bash
python gaussian_transfer_distortion_debugged_compare.py \
  --compare-only \
  --dim 512 \
  --bits 6 8 \
  --fp-formats e2m3 e5m2 e4m3 e3m4 \
  --angles-start 0 \
  --angles-stop 90 \
  --angles-step 5 \
  --trials 50 \
  --quad-nodes 32 \
  --fp-theory-samples 100000 \
  --output quick/base.pdf
```

For cleaner final figures, increase:

```text
--dim 4096
--angles-step 1
--trials 300 or higher
--quad-nodes 96
--fp-theory-samples 500000 or higher
```

Integer theory can become slower at high bit-width because the quadrature integrates over more quantization bins.  FP theory can become smoother but slower as `--fp-theory-samples` increases.

## Notes and limitations

The Gaussian transfer theory is an asymptotic high-dimensional model.  The finite-vector Monte Carlo points are the direct finite-dimensional simulation check.

The `std` and `maxabs` empirical integer paths use the actual sample standard deviation or maximum absolute coordinate for each vector.  The corresponding theory uses a fixed effective threshold, so small mismatches are expected.

The fake-FP transfer curve is currently Monte Carlo estimated.  It is a theory-style estimate of the infinite-coordinate Gaussian transfer, but it is still random.  Increasing `--fp-theory-samples` reduces the visual wobble.

IEEE-like `E4M3` and `FP4 E2M1` are not the same as all-finite or FN-style ML formats.  Use `--fp-preset ml`, `e4m3fn`, or `finite:e2m1` when those semantics are intended.
