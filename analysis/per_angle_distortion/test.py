
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

rng = np.random.default_rng(7)

# ----------------------------
# Part 1: simple capacity scaling law
# ----------------------------
dims = np.array([64, 128, 256, 512, 1024, 2048, 4096])
bits = {
    "binary": 1.0,
    "ternary": np.log2(3),
    "int3": 3.0,
    "fp8-like": 8.0,
}

# Effective angular capacity proxy:
# larger = more distinguishable local directions / concepts.
# This is not literal parameter count; it is a directional-resolution proxy.
capacity_rows = []
for name, b in bits.items():
    for d in dims:
        capacity_rows.append({
            "format": name,
            "dimension": d,
            "bits_or_log2_levels": b,
            "capacity_proxy_d_times_bits": d * b,
        })

capacity_df = pd.DataFrame(capacity_rows)

plt.figure(figsize=(8, 5.5))
for name, b in bits.items():
    plt.plot(dims, dims * b, marker="o", label=name)

plt.xlabel("Vector dimension")
plt.ylabel("Capacity proxy: d × log2(levels)")
plt.title("Concept capacity proxy vs dimension and quantization")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()
capacity_path = "concept_capacity_proxy_vs_dimension_quantization.png"
plt.savefig(capacity_path, dpi=180)
plt.show()

# ----------------------------
# Part 2: spherical-cap collapse simulation
# ----------------------------

def normalize(x, axis=-1, eps=1e-12):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)

def sample_cap(center, n, cap_noise):
    # center is unit vector; samples are center + isotropic noise, renormalized
    x = center[None, :] + cap_noise * rng.normal(size=(n, center.shape[0]))
    return normalize(x)

def quantize_binary(x):
    # sign quantization, normalized
    q = np.where(x >= 0, 1.0, -1.0)
    return normalize(q)

def quantize_ternary(x, threshold=0.5):
    # threshold relative to per-vector std; produces {-1,0,+1}
    s = np.std(x, axis=1, keepdims=True) + 1e-12
    q = np.where(x > threshold * s, 1.0, np.where(x < -threshold * s, -1.0, 0.0))
    # avoid all-zero vectors
    all_zero = np.linalg.norm(q, axis=1) == 0
    if np.any(all_zero):
        idx = np.argmax(np.abs(x[all_zero]), axis=1)
        q[all_zero, idx] = np.sign(x[all_zero, idx])
    return normalize(q)

def quantize_uniform_levels(x, levels=8):
    # symmetric per-vector maxabs quantization to int-like levels
    maxabs = np.max(np.abs(x), axis=1, keepdims=True) + 1e-12
    # levels=8 approximates signed int3-ish grid
    qmin, qmax = -(levels // 2), levels // 2 - 1
    scaled = x / maxabs * qmax
    q = np.clip(np.round(scaled), qmin, qmax)
    deq = q / max(qmax, 1) * maxabs
    return normalize(deq)

def quantize_fp8_like(x):
    # Very rough fp8-like model: quantize mantissa coarsely but preserve exponent-ish scale.
    # This is not IEEE E4M3; it is just a directional "finer than int3" toy model.
    ax = np.abs(x) + 1e-12
    signs = np.sign(x)
    exponents = np.floor(np.log2(ax))
    mant = ax / (2.0 ** exponents)
    mant_q = np.round((mant - 1.0) * 8) / 8 + 1.0  # ~3 mantissa bits
    q = signs * mant_q * (2.0 ** exponents)
    return normalize(q)

def mean_pairwise_angle_deg(x, max_pairs=6000):
    n = x.shape[0]
    pairs = rng.integers(0, n, size=(max_pairs, 2))
    mask = pairs[:, 0] != pairs[:, 1]
    pairs = pairs[mask]
    dots = np.sum(x[pairs[:,0]] * x[pairs[:,1]], axis=1)
    dots = np.clip(dots, -1, 1)
    return np.degrees(np.arccos(dots)).mean()

def unique_code_fraction_binary_like(q):
    # For low-bit collapse, estimate exact code collisions after sign/ternary/int grids.
    # Here use rounded normalized vectors as a generic hash.
    rounded = np.round(q, 3)
    codes = {tuple(row) for row in rounded}
    return len(codes) / len(q)

D = 512
n_caps = 16
points_per_cap = 128
cap_noise_values = [0.015, 0.03, 0.06, 0.12, 0.24]

formats = {
    "binary": quantize_binary,
    "ternary": quantize_ternary,
    "int3": lambda x: quantize_uniform_levels(x, levels=8),
    "fp8-like": quantize_fp8_like,
}

sim_rows = []

for cap_noise in cap_noise_values:
    centers = normalize(rng.normal(size=(n_caps, D)))
    for cap_id in range(n_caps):
        x = sample_cap(centers[cap_id], points_per_cap, cap_noise)
        original_spread = mean_pairwise_angle_deg(x)
        
        for fmt, fn in formats.items():
            q = fn(x)
            q_spread = mean_pairwise_angle_deg(q)
            
            # "collapse ratio": quantized within-cap spread / original within-cap spread.
            # <1 means the cap collapsed inward; >1 means quantization exploded/noised the cap.
            collapse_ratio = q_spread / (original_spread + 1e-12)
            
            # Angular error from original to quantized vector
            dots = np.sum(x * q, axis=1)
            dots = np.clip(dots, -1, 1)
            angle_err = np.degrees(np.arccos(dots))
            
            sim_rows.append({
                "D": D,
                "n_caps": n_caps,
                "points_per_cap": points_per_cap,
                "cap_noise": cap_noise,
                "format": fmt,
                "original_within_cap_spread_deg": original_spread,
                "quantized_within_cap_spread_deg": q_spread,
                "spread_ratio_quantized_over_original": collapse_ratio,
                "mean_angle_error_deg": angle_err.mean(),
                "p95_angle_error_deg": np.percentile(angle_err, 95),
                "unique_fraction_rounded": unique_code_fraction_binary_like(q),
            })

sim_df = pd.DataFrame(sim_rows)
agg = sim_df.groupby(["cap_noise", "format"], as_index=False).mean(numeric_only=True)

# Plot mean angular error
plt.figure(figsize=(8, 5.5))
for fmt in formats:
    sub = agg[agg["format"] == fmt]
    plt.plot(sub["cap_noise"], sub["mean_angle_error_deg"], marker="o", label=fmt)

plt.xlabel("Cap noise / within-cap spread control")
plt.ylabel("Mean angle error after quantization (degrees)")
plt.title("Quantization angular error inside spherical caps")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()
err_path = "cap_quantization_mean_angle_error.png"
plt.savefig(err_path, dpi=180)
plt.show()

# Plot spread ratio
plt.figure(figsize=(8, 5.5))
for fmt in formats:
    sub = agg[agg["format"] == fmt]
    plt.plot(sub["cap_noise"], sub["spread_ratio_quantized_over_original"], marker="o", label=fmt)

plt.axhline(1.0, linestyle="--", linewidth=1)
plt.xlabel("Cap noise / within-cap spread control")
plt.ylabel("Quantized spread / original spread")
plt.title("Spherical-cap collapse or expansion after quantization")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()
spread_path = "cap_quantization_spread_ratio.png"
plt.savefig(spread_path, dpi=180)
plt.show()

capacity_path, err_path, spread_path

