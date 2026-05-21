
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

rng = np.random.default_rng(2026)

# ----------------------------
# Closed-form-ish approximation
# ----------------------------
# For random-ish vectors quantized with uniform signed b-bit quantization:
# coordinate quantization noise variance roughly Delta^2 / 12.
# With max-abs scaling over D Gaussian coordinates:
# max |x_i| ~ sqrt(2 log D) * sigma.
# step Delta ~ 2 maxabs / (2^b - 1)
# If x is normalized and isotropic, sigma^2 ~ 1/D.
# Noise-to-signal ratio per vector:
# rho^2 = E||e||^2 / E||x||^2 ~ D * Delta^2 / 12
#       ~ [4 * 2 log D / (12 * (2^b - 1)^2)]
#       ~ [2 log D / (3 * (2^b - 1)^2)]
# Small-angle vector direction error alpha ~ atan(rho).
# Pairwise angle distortion around a generic angle can be modeled by:
# abs distortion ~ C * alpha * sqrt(1 + |cos(theta)|)
# and signed bias pulls angles toward 90 degrees:
# signed distortion ~ alpha^2 * cot(theta) in rough sign direction.
#
# This is a heuristic, not a theorem.

def approx_vector_error_deg(bits, D, C=1.0):
    rho = np.sqrt(2 * np.log(D) / (3 * (2**bits - 1)**2))
    return np.degrees(np.arctan(C * rho))

bits_grid = np.array([2, 3, 4, 5, 6, 8], dtype=float)
dims = np.array([256, 512, 1024, 2048, 4096, 8192], dtype=int)

closed_rows = []
for D in dims:
    for b in bits_grid:
        closed_rows.append({
            "D": D,
            "bits": b,
            "approx_single_vector_direction_error_deg": approx_vector_error_deg(b, D),
        })
closed_df = pd.DataFrame(closed_rows)

plt.figure(figsize=(8.5,5.4))
for D in dims:
    sub = closed_df[closed_df["D"] == D]
    plt.plot(sub["bits"], sub["approx_single_vector_direction_error_deg"], marker="o", label=f"D={D}")
plt.xlabel("Quantization bits")
plt.ylabel("Approx single-vector direction error (degrees)")
plt.title("Closed-form approximation: direction error vs bits and dimension")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()
closed_path = "closed_form_direction_error_vs_bits_D.png"
plt.savefig(closed_path, dpi=180)
plt.show()

# ----------------------------
# Attention head degradation under PTQ
# ----------------------------

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)

def normalize_rows(x, eps=1e-12):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)

def quant_symmetric_tensor(x, bits):
    # Per-tensor symmetric maxabs PTQ.
    qmax = 2**(bits-1) - 1
    qmin = -2**(bits-1)
    m = np.max(np.abs(x)) + 1e-12
    q = np.clip(np.round(x / m * qmax), qmin, qmax)
    return q / qmax * m

def quant_binary_tensor(x):
    scale = np.mean(np.abs(x)) + 1e-12
    return scale * np.where(x >= 0, 1.0, -1.0)

def quant_ternary_tensor(x, threshold=0.7):
    scale = np.mean(np.abs(x)) + 1e-12
    t = threshold * scale
    q = np.where(x > t, scale, np.where(x < -t, -scale, 0.0))
    return q

def kl_div(p, q, eps=1e-12):
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    return np.sum(p * (np.log(p) - np.log(q)), axis=-1)

def run_attention_sim(d_model=512, d_head=64, seq_len=128, batch=64, repeats=12):
    formats = ["binary", "ternary", "int3", "int4", "int5", "int8"]
    rows = []
    
    for rep in range(repeats):
        # Random activations and random attention projection weights.
        X = rng.normal(size=(batch, seq_len, d_model)) / np.sqrt(d_model)
        Wq = rng.normal(size=(d_model, d_head)) / np.sqrt(d_model)
        Wk = rng.normal(size=(d_model, d_head)) / np.sqrt(d_model)
        Wv = rng.normal(size=(d_model, d_head)) / np.sqrt(d_model)
        
        Q = X @ Wq
        K = X @ Wk
        V = X @ Wv
        
        logits = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d_head)
        attn = softmax(logits, axis=-1)
        out = attn @ V
        
        for fmt in formats:
            if fmt == "binary":
                qWq, qWk, qWv = quant_binary_tensor(Wq), quant_binary_tensor(Wk), quant_binary_tensor(Wv)
                bit_value = 1.0
            elif fmt == "ternary":
                qWq, qWk, qWv = quant_ternary_tensor(Wq), quant_ternary_tensor(Wk), quant_ternary_tensor(Wv)
                bit_value = np.log2(3)
            else:
                b = int(fmt.replace("int", ""))
                qWq, qWk, qWv = quant_symmetric_tensor(Wq, b), quant_symmetric_tensor(Wk, b), quant_symmetric_tensor(Wv, b)
                bit_value = b
            
            qQ = X @ qWq
            qK = X @ qWk
            qV = X @ qWv
            
            qlogits = qQ @ np.swapaxes(qK, -1, -2) / np.sqrt(d_head)
            qattn = softmax(qlogits, axis=-1)
            qout = qattn @ qV
            
            # Metrics
            logit_rmse = np.sqrt(np.mean((qlogits - logits)**2))
            attn_kl = np.mean(kl_div(attn.reshape(-1, seq_len), qattn.reshape(-1, seq_len)))
            top1_flip = np.mean(np.argmax(attn, axis=-1) != np.argmax(qattn, axis=-1))
            out_rel_rmse = np.sqrt(np.mean((qout - out)**2)) / (np.sqrt(np.mean(out**2)) + 1e-12)
            
            # Mean row-wise cosine of attention output vectors.
            out_flat = out.reshape(-1, d_head)
            qout_flat = qout.reshape(-1, d_head)
            dots = np.sum(out_flat*qout_flat, axis=1)
            denom = np.linalg.norm(out_flat, axis=1)*np.linalg.norm(qout_flat, axis=1) + 1e-12
            out_cos = np.mean(dots/denom)
            
            rows.append({
                "repeat": rep,
                "format": fmt,
                "bits_or_log2_levels": bit_value,
                "d_model": d_model,
                "d_head": d_head,
                "seq_len": seq_len,
                "batch": batch,
                "logit_rmse": logit_rmse,
                "attention_kl": attn_kl,
                "top1_attention_flip_rate": top1_flip,
                "output_relative_rmse": out_rel_rmse,
                "output_mean_cosine": out_cos,
            })
    return pd.DataFrame(rows)

attn_df = run_attention_sim()
attn_summary = attn_df.groupby("format", as_index=False).agg({
    "bits_or_log2_levels": "mean",
    "logit_rmse": ["mean", "std"],
    "attention_kl": ["mean", "std"],
    "top1_attention_flip_rate": ["mean", "std"],
    "output_relative_rmse": ["mean", "std"],
    "output_mean_cosine": ["mean", "std"],
})
attn_summary.columns = ["_".join([c for c in col if c]) for col in attn_summary.columns.to_flat_index()]

# Ordered formats
order = ["binary", "ternary", "int3", "int4", "int5", "int8"]
plot_df = attn_df.copy()
plot_df["format"] = pd.Categorical(plot_df["format"], order, ordered=True)
agg = plot_df.groupby("format", observed=True).mean(numeric_only=True).reset_index()

# Plot attention KL
plt.figure(figsize=(8,5.2))
plt.plot(agg["format"].astype(str), agg["attention_kl"], marker="o")
plt.xlabel("Quantization format")
plt.ylabel("Mean KL(original attention || quantized attention)")
plt.title("Attention distribution drift under PTQ")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()
kl_path = "ata/attention_ptq_kl_by_format.png"
plt.savefig(kl_path, dpi=180)
plt.show()

# Plot top1 flip
plt.figure(figsize=(8,5.2))
plt.plot(agg["format"].astype(str), 100*agg["top1_attention_flip_rate"], marker="o")
plt.xlabel("Quantization format")
plt.ylabel("Top-1 attention target flip rate (%)")
plt.title("Attention argmax instability under PTQ")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()
flip_path = "attention_ptq_top1_flip_by_format.png"
plt.savefig(flip_path, dpi=180)
plt.show()

# Plot output rel RMSE and cosine
plt.figure(figsize=(8,5.2))
plt.plot(agg["format"].astype(str), agg["output_relative_rmse"], marker="o", label="relative RMSE")
plt.xlabel("Quantization format")
plt.ylabel("Output relative RMSE")
plt.title("Attention output degradation under PTQ")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()
rmse_path = "attention_ptq_output_rmse_by_format.png"
plt.savefig(rmse_path, dpi=180)
plt.show()

plt.figure(figsize=(8,5.2))
plt.plot(agg["format"].astype(str), agg["output_mean_cosine"], marker="o")
plt.xlabel("Quantization format")
plt.ylabel("Mean cosine(original output, quantized output)")
plt.title("Attention output directional preservation under PTQ")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()
cos_path = "attention_ptq_output_cosine_by_format.png"
plt.savefig(cos_path, dpi=180)
plt.show()

closed_path, kl_path, flip_path, rmse_path, cos_path

