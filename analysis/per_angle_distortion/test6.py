import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

rng = np.random.default_rng(777)

def normalize(x, axis=None, eps=1e-12):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)

def angle_deg(a, b):
    c = np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))

def make_pair(D, angle_deg_target, mode):
    """
    mode='unit': exact unit vectors with target angle.
    mode='gaussian': same directions but with random Gaussian-like radii.
    Angle is identical before quantization; scale differs.
    """
    a = normalize(rng.normal(size=D))
    u = rng.normal(size=D)
    u = u - np.dot(u, a) * a
    u = normalize(u)
    theta = np.deg2rad(angle_deg_target)
    b = np.cos(theta) * a + np.sin(theta) * u

    if mode == "unit":
        return a, b
    elif mode == "gaussian":
        # Random radius resembling norm of N(0, I_D), with independent scales.
        ra = np.sqrt(rng.chisquare(D))
        rb = np.sqrt(rng.chisquare(D))
        return ra * a, rb * b
    else:
        raise ValueError(mode)

def q_binary(x):
    return normalize(np.where(x >= 0, 1.0, -1.0))

def q_ternary(x):
    s = np.std(x) + 1e-12
    q = np.where(x > 0.5*s, 1.0, np.where(x < -0.5*s, -1.0, 0.0))
    if np.linalg.norm(q) == 0:
        idx = np.argmax(np.abs(x))
        q[idx] = np.sign(x[idx])
    return normalize(q)

def q_symmetric_int(x, bits):
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    m = np.max(np.abs(x)) + 1e-12
    q = np.clip(np.round(x / m * qmax), qmin, qmax)
    deq = q / max(qmax, 1) * m
    return normalize(deq)

quantizers = {
    "binary": q_binary,
    "ternary": q_ternary,
    "int3": lambda x: q_symmetric_int(x, 3),
    "int4": lambda x: q_symmetric_int(x, 4),
    "int5": lambda x: q_symmetric_int(x, 5),
}

dims_list = [256, 512, 1024, 2048, 4096, 8192]
angles = np.array([1, 5, 10, 20, 30, 45, 60, 75, 90, 105, 120, 135, 150, 160, 170, 175, 179], dtype=float)
trials = 120
modes = ["unit", "gaussian"]

rows = []
for mode in modes:
    for fmt, qfn in quantizers.items():
        for D in dims_list:
            for init_angle in angles:
                errs = []
                signed = []
                for _ in range(trials):
                    a, b = make_pair(D, init_angle, mode)
                    qa, qb = qfn(a), qfn(b)
                    qang = angle_deg(qa, qb)
                    delta = qang - init_angle
                    errs.append(abs(delta))
                    signed.append(delta)
                rows.append({
                    "mode": mode,
                    "format": fmt,
                    "D": D,
                    "initial_angle_deg": init_angle,
                    "mean_abs_distortion_deg": np.mean(errs),
                    "median_abs_distortion_deg": np.median(errs),
                    "p10_abs_distortion_deg": np.percentile(errs, 10),
                    "p90_abs_distortion_deg": np.percentile(errs, 90),
                    "mean_signed_distortion_deg": np.mean(signed),
                    "std_signed_distortion_deg": np.std(signed),
                })

df = pd.DataFrame(rows)

# 1. Combined format comparison at D=2048, one panel-like separate figure for mean abs distortion
D_focus = 2048
plt.figure(figsize=(9, 5.8))
for mode, ls in [("unit", "-"), ("gaussian", "--")]:
    for fmt in quantizers:
        sub = df[(df["mode"] == mode) & (df["format"] == fmt) & (df["D"] == D_focus)]
        plt.plot(
            sub["initial_angle_deg"],
            sub["mean_abs_distortion_deg"],
            linestyle=ls,
            marker="o" if mode == "unit" else "x",
            label=f"{fmt} {mode}",
            alpha=0.9,
        )
plt.xlabel("Initial angle between original vectors (degrees)")
plt.ylabel("Mean absolute angular distortion (degrees)")
plt.title(f"Unit vs Gaussian vectors: format comparison at D={D_focus}")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.legend(ncol=2, fontsize=8)
plt.tight_layout()
format_compare_path = "unit_vs_gaussian_format_comparison_D2048.png"
plt.savefig(format_compare_path, dpi=180)
plt.show()

# 2. Distribution comparison: average over angles, distortion vs D for each format
avg_df = df.groupby(["mode", "format", "D"], as_index=False).mean(numeric_only=True)
plt.figure(figsize=(9, 5.8))
for mode, ls in [("unit", "-"), ("gaussian", "--")]:
    for fmt in quantizers:
        sub = avg_df[(avg_df["mode"] == mode) & (avg_df["format"] == fmt)]
        plt.plot(
            sub["D"], sub["mean_abs_distortion_deg"],
            linestyle=ls,
            marker="o" if mode == "unit" else "x",
            label=f"{fmt} {mode}",
        )
plt.xscale("log", base=2)
plt.xlabel("Starting dimension D")
plt.ylabel("Mean distortion averaged over initial angles (degrees)")
plt.title("Unit vs Gaussian vectors: distortion vs starting dimension")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.legend(ncol=2, fontsize=8)
plt.tight_layout()
dist_vs_D_path = "unit_vs_gaussian_distortion_vs_D.png"
plt.savefig(dist_vs_D_path, dpi=180)
plt.show()

# 3. Signed bias at D=2048
plt.figure(figsize=(9, 5.8))
for mode, ls in [("unit", "-"), ("gaussian", "--")]:
    for fmt in quantizers:
        sub = df[(df["mode"] == mode) & (df["format"] == fmt) & (df["D"] == D_focus)]
        plt.plot(
            sub["initial_angle_deg"],
            sub["mean_signed_distortion_deg"],
            linestyle=ls,
            marker="o" if mode == "unit" else "x",
            label=f"{fmt} {mode}",
            alpha=0.9,
        )
plt.axhline(0, linestyle="--", linewidth=1)
plt.xlabel("Initial angle between original vectors (degrees)")
plt.ylabel("Mean signed distortion: quantized angle - original angle (degrees)")
plt.title(f"Unit vs Gaussian vectors: signed bias at D={D_focus}")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.legend(ncol=2, fontsize=8)
plt.tight_layout()
signed_bias_path = "unit_vs_gaussian_signed_bias_D2048.png"
plt.savefig(signed_bias_path, dpi=180)
plt.show()

format_compare_path, dist_vs_D_path, signed_bias_path

