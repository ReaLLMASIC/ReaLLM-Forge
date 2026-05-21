import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

rng = np.random.default_rng(123)

def normalize(x):
    return x / (np.linalg.norm(x) + 1e-12)

def make_pair(D, angle_deg):
    a = normalize(rng.normal(size=D))
    u = rng.normal(size=D)
    u = u - np.dot(u, a) * a
    u = normalize(u)
    theta = np.deg2rad(angle_deg)
    b = np.cos(theta) * a + np.sin(theta) * u
    return a, b

def angle(a, b):
    c = np.dot(a, b) / ((np.linalg.norm(a)*np.linalg.norm(b)) + 1e-12)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))

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
    # Signed symmetric-ish quantization:
    # int3: [-4, 3], int4: [-8, 7], int5: [-16, 15]
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
initial_angles = np.array([1, 5, 10, 20, 30, 45, 60, 75, 90, 105, 120, 135, 150, 160, 170, 175, 179], dtype=float)
trials = 180

rows = []

for fmt, qfn in quantizers.items():
    for D in dims_list:
        for init_angle in initial_angles:
            errs = []
            signed = []
            for _ in range(trials):
                a, b = make_pair(D, init_angle)
                qa, qb = qfn(a), qfn(b)
                qang = angle(qa, qb)
                delta = qang - init_angle
                errs.append(abs(delta))
                signed.append(delta)
            errs = np.array(errs)
            signed = np.array(signed)
            rows.append({
                "format": fmt,
                "D": D,
                "initial_angle_deg": init_angle,
                "trials": trials,
                "mean_abs_distortion_deg": errs.mean(),
                "median_abs_distortion_deg": np.median(errs),
                "p10_abs_distortion_deg": np.percentile(errs, 10),
                "p90_abs_distortion_deg": np.percentile(errs, 90),
                "mean_signed_distortion_deg": signed.mean(),
                "std_signed_distortion_deg": signed.std(),
            })

df = pd.DataFrame(rows)

# Save one plot per quantizer: dimension traces
plot_paths = []
for fmt in quantizers:
    sub_fmt = df[df["format"] == fmt]
    plt.figure(figsize=(8.5, 5.4))
    for D in dims_list:
        sub = sub_fmt[sub_fmt["D"] == D]
        plt.plot(sub["initial_angle_deg"], sub["mean_abs_distortion_deg"], marker="o", label=f"D={D}")
    plt.xlabel("Initial angle between original vectors (degrees)")
    plt.ylabel("Mean absolute distortion after quantization (degrees)")
    plt.title(f"{fmt}: distortion vs angle across starting dimensions")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    path = f"{fmt}_distortion_vs_angle_across_dimensions.png"
    plt.savefig(path, dpi=180)
    plt.show()
    plot_paths.append(path)

# Combined plot at D=2048
plt.figure(figsize=(8.5, 5.4))
D_focus = 2048
for fmt in quantizers:
    sub = df[(df["format"] == fmt) & (df["D"] == D_focus)]
    plt.plot(sub["initial_angle_deg"], sub["mean_abs_distortion_deg"], marker="o", label=fmt)
plt.xlabel("Initial angle between original vectors (degrees)")
plt.ylabel("Mean absolute distortion after quantization (degrees)")
plt.title(f"Format comparison at D={D_focus}")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()
combined_path = f"format_comparison_distortion_D{D_focus}.png"
plt.savefig(combined_path, dpi=180)
plt.show()

plot_paths + [combined_path]

