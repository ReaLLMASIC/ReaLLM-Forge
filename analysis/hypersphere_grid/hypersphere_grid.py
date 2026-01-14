# hypersphere_grid.py

# Generate quasi-uniform point sets on the surface of an (n-1)-sphere embedded in R^n
# (default n=384) using lattice-like low-discrepancy methods, a Fibonacci grid (S^2),
# or HEALPix (S^2, if healpy is available).

# Features
# --------
# - Point-set generators: kronecker (default), halton, random; plus S^2: fibonacci, healpix.
# - Traversal & geometry: angular distance, SLERP midpoint, SLERP polyline.
# - Snap-to-grid: cosine similarity argmax.
# - Statistics & plots: NN angles, clustering (CV), coherence, MSIP vs Welch bound,
#   halfspace discrepancy, approx covering radius, pairwise cosine hist vs theory.
# - Comparators: naive lat/long (S^2) or naive angle-uniform hyperspherical (n>3).
# - k-NN graph builder: compute and save k-NN (cosine) graph (NPZ/CSV), optional symmetrization.
# - CSV export of stats.
# - Benchmark mode: sweep N over methods, write CSV + optional metric-vs-N plots.

# Usage snippets
# --------------
# Stats + plots + comparator (default dim=384)
# python hypersphere_grid.py --dim 384 --num 2048 --method kronecker \
#   --stats --plots out_plots --compare latlong

# # k-NN graph (10-NN), save as NPZ
# python hypersphere_grid.py --load points.npy --knn 10 --knn-out graph.npz --knn-symmetric

# # Export stats to CSV
# python hypersphere_grid.py --dim 384 --num 4096 --method halton --stats --stats-csv-out stats.csv

# # Benchmark sweep
# python hypersphere_grid.py --benchmark --dim 384 \
#   --bench-methods kronecker,halton,random \
#   --bench-Ns 512,1024,2048 \
#   --bench-repeats 1 \
#   --bench-out bench.csv --bench-plots bench_plots
import argparse
import csv
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from typing import Iterable, Tuple, Dict, Any, Optional, List

import numpy as np

# ----------------------------- Utilities -------------------------------------

def _unit_norm(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    nrm = np.linalg.norm(x, axis=axis, keepdims=True)
    nrm = np.maximum(nrm, eps)
    return x / nrm

def _clip_cosine(c: np.ndarray) -> np.ndarray:
    return np.clip(c, -1.0, 1.0)

def angular_distance(u: np.ndarray, v: np.ndarray) -> float:
    u = _unit_norm(u)
    v = _unit_norm(v)
    c = float(np.dot(u, v))
    return float(np.arccos(_clip_cosine(np.array(c))))

def slerp(u: np.ndarray, v: np.ndarray, t: float) -> np.ndarray:
    u = _unit_norm(u).astype(np.float64, copy=False)
    v = _unit_norm(v).astype(np.float64, copy=False)
    c = float(np.dot(u, v))
    c = float(_clip_cosine(np.array(c)))
    theta = math.acos(c)
    if theta < 1e-12:
        return u.copy()
    sin_theta = math.sin(theta)
    a = math.sin((1.0 - t) * theta) / sin_theta
    b = math.sin(t * theta) / sin_theta
    return _unit_norm(a * u + b * v)

def slerp_polyline(u: np.ndarray, v: np.ndarray, steps: int) -> np.ndarray:
    ts = np.linspace(0.0, 1.0, steps + 1)
    pts = np.stack([slerp(u, v, t) for t in ts], axis=0)
    return pts

# ---------------------- Inverse Normal CDF (Acklam) --------------------------

def _inv_norm_cdf(p: np.ndarray) -> np.ndarray:
    a = np.array([
        -3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00
    ])
    b = np.array([
        -5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01
    ])
    c = np.array([
        -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
        -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00
    ])
    d = np.array([
         7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,
         3.754408661907416e+00
    ])
    p = np.asarray(p, dtype=np.float64)
    eps = np.finfo(np.float64).eps
    p = np.clip(p, eps, 1.0 - eps)
    pl = p < 0.02425
    pu = p > 1.0 - 0.02425
    pm = ~(pl | pu)
    x = np.empty_like(p, dtype=np.float64)
    if np.any(pl):
        q = np.sqrt(-2.0 * np.log(p[pl]))
        x[pl] = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                 ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    if np.any(pm):
        q = p[pm] - 0.5
        r = q * q
        x[pm] = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
                 (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)
    if np.any(pu):
        q = np.sqrt(-2.0 * np.log(1.0 - p[pu]))
        x[pu] = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                  ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    return x

# --------------------------- Halton sequence ---------------------------------

def _prime_sieve(n: int) -> np.ndarray:
    sieve = np.ones(n + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i:n+1:i] = False
    return np.flatnonzero(sieve)

def _first_primes(k: int) -> np.ndarray:
    if k < 6:
        bound = 15
    else:
        bound = int(k * (math.log(k) + math.log(math.log(k))) * 1.2) + 10
    primes = _prime_sieve(bound)
    while len(primes) < k:
        bound *= 2
        primes = _prime_sieve(bound)
    return primes[:k]

def _radical_inverse(i: int, base: int) -> float:
    f = 1.0
    r = 0.0
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r

def halton_sequence(dim: int, n: int, start_index: int = 1) -> np.ndarray:
    bases = _first_primes(dim)
    seq = np.empty((n, dim), dtype=np.float64)
    for j, b in enumerate(bases):
        seq[:, j] = np.array([_radical_inverse(i, int(b)) for i in range(start_index, start_index + n)])
    return seq

# ----------------------- Kronecker (rank-1) lattice --------------------------

def _irrational_vector(dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shift = rng.uniform(0.0, 1.0)
    primes = _first_primes(dim)
    alphas = np.sqrt(primes.astype(np.float64) + shift)
    return np.modf(alphas)[0]

def kronecker_sequence(dim: int, n: int, seed: int = 0) -> np.ndarray:
    alpha = _irrational_vector(dim, seed=seed)
    phi = (1.0 + 5.0 ** 0.5) / 2.0
    offset = np.array([math.modf(phi ** (i + 1))[0] for i in range(dim)], dtype=np.float64)
    k = np.arange(1, n + 1, dtype=np.float64)[:, None]
    seq = (k * alpha[None, :] + offset[None, :]) % 1.0
    return seq

# -------------------------- Map to the sphere --------------------------------

def _cube_to_gaussian(u01: np.ndarray) -> np.ndarray:
    return _inv_norm_cdf(u01)

def _points_on_sphere_from_u01(u01: np.ndarray) -> np.ndarray:
    z = _cube_to_gaussian(u01)
    return _unit_norm(z, axis=1)

def halton_sphere(n: int, dim: int) -> np.ndarray:
    u = halton_sequence(dim, n, start_index=1)
    return _points_on_sphere_from_u01(u)

def kronecker_sphere(n: int, dim: int, seed: int = 0) -> np.ndarray:
    u = kronecker_sequence(dim, n, seed=seed)
    return _points_on_sphere_from_u01(u)

def random_sphere(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(size=(n, dim))
    return _unit_norm(z, axis=1)

# ----------------------- S^2-specific constructions --------------------------

def fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n, dtype=np.float64)
    ga = math.pi * (3.0 - 5.0 ** 0.5)  # golden angle
    z = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = i * ga
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    pts = np.stack([x, y, z], axis=1)
    return _unit_norm(pts, axis=1)

def healpix_sphere(n_approx: int) -> np.ndarray:
    try:
        import healpy as hp  # type: ignore
    except Exception as e:
        raise RuntimeError("healpy is required for --method healpix on S^2. Install with `pip install healpy`.") from e
    nside = max(1, int(math.sqrt(max(1, n_approx) / 12.0)))
    npix = 12 * nside * nside
    idx = np.arange(npix, dtype=np.int64)
    vec = np.array(hp.pix2vec(nside, idx)).T  # shape (npix, 3)
    return _unit_norm(vec, axis=1)

# --------------- "Angular lat/long" naive constructions ----------------------

def latlong_grid_s2(n: int) -> np.ndarray:
    n_lat = max(2, int(round(math.sqrt(n))))
    n_lon = max(2, int(round(n / n_lat)))
    lats = np.linspace(-math.pi/2, math.pi/2, n_lat, endpoint=True)
    lons = np.linspace(0.0, 2.0*math.pi, n_lon, endpoint=False)
    xs, ys, zs = [], [], []
    for lat in lats:
        cl = math.cos(lat); sl = math.sin(lat)
        for lon in lons:
            cln = math.cos(lon); sln = math.sin(lon)
            xs.append(cl*cln); ys.append(cl*sln); zs.append(sl)
    P = np.stack([xs, ys, zs], axis=1)
    P = _unit_norm(P, axis=1)
    if P.shape[0] > n:
        P = P[:n]
    return P

def latlong_random_nd(n_points: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if dim < 3:
        raise ValueError("dim must be >=3 for hyperspherical coordinates")
    thetas = rng.uniform(0.0, math.pi, size=(n_points, dim - 2))
    phis = rng.uniform(0.0, 2.0*math.pi, size=(n_points, 1))
    angles = np.concatenate([thetas, phis], axis=1)
    P = np.empty((n_points, dim), dtype=np.float64)
    sin_prod = np.ones(n_points, dtype=np.float64)
    for j in range(dim - 1):
        if j == 0:
            P[:, j] = np.cos(angles[:, 0])
            sin_prod *= np.sin(angles[:, 0])
        elif j < dim - 1 - 1:
            P[:, j] = sin_prod * np.cos(angles[:, j])
            sin_prod *= np.sin(angles[:, j])
        else:
            P[:, j] = sin_prod * np.cos(angles[:, -1])
    P[:, -1] = sin_prod * np.sin(angles[:, -1])
    return _unit_norm(P, axis=1)

# ------------------------------ I/O helpers ----------------------------------

def _save(points: np.ndarray, path: str, fmt: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if fmt == "npy":
        np.save(path, points)
    elif fmt == "csv":
        np.savetxt(path, points, delimiter=",")
    elif fmt == "jsonl":
        with open(path, "w", encoding="utf-8") as f:
            for row in points.tolist():
                f.write(json.dumps(row) + "\n")
    else:
        raise ValueError(f"Unknown format: {fmt}")

def _load_points(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        pts = np.load(path)
    elif ext in (".csv", ".txt"):
        pts = np.loadtxt(path, delimiter=",")
    elif ext in (".jsonl", ".ndjson"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
        pts = np.array(rows, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported file extension for points: {ext}")
    return _unit_norm(np.asarray(pts, dtype=np.float64), axis=1)

def _parse_vector_arg(s: str) -> np.ndarray:
    if os.path.exists(s):
        v = np.asarray(_load_points(s), dtype=np.float64)
        if v.ndim == 1:
            return _unit_norm(v)
        elif v.ndim == 2 and v.shape[0] == 1:
            return _unit_norm(v[0])
        else:
            raise ValueError("Vector file must contain exactly one vector (shape (d,) or (1,d)).")
    else:
        try:
            vals = [float(x.strip()) for x in s.split(",") if x.strip()]
        except Exception as e:
            raise ValueError(f"Could not parse vector from '{s}'") from e
        return _unit_norm(np.asarray(vals, dtype=np.float64))

# --------------------------- Snap-to-grid ------------------------------------

def snap_to_grid(points: np.ndarray, q: np.ndarray) -> Tuple[int, float]:
    q = _unit_norm(q)
    sims = points @ q
    idx = int(np.argmax(sims))
    return idx, float(sims[idx])

# --------------------------- Statistics --------------------------------------

@dataclass
class Stats:
    method: str
    num_points: int
    dim: int
    avg_nn_deg: float
    median_nn_deg: float
    std_nn_deg: float
    cv_nn: float
    min_nn_deg: float
    max_nn_deg: float
    coherence: float
    msip: float
    welch_bound: float
    msip_to_welch: float
    halfspace_discrepancy: float
    halfspace_discrepancy_mean: float
    approx_covering_radius_deg: float
    notes: str = ""

def _nn_cosines(points: np.ndarray, block: int = 2048) -> np.ndarray:
    P = points.astype(np.float64, copy=False)
    N = P.shape[0]
    if N <= block:
        S = P @ P.T
        np.fill_diagonal(S, -np.inf)
        m = S.max(axis=1)
        return m
    else:
        m = np.full(N, -np.inf, dtype=np.float64)
        for i0 in range(0, N, block):
            i1 = min(N, i0 + block)
            B = P[i0:i1] @ P.T
            for k, i in enumerate(range(i0, i1)):
                B[k, i] = -np.inf
            m[i0:i1] = np.maximum(m[i0:i1], np.max(B, axis=1))
        return m

def _sum_squares_all_entries(points: np.ndarray, block: int = 2048) -> float:
    P = points.astype(np.float64, copy=False)
    N = P.shape[0]
    total = 0.0
    for i0 in range(0, N, block):
        i1 = min(N, i0 + block)
        B = P[i0:i1] @ P.T
        total += float(np.sum(B * B))
    return total

def _coherence(points: np.ndarray, block: int = 2048) -> float:
    P = points.astype(np.float64, copy=False)
    N = P.shape[0]
    mx = 0.0
    for i0 in range(0, N, block):
        i1 = min(N, i0 + block)
        B = P[i0:i1] @ P.T
        for k, i in enumerate(range(i0, i1)):
            B[k, i] = 0.0
        mx = max(mx, float(np.max(np.abs(B))))
    return mx

def _uniform_cos_pdf(t: np.ndarray, dim: int) -> np.ndarray:
    a = 0.5
    b = 0.5 * (dim - 1)
    lg = math.lgamma
    log_c = lg(a + b) - lg(a) - lg(b)
    c = math.exp(log_c)
    t = np.asarray(t, dtype=np.float64)
    out = np.zeros_like(t, dtype=np.float64)
    mask = (t >= -1.0) & (t <= 1.0)
    out[mask] = c * np.power(1.0 - t[mask] * t[mask], 0.5 * (dim - 3))
    return out

def _halfspace_discrepancy(points: np.ndarray, probes: int = 256, seed: int = 0) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    dim = points.shape[1]
    W = random_sphere(probes, dim, seed=rng.integers(1<<31))
    fracs = []
    for w in W:
        sims = points @ w
        frac = np.count_nonzero(sims >= 0.0) / points.shape[0]
        fracs.append(abs(frac - 0.5))
    return float(np.max(fracs)), float(np.mean(fracs))

def _approx_covering_radius(points: np.ndarray, probes: int = 256, seed: int = 0, block: int = 2048) -> float:
    rng = np.random.default_rng(seed)
    dim = points.shape[1]
    max_min_angle = 0.0
    for start in range(0, probes, block):
        b = min(block, probes - start)
        W = random_sphere(b, dim, seed=rng.integers(1<<31))
        S = W @ points.T
        m = np.max(S, axis=1)
        ang = np.arccos(_clip_cosine(m))
        max_min_angle = max(max_min_angle, float(np.max(ang)))
    return max_min_angle

def compute_stats(points: np.ndarray, method: str, nn_block: int = 2048,
                  probes: int = 256, seed: int = 0) -> Tuple[Stats, Dict[str, Any]]:
    N, dim = points.shape
    nn_cos = _nn_cosines(points, block=nn_block)
    nn_angles = np.arccos(_clip_cosine(nn_cos))
    nn_deg = np.degrees(nn_angles)
    avg_nn = float(np.mean(nn_deg))
    med_nn = float(np.median(nn_deg))
    std_nn = float(np.std(nn_deg, ddof=1)) if N > 1 else 0.0
    cv = float(std_nn / max(avg_nn, 1e-12))
    min_nn = float(np.min(nn_deg))
    max_nn = float(np.max(nn_deg))
    coh = _coherence(points, block=nn_block)
    sumsq = _sum_squares_all_entries(points, block=nn_block)
    msip = (sumsq - N) / max(N*(N-1), 1) if N > 1 else 0.0
    welch = max((N - dim) / (dim * (N - 1)), 0.0) if N > 1 else 0.0
    msip_ratio = msip / max(welch, 1e-12) if welch > 0 else float('inf') if msip > 0 else 0.0
    hdisc_max, hdisc_mean = _halfspace_discrepancy(points, probes=probes, seed=seed)
    cov_rad = _approx_covering_radius(points, probes=probes, seed=seed, block=min(probes, 1024))

    stats = Stats(
        method=method, num_points=N, dim=dim,
        avg_nn_deg=avg_nn, median_nn_deg=med_nn, std_nn_deg=std_nn, cv_nn=cv,
        min_nn_deg=min_nn, max_nn_deg=max_nn,
        coherence=float(coh), msip=float(msip), welch_bound=float(welch), msip_to_welch=float(msip_ratio),
        halfspace_discrepancy=float(hdisc_max), halfspace_discrepancy_mean=float(hdisc_mean),
        approx_covering_radius_deg=float(np.degrees(cov_rad)),
        notes=""
    )
    extras = {"nn_angles_deg": nn_deg}
    return stats, extras

def sample_pairwise_cos(points: np.ndarray, num_pairs: int = 200000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    N = points.shape[0]
    total_pairs = N * (N - 1) // 2
    num_pairs = min(num_pairs, max(total_pairs, 0))
    if total_pairs <= 0 or num_pairs == 0:
        return np.empty(0, dtype=np.float64)
    idx_i = rng.integers(0, N, size=num_pairs, endpoint=False)
    idx_j = rng.integers(0, N, size=num_pairs, endpoint=False)
    mask = idx_i != idx_j
    idx_i = idx_i[mask]; idx_j = idx_j[mask]
    sims = np.sum(points[idx_i] * points[idx_j], axis=1)
    return sims

# ------------------------------- Plotting ------------------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def plot_nn_hist(nn_deg: np.ndarray, out_path: str, title: str) -> None:
    import matplotlib.pyplot as plt
    plt.figure()
    plt.hist(nn_deg, bins=50, density=True)
    plt.axvline(float(np.mean(nn_deg)), linestyle='--', label='mean')
    plt.xlabel("Nearest-neighbor angle (degrees)")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_cos_hist_with_uniform(cos_samples: np.ndarray, dim: int, out_path: str, title: str) -> None:
    import matplotlib.pyplot as plt
    plt.figure()
    plt.hist(cos_samples, bins=100, density=True, alpha=0.6, label='empirical')
    t = np.linspace(-1, 1, 1000)
    pdf = _uniform_cos_pdf(t, dim)
    plt.plot(t, pdf, label='uniform-on-sphere pdf')
    plt.xlabel("Pairwise cosine")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_metric_bars(stats_list: Iterable[Stats], out_path: str, title: str) -> None:
    import matplotlib.pyplot as plt
    labels = [s.method for s in stats_list]
    metrics = [
        ("avg_nn_deg", "Avg NN angle (deg)"),
        ("cv_nn", "NN coeff. of variation"),
        ("coherence", "Coherence (max |dot|)"),
        ("msip_to_welch", "MSIP / Welch bound"),
        ("halfspace_discrepancy", "Halfspace discrepancy (max)"),
        ("approx_covering_radius_deg", "Approx covering radius (deg)"),
    ]
    for key, label in metrics:
        values = [getattr(s, key) for s in stats_list]
        import matplotlib.pyplot as plt
        plt.figure()
        plt.bar(labels, values)
        plt.ylabel(label)
        plt.title(f"{title} — {label}")
        plt.tight_layout()
        base, ext = os.path.splitext(out_path)
        path_metric = f"{base}_{key}{ext}"
        plt.savefig(path_metric, dpi=150)
        plt.close()

def plot_benchmark_lines(rows: List[Dict[str, Any]], metrics: List[str], out_dir: str, title_prefix: str) -> None:
    import matplotlib.pyplot as plt
    _ensure_dir(out_dir)
    methods = sorted(set(r["method"] for r in rows))
    Ns = sorted(set(int(r["num_points"]) for r in rows))
    for mkey in metrics:
        plt.figure()
        for meth in methods:
            xs = []
            ys = []
            for N in Ns:
                vals = [r[mkey] for r in rows if r["method"] == meth and int(r["num_points"]) == N]
                if len(vals) == 0:
                    continue
                xs.append(N)
                ys.append(float(np.mean(vals)))
            if len(xs) > 0:
                xs_sorted = [x for _, x in sorted(zip(xs, xs))]
                ys_sorted = [y for _, y in sorted(zip(xs, ys))]
                plt.plot(xs_sorted, ys_sorted, marker='o', label=meth)
        plt.xlabel("N (number of points)")
        plt.ylabel(mkey)
        plt.title(f"{title_prefix}: {mkey} vs N")
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"bench_{mkey}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()

# -------------------------- k-NN graph builder --------------------------------

def compute_knn_graph(points: np.ndarray, k: int, block: int, symmetric: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute top-k cosine neighbors for each point.

    Returns (indices, sims) each of shape (N, k), where indices[i] are the neighbors
    of i sorted by descending similarity, with corresponding sims[i]. If symmetric=True,
    we return the symmetric union by ensuring that if (i,j) appears, (j,i) also appears
    among the k entries when possible; due to fixed k, exact union may not fit — here
    we simply mirror and keep top-k per row again.
    """
    P = points.astype(np.float64, copy=False)
    N = P.shape[0]
    k = min(k, max(N-1, 0))
    if k <= 0 or N == 0:
        return np.zeros((N,0), dtype=int), np.zeros((N,0), dtype=np.float64)

    all_idx = np.empty((N, k), dtype=np.int64)
    all_sim = np.empty((N, k), dtype=np.float64)

    for i0 in range(0, N, block):
        i1 = min(N, i0 + block)
        B = P[i0:i1] @ P.T  # (b, N)
        # mask self
        for r, i in enumerate(range(i0, i1)):
            B[r, i] = -np.inf
        # top-k per row
        # argpartition returns indices in arbitrary order among top-k; then sort
        part = np.argpartition(B, -k, axis=1)[:, -k:]
        part_vals = np.take_along_axis(B, part, axis=1)
        order = np.argsort(-part_vals, axis=1)
        idx_sorted = np.take_along_axis(part, order, axis=1)
        vals_sorted = np.take_along_axis(part_vals, order, axis=1)
        all_idx[i0:i1] = idx_sorted
        all_sim[i0:i1] = vals_sorted

    if symmetric:
        # Mirror edges and keep top-k by similarity per row
        # Build list of edges for each row: existing neighbors + mirrored from others
        # To avoid huge memory, do it in chunks.
        idx_sym = np.empty_like(all_idx)
        sim_sym = np.empty_like(all_sim)
        deg = k
        # Build reverse adjacency lists via accumulation
        # For simplicity, we concatenate current and mirrored then reselect top-k.
        for i0 in range(0, N, block):
            i1 = min(N, i0 + block)
            # Gather candidates: existing neighbors
            idx_cand = [all_idx[i0:i1]]
            sim_cand = [all_sim[i0:i1]]
            # Mirrored: collect rows where i in neighbors of j
            # We find for each j block, which rows have i in their neighbor list
            # Efficient approach: scatter add is tricky; we'll do a pass over columns
            # Create maps for this slice
            for j0 in range(0, N, block):
                j1 = min(N, j0 + block)
                # neighbors of rows j0:j1
                neigh = all_idx[j0:j1]  # (b2, k)
                sims  = all_sim[j0:j1]  # (b2, k)
                # For each row i in [i0,i1), we want entries where neigh == i
                # We'll check membership by building a boolean mask per i range
                # This is O(N^2/k) worst-case; acceptable for moderate N and small k.
                # Compute a mask for indices in [i0,i1)
                mask = (neigh >= i0) & (neigh < i1)
                if not np.any(mask):
                    continue
                # extract for those entries
                rows_j, pos_k = np.where(mask)
                # Map neighbor index -> position in [i0,i1)
                neigh_flat = neigh[rows_j, pos_k]
                sims_flat  = sims[rows_j, pos_k]
                rel = neigh_flat - i0  # relative row index for idx_sym slice
                # For each (rel, j), we add candidate j with sim sims_flat
                # Aggregate per rel row
                # Build arrays to append; we'll do per row grouping
                # Start with empty lists per row
                bucket_idx = [[] for _ in range(i1 - i0)]
                bucket_sim = [[] for _ in range(i1 - i0)]
                for rr, jj, ss in zip(rel, rows_j + j0, sims_flat):
                    bucket_idx[rr].append(jj)
                    bucket_sim[rr].append(ss)
                # Append buckets
                extra_idx = np.array([np.array(v, dtype=np.int64) if len(v)>0 else np.empty(0, dtype=np.int64)
                                      for v in bucket_idx], dtype=object)
                extra_sim = np.array([np.array(v, dtype=np.float64) if len(v)>0 else np.empty(0, dtype=np.float64)
                                      for v in bucket_sim], dtype=object)
                # Merge with existing candidates
                base_idx = idx_cand[-1]
                base_sim = sim_cand[-1]
                merged_idx = []
                merged_sim = []
                for r in range(i1 - i0):
                    if extra_idx[r].size == 0:
                        merged_idx.append(base_idx[r:r+1][0])
                        merged_sim.append(base_sim[r:r+1][0])
                    else:
                        mi = np.concatenate([base_idx[r:r+1][0], extra_idx[r]])
                        ms = np.concatenate([base_sim[r:r+1][0], extra_sim[r]])
                        merged_idx.append(mi); merged_sim.append(ms)
                idx_cand[-1] = np.array(merged_idx, dtype=object)
                sim_cand[-1] = np.array(merged_sim, dtype=object)

            # Now select top-k from merged candidates per row
            final_idx_rows = []
            final_sim_rows = []
            for r in range(i1 - i0):
                mi = np.array(idx_cand[-1][r], dtype=np.int64)
                ms = np.array(sim_cand[-1][r], dtype=np.float64)
                # deduplicate by keeping max sim per neighbor
                if mi.size == 0:
                    final_idx_rows.append(np.full(k, -1, dtype=np.int64))
                    final_sim_rows.append(np.full(k, -np.inf, dtype=np.float64))
                    continue
                # Use argsort to get top-k
                order = np.argsort(-ms)
                mi = mi[order]; ms = ms[order]
                # unique while keeping first occurrence (highest sim)
                uniq_idx = []
                uniq_sim = []
                seen = set()
                for jj, ss in zip(mi, ms):
                    if jj not in seen and jj != (i0 + r):
                        uniq_idx.append(jj); uniq_sim.append(ss); seen.add(jj)
                    if len(uniq_idx) >= k:
                        break
                # pad if needed
                if len(uniq_idx) < k:
                    pad = k - len(uniq_idx)
                    uniq_idx += [-1] * pad
                    uniq_sim += [-np.inf] * pad
                final_idx_rows.append(np.array(uniq_idx[:k], dtype=np.int64))
                final_sim_rows.append(np.array(uniq_sim[:k], dtype=np.float64))
            idx_sym[i0:i1] = np.stack(final_idx_rows, axis=0)
            sim_sym[i0:i1] = np.stack(final_sim_rows, axis=0)
        all_idx, all_sim = idx_sym, sim_sym

    return all_idx, all_sim

def save_knn(indices: np.ndarray, sims: np.ndarray, path: str, store_angle: bool = False) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if store_angle:
        ang = np.degrees(np.arccos(_clip_cosine(sims)))
    if ext == ".npz":
        np.savez_compressed(path, indices=indices, sims=sims, angles_deg=(ang if store_angle else None))
    elif ext == ".csv":
        N, K = indices.shape
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            header = ["src", "dst", "cosine"] + (["angle_deg"] if store_angle else [])
            w.writerow(header)
            for i in range(N):
                for j in range(K):
                    dst = int(indices[i, j])
                    if dst < 0:
                        continue
                    row = [i, dst, float(sims[i, j])]
                    if store_angle:
                        row.append(float(ang[i, j]))
                    w.writerow(row)
    else:
        raise ValueError("knn-out must end with .npz or .csv")

# ------------------------------- CSV helpers ---------------------------------

def save_stats_csv(stats_list: Iterable['Stats'], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rows = [asdict(s) for s in stats_list]
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

# ------------------------------- Main CLI ------------------------------------

def build_points(method: str, num: int, dim: int, seed: int) -> np.ndarray:
    if method == "kronecker":
        return kronecker_sphere(num, dim, seed=seed)
    elif method == "halton":
        return halton_sphere(num, dim)
    elif method == "random":
        return random_sphere(num, dim, seed=seed)
    elif method == "fibonacci":
        if dim != 3:
            print("[warn] --method fibonacci is defined on S^2 (dim=3). Falling back to kronecker.", file=sys.stderr)
            return kronecker_sphere(num, dim, seed=seed)
        return fibonacci_sphere(num)
    elif method == "healpix":
        if dim != 3:
            raise ValueError("--method healpix is only defined for --dim 3 (S^2).")
        return healpix_sphere(num)
    elif method == "latlong":
        if dim == 3:
            return latlong_grid_s2(num)
        else:
            return latlong_random_nd(num, dim, seed=seed)
    else:
        raise ValueError(f"Unknown method: {method}")

def _parse_bench_Ns(spec: str) -> List[int]:
    spec = spec.strip()
    if ":" in spec:
        # format start:stop:step (inclusive stop)
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError("--bench-Ns must be 'start:stop:step' or comma-separated ints")
        a, b, s = map(int, parts)
        if s <= 0:
            raise ValueError("step in --bench-Ns must be positive")
        return list(range(a, b + 1, s))
    else:
        return [int(x.strip()) for x in spec.split(",") if x.strip()]

def main():
    parser = argparse.ArgumentParser(description="Quasi-uniform grids on S^(n-1) in R^n with traversal, snap-to-grid, statistics, k-NN, and benchmarking.")
    # Generation / I/O
    parser.add_argument("--dim", type=int, default=384, help="Ambient dimension n (sphere is S^(n-1)). Default: 384.")
    parser.add_argument("--num", type=int, default=4096, help="Number of points to generate (or ~ for HEALPix).")
    parser.add_argument("--method", type=str, default="kronecker",
                        choices=["kronecker", "halton", "random", "fibonacci", "healpix", "latlong"],
                        help="Point set construction. 'fibonacci' and 'healpix' only for dim=3. 'latlong' is naive.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for deterministic constructions where applicable.")
    parser.add_argument("--output", type=str, default="points.npy", help="Output file for generated points.")
    parser.add_argument("--format", type=str, default="npy", choices=["npy", "csv", "jsonl"], help="Output format.")
    parser.add_argument("--load", type=str, default=None, help="Load an existing grid (npy/csv/jsonl) instead of generating.")

    # Geometry / traversal
    parser.add_argument("--snap", type=str, default=None, help="Snap this vector (file path or comma-separated floats) to the nearest grid point.")
    parser.add_argument("--distance", type=str, default=None, help="Compute angular distance between two grid indices: i,j.")
    parser.add_argument("--midpoint", type=str, default=None, help="Compute midpoint (SLERP t=0.5) between two grid indices: i,j.")
    parser.add_argument("--midpoint-out", type=str, default=None, help="Where to save the midpoint vector (npy/csv/jsonl).")
    parser.add_argument("--traverse", type=str, default=None, help="Traverse between two grid indices i,j via SLERP polyline.")
    parser.add_argument("--steps", type=int, default=16, help="Number of segments for traversal polyline (steps+1 points).")
    parser.add_argument("--traverse-out", type=str, default=None, help="Where to save the traverse polyline (npy/csv/jsonl).")

    # Stats / plots / compare
    parser.add_argument("--stats", action="store_true", help="Compute statistics for the generated/loaded set.")
    parser.add_argument("--stats-out", type=str, default=None, help="Path to save stats JSON.")
    parser.add_argument("--stats-csv-out", type=str, default=None, help="Path to save stats CSV (one row per method).")
    parser.add_argument("--plots", type=str, default=None, help="If set, directory to save plots.")
    parser.add_argument("--compare", type=str, default="none", choices=["none", "latlong", "random"],
                        help="Build a comparison set with the same N and dim using this method.")
    parser.add_argument("--pairs", type=int, default=200000, help="Number of pairwise cosine samples for histograms.")
    parser.add_argument("--probe", type=int, default=256, help="Number of random probes for discrepancy and covering radius.")
    parser.add_argument("--nn-block", type=int, default=2048, help="Block size for nearest neighbor / matrix ops.")

    # k-NN graph
    parser.add_argument("--knn", type=int, default=0, help="If >0, build k-NN (cosine) graph with k neighbors per node.")
    parser.add_argument("--knn-out", type=str, default=None, help="Output path for k-NN graph (.npz or .csv).")
    parser.add_argument("--knn-symmetric", action="store_true", help="Symmetrize the k-NN graph (union, keep top-k per row).")
    parser.add_argument("--knn-block", type=int, default=None, help="Block size override for k-NN (defaults to --nn-block).")
    parser.add_argument("--knn-store-angle", action="store_true", help="Store angle degrees in k-NN output.")

    # Benchmark
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark sweep over N and methods; ignores most other actions.")
    parser.add_argument("--bench-methods", type=str, default="kronecker,halton,random", help="Comma list of methods to benchmark.")
    parser.add_argument("--bench-Ns", type=str, default="512,1024,2048", help="Comma list or 'start:stop:step' (inclusive stop).")
    parser.add_argument("--bench-repeats", type=int, default=1, help="How many repeats (different seeds) per (method,N).")
    parser.add_argument("--bench-out", type=str, default="bench.csv", help="CSV path for benchmark results.")
    parser.add_argument("--bench-plots", type=str, default=None, help="Directory to save benchmark plots (metric vs N).")

    args = parser.parse_args()

    # Benchmark mode
    if args.benchmark:
        methods = [m.strip() for m in args.bench_methods.split(",") if m.strip()]
        Ns = _parse_bench_Ns(args.bench_Ns)
        rows: List[Dict[str, Any]] = []
        for meth in methods:
            for N in Ns:
                for r in range(args.bench_repeats):
                    seed = args.seed + 1000*r + 17*hash(meth) % (1<<16)
                    pts = build_points(meth, N, args.dim, seed=seed)
                    stats, _ = compute_stats(pts, method=meth, nn_block=args.nn_block, probes=args.probe, seed=seed)
                    row = asdict(stats)
                    row["seed"] = seed
                    rows.append(row)
                    print(f"[bench] {meth} N={N} avgNN={row['avg_nn_deg']:.3f} cvNN={row['cv_nn']:.3f} coh={row['coherence']:.4f}")
        # Write CSV
        os.makedirs(os.path.dirname(args.bench_out) or ".", exist_ok=True)
        with open(args.bench_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"[bench] wrote CSV -> {args.bench_out}")
        # Plots
        if args.bench_plots and rows:
            import matplotlib
            matplotlib.use("Agg")
            metrics = ["avg_nn_deg", "cv_nn", "coherence", "msip_to_welch", "halfspace_discrepancy", "approx_covering_radius_deg"]
            plot_benchmark_lines(rows, metrics, args.bench_plots, f"dim={args.dim}")
        # Done
        return

    # Build or load points (non-benchmark mode)
    if args.load is not None:
        points = _load_points(args.load)
        dim = points.shape[1]
        print(f"[info] Loaded {points.shape[0]} points in dimension {dim} from {args.load}.")
    else:
        points = build_points(args.method, args.num, args.dim, args.seed)
        dim = args.dim
        _save(points, args.output, args.format)
        print(f"[info] Generated {points.shape[0]} points in dimension {dim} with method '{args.method}' -> {args.output} ({args.format}).")

    # Snap-to-grid
    if args.snap is not None:
        q = _parse_vector_arg(args.snap)
        if q.shape[0] != points.shape[1]:
            raise ValueError(f"Snap vector has dimension {q.shape[0]} but grid points are in dimension {points.shape[1]}.")
        idx, sim = snap_to_grid(points, q)
        ang = math.degrees(math.acos(_clip_cosine(np.array(sim))))
        print(f"[snap] index={idx} cosine={sim:.6f} angle_deg={ang:.6f}")

    # Distance
    if args.distance is not None:
        try:
            i_str, j_str = args.distance.split(",")
            i, j = int(i_str), int(j_str)
        except Exception as e:
            raise ValueError("--distance must be two integers i,j") from e
        u, v = points[i], points[j]
        dist = angular_distance(u, v)
        print(f"[dist] i={i} j={j} radians={dist:.9f} degrees={math.degrees(dist):.9f}")

    # Midpoint
    if args.midpoint is not None:
        try:
            i_str, j_str = args.midpoint.split(",")
            i, j = int(i_str), int(j_str)
        except Exception as e:
            raise ValueError("--midpoint must be two integers i,j") from e
        u, v = points[i], points[j]
        m = slerp(u, v, 0.5)
        if args.midpoint_out:
            _save(m[None, :], args.midpoint_out, args.format)
            print(f"[mid] saved midpoint of ({i},{j}) -> {args.midpoint_out}")
        else:
            print(f"[mid] midpoint({i},{j}) first 8 comps: {np.array2string(m[:8], precision=6, separator=', ')} ...")

    # Traverse
    if args.traverse is not None:
        try:
            i_str, j_str = args.traverse.split(",")
            i, j = int(i_str), int(j_str)
        except Exception as e:
            raise ValueError("--traverse must be two integers i,j") from e
        u, v = points[i], points[j]
        path = slerp_polyline(u, v, steps=args.steps)
        if args.traverse_out:
            _save(path, args.traverse_out, args.format)
            print(f"[path] saved SLERP polyline ({i}->{j}) with {args.steps+1} points -> {args.traverse_out}")
        else:
            print(f"[path] SLERP ({i}->{j}) produced {path.shape[0]} points; first point dot last: {float(np.dot(path[0], path[-1])):.6f}")

    # k-NN graph
    if args.knn > 0:
        kblock = args.knn_block if args.knn_block is not None else args.nn_block
        idx, sim = compute_knn_graph(points, k=args.knn, block=kblock, symmetric=args.knn_symmetric)
        if args.knn_out:
            save_knn(idx, sim, args.knn_out, store_angle=args.knn_store_angle)
            print(f"[knn] saved {args.knn}-NN graph -> {args.knn_out}")
        else:
            print(f"[knn] computed {args.knn}-NN graph (not saved). Example row 0 neighbors: {idx[0, :min(8, idx.shape[1])]}.")

    # Stats / comparison / plots
    if args.stats or args.plots or args.compare != "none":
        stats_main, extras_main = compute_stats(points, method=args.method, nn_block=args.nn_block,
                                                probes=args.probe, seed=args.seed)
        all_stats = [stats_main]
        cos_main = sample_pairwise_cos(points, num_pairs=args.pairs, seed=args.seed)

        if args.compare != "none":
            comp_method = "latlong" if args.compare == "latlong" else "random"
            comp_points = build_points(comp_method, points.shape[0], points.shape[1], seed=args.seed+1)
            stats_comp, extras_comp = compute_stats(comp_points, method=comp_method, nn_block=args.nn_block,
                                                    probes=args.probe, seed=args.seed+1)
            all_stats.append(stats_comp)
            cos_comp = sample_pairwise_cos(comp_points, num_pairs=args.pairs, seed=args.seed+1)
        else:
            extras_comp = {}
            cos_comp = None

        # Print stats summary
        for s in all_stats:
            print(f"[stats] {s.method}: N={s.num_points}, dim={s.dim}, "
                  f"avgNN={s.avg_nn_deg:.3f}deg, cvNN={s.cv_nn:.3f}, "
                  f"coh={s.coherence:.4f}, msip/welch={s.msip_to_welch:.3f}, "
                  f"halfspace_disc_max={s.halfspace_discrepancy:.4f}, "
                  f"covering≈{s.approx_covering_radius_deg:.3f}deg")

        # Save stats JSON or CSV
        if args.stats_out:
            payload = {"stats": [asdict(s) for s in all_stats]}
            with open(args.stats_out, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"[stats] wrote JSON -> {args.stats_out}")
        if args.stats_csv_out:
            save_stats_csv(all_stats, args.stats_csv_out)
            print(f"[stats] wrote CSV -> {args.stats_csv_out}")

        # Plots
        if args.plots:
            _ensure_dir(args.plots)
            import matplotlib
            matplotlib.use("Agg")
            plot_nn_hist(extras_main["nn_angles_deg"],
                         os.path.join(args.plots, f"nn_hist_{args.method}.png"),
                         f"Nearest-neighbor angles — {args.method}")
            if cos_main is not None and cos_main.size > 0:
                plot_cos_hist_with_uniform(cos_main, points.shape[1],
                                           os.path.join(args.plots, f"cos_hist_{args.method}.png"),
                                           f"Pairwise cosines — {args.method}")
            if args.compare != "none":
                if "nn_angles_deg" in extras_comp:
                    plot_nn_hist(extras_comp["nn_angles_deg"],
                                 os.path.join(args.plots, f"nn_hist_{comp_method}.png"),
                                 f"Nearest-neighbor angles — {comp_method}")
                if cos_comp is not None and cos_comp.size > 0:
                    plot_cos_hist_with_uniform(cos_comp, points.shape[1],
                                               os.path.join(args.plots, f"cos_hist_{comp_method}.png"),
                                               f"Pairwise cosines — {comp_method}")
                plot_metric_bars(all_stats,
                                 os.path.join(args.plots, "metrics.png"),
                                 "Evenness & clustering metrics")

if __name__ == "__main__":
    main()

