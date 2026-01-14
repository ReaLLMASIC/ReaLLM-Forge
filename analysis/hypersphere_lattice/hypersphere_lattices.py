# Write a single, self-contained Python script that includes:
# - point set generation methods (rseq, halton, random, sfib, healpix) + latlong (S^2)
# - mapping to S^{n-1}, relaxation, distance/SLERP
# - KNN graph + Dijkstra path
# - compare mode with visualizations (matplotlib) and CSV/JSON summaries
# Everything you need—one file—to:
#   1) generate approximately even point sets on the hypersphere S^{n-1} in R^n,
#   2) compute angles, SLERP midpoints, build a simple KNN graph and traverse paths,
#   3) compare multiple construction methods with **visualizations** and **metrics**.

# Methods provided
# ----------------
# - rseq   : irrational rank‑1 lattice (Kronecker sequence) in [0,1]^d, mapped to the sphere via Box–Muller → normalize.
#            Uses alpha_j = frac(sqrt(prime_j)) for robust high-d behavior.
# - halton : Halton low-discrepancy sequence → sphere.
# - random : i.i.d. standard normals → normalize (baseline).
# - sfib   : spherical Fibonacci spiral (only for dim=3; S^2).
# - healpix: HEALPix pixel centers (only for dim=3; requires 'healpy').
# - latlong: (compare mode only) equal‑angular latitude/longitude grid (S^2) – **not equal-area**, for contrast.

# Visualizations (matplotlib, no seaborn; each chart = its own figure)
# --------------------------------------------------------------------
# - Pairwise angle histogram vs. theoretical density g(θ) ∝ sin^{n-2}(θ)
# - Nearest-neighbor angle histogram
# - Equal-area heatmap on S^2 (2D histogram in longitude λ and z=cosθ)

# Metrics (approximate; controllable sampling for speed)
# ------------------------------------------------------
# - separation_min_deg      : approximate minimum nearest-neighbor angle (↑ better)
# - covering_radius_est_deg : approximate covering radius via random directions (↓ better)
# - cap_discrepancy_sup     : spherical cap discrepancy (S^2 only) – sup norm over random caps (↓ better)
# - cap_discrepancy_mean    : mean absolute discrepancy over random caps (S^2 only) (↓ better)
# - riesz1_energy_sample    : sampled Riesz s=1 energy on chord distances (↓ better)
# - pairwise/global checks  : pair-angle histogram vs theory (visual)

# CLI quick examples
# ------------------
# # 1) Generate 10k points on S^383 (R^384) using a lattice sequence
# python hypersphere_all_in_one.py generate --dim 384 --n 10000 --method rseq --output s384_rseq_10k.npy

# # 2) On S^2, spherical Fibonacci (equal-area-ish, iso-latitude spiral)
# python hypersphere_all_in_one.py generate --dim 3 --n 20000 --method sfib --output s2_sfib_20k.npy

# # 3) HEALPix (requires healpy); N is implied by NSIDE (N=12*NSIDE^2)
# python hypersphere_all_in_one.py generate --dim 3 --method healpix --healpix-nside 64 --output s2_hp_n64.npy

# # 4) Angle & midpoint
# python hypersphere_all_in_one.py angle --points s384_rseq_10k.npy --i 42 --j 117
# python hypersphere_all_in_one.py midpoint --points s384_rseq_10k.npy --i 42 --j 117 --output mid.npy

# # 5) Simple KNN graph and approx path (O(N^2) demo version)
# python hypersphere_all_in_one.py knn --points s384_rseq_10k.npy --k 16 --graph knn.npz
# python hypersphere_all_in_one.py path --points s384_rseq_10k.npy --graph knn.npz --i 42 --j 117

# # 6) Compare methods and make plots
# #    For S^2, methods default to rseq halton random sfib latlong (healpix if --healpix-nside given)
# python hypersphere_all_in_one.py compare --dim 3 --n 6000 --plots-dir ./plots --summary-csv ./plots/s2_summary.csv
# #    For high‑d, defaults to rseq halton random
# python hypersphere_all_in_one.py compare --dim 384 --n 8000 --plots-dir ./plots_hd --summary-csv ./plots_hd/hd_summary.csv
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# Matplotlib is needed for visualization in 'compare' subcommand.
# We import lazily inside plotting functions to allow non-plot commands without it.



# =============================
# Utilities: geometry & mapping
# =============================

def unit_rows(X: np.ndarray) -> np.ndarray:
    """Normalize each row vector to unit length; X is (N, d)."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return X / norms


def angular_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Great-circle (geodesic) angle in radians between unit vectors x,y on S^{d-1}."""
    dot = float(np.dot(x, y))
    dot = max(-1.0, min(1.0, dot))
    return float(np.arccos(dot))


def slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between unit vectors a,b on S^{d-1} (returns unit)."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    dot = float(np.dot(a, b))
    dot = max(-1.0, min(1.0, dot))
    theta = math.acos(dot)
    if theta < 1e-9:
        return unit_rows((1.0 - t) * a[None, :] + t * b[None, :])[0]
    s = math.sin(theta)
    w1 = math.sin((1.0 - t) * theta) / s
    w2 = math.sin(t * theta) / s
    return unit_rows((w1 * a + w2 * b)[None, :])[0]


# =======================
# Low-discrepancy engines
# =======================

def primes_first_k(k: int) -> np.ndarray:
    """Return the first k primes (simple sieve + extension)."""
    if k <= 0:
        return np.array([], dtype=int)
    if k < 6:
        bound = 15
    else:
        kk = float(k)
        bound = int(kk * (math.log(kk) + math.log(math.log(kk))) + 10)
    sieve = np.ones(bound + 1, dtype=bool)
    sieve[:2] = False
    for p in range(2, int(math.sqrt(bound)) + 1):
        if sieve[p]:
            sieve[p*p:bound+1:p] = False
    primes = np.nonzero(sieve)[0]
    if len(primes) >= k:
        return primes[:k]
    # extend if our bound was not enough
    n = bound + 1
    while len(primes) < k:
        is_prime = True
        r = int(math.sqrt(n))
        for p in primes:
            if p > r:
                break
            if n % p == 0:
                is_prime = False
                break
        if is_prime:
            primes = np.append(primes, n)
        n += 1
    return primes[:k]


def radical_inverse(n: int, base: int) -> float:
    """Van der Corput radical inverse of integer n in given base -> [0,1)."""
    inv_base = 1.0 / base
    result = 0.0
    f = inv_base
    nn = n
    while nn > 0:
        result += (nn % base) * f
        nn //= base
        f *= inv_base
    return result


def halton_sequence(npts: int, dim: int, start: int = 1) -> np.ndarray:
    """Halton sequence in [0,1)^dim with first npts elements; index starts at 'start'."""
    primes = primes_first_k(dim)
    H = np.empty((npts, dim), dtype=float)
    for j in range(dim):
        base = int(primes[j])
        H[:, j] = [radical_inverse(i, base) for i in range(start, start + npts)]
    return H


def rseq_lattice(npts: int, dim: int, offset: Optional[np.ndarray] = None) -> np.ndarray:
    """Irrational rank‑1 lattice in [0,1)^dim using slopes alpha_j = frac(sqrt(prime_j))."""
    primes = primes_first_k(dim)
    alpha = np.sqrt(primes).astype(float)
    alpha = alpha - np.floor(alpha)  # fractional part in (0,1)
    if offset is None:
        offset = np.full(dim, 0.5, dtype=float)
    k = np.arange(npts, dtype=float)[:, None]  # (N,1)
    U = (offset + k * alpha[None, :]) % 1.0
    return U


# ==============
# Cube -> Sphere
# ==============

def _box_muller_pair(u1: np.ndarray, u2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Box–Muller transform from (0,1)^2 -> two independent N(0,1)."""
    eps = 1e-12
    u1c = np.clip(u1, eps, 1.0 - eps)
    u2c = np.clip(u2, eps, 1.0 - eps)
    r = np.sqrt(-2.0 * np.log(u1c))
    th = 2.0 * math.pi * u2c
    return r * np.cos(th), r * np.sin(th)


def uniforms_to_sphere(U: np.ndarray, out_dim: Optional[int] = None) -> np.ndarray:
    """Map U∈[0,1]^d to S^{out_dim-1} via Box–Muller → normalize. Supports odd out_dim."""
    N, Din = U.shape
    D = out_dim if out_dim is not None else Din
    need_cols = D if D % 2 == 0 else D + 1
    if Din < need_cols:
        reps = math.ceil(need_cols / Din)
        UU = np.tile(U, (1, reps))[:, :need_cols].copy()
        # tiny deterministic shift to avoid exact repeats after tiling
        shift = (np.arange(need_cols, dtype=float) * (math.sqrt(2) - 1.0)) % 1.0
        UU = (UU + shift[None, :]) % 1.0
    else:
        UU = U[:, :need_cols]
    Z = np.empty((N, need_cols), dtype=float)
    for j in range(0, need_cols, 2):
        z1, z2 = _box_muller_pair(UU[:, j], UU[:, j + 1])
        Z[:, j] = z1
        Z[:, j + 1] = z2
    if D < need_cols:
        Z = Z[:, :D]
    return unit_rows(Z)


# ==========================
# Special S^2 constructions
# ==========================

def spherical_fibonacci(npts: int) -> np.ndarray:
    """Approx equal-area iso-latitude spiral on S^2 (3D)."""
    i = np.arange(npts, dtype=float)
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    ga = 2.0 * math.pi * (1.0 - 1.0 / (phi * phi))  # golden-angle ~ 2π/φ^2
    z = 1.0 - (2.0 * (i + 0.5) / npts)
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = ga * i
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y, z], axis=1)


def healpix_points(nside: int) -> np.ndarray:
    """HEALPix pixel centers on S^2 via healpy (if installed)."""
    try:
        import healpy as hp  # type: ignore
    except Exception as e:
        raise RuntimeError("healpy is required for method=healpix. Try: pip install healpy") from e
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    st = np.sin(theta)
    x = st * np.cos(phi)
    y = st * np.sin(phi)
    z = np.cos(theta)
    return np.stack([x, y, z], axis=1)


def latlong_grid(n_target: int) -> np.ndarray:
    """Even angular spacing in latitude/longitude on S^2 (not equal-area)."""
    if n_target <= 2:
        return np.array([[0, 0, 1], [0, 0, -1]])[:n_target]
    # Choose ring count L and per-ring points M so that ~ 2 + L*M ≈ n_target
    L = max(1, int(math.sqrt(n_target / 2)))
    M = max(3, int(round((n_target - 2) / L)))
    theta = np.linspace(1.0/(L+1)*math.pi, L/(L+1)*math.pi, L)  # colatitude, avoid exact poles
    phis = 2 * math.pi * (np.arange(M, dtype=float) / M)
    pts = [np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])]
    for th in theta:
        st, ct = math.sin(th), math.cos(th)
        x = st * np.cos(phis)
        y = st * np.sin(phis)
        z = np.full_like(x, ct)
        ring = np.stack([x, y, z], axis=1)
        pts.append(ring)
    X = np.concatenate([p[None, :] if p.ndim == 1 else p for p in pts], axis=0)
    return X[:n_target]


# ==========================
# Generators / façade
# ==========================

def generate_points(n: int, dim: int, method: str,
                    healpix_nside: Optional[int] = None,
                    relax_steps: int = 0, relax_k: int = 16, relax_lr: float = 0.1,
                    seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if method == "random":
        Z = rng.standard_normal(size=(n, dim))
        X = unit_rows(Z)
    elif method == "rseq":
        U = rseq_lattice(n, max(dim, 2))
        X = uniforms_to_sphere(U, out_dim=dim)
    elif method == "halton":
        U = halton_sequence(n, max(dim, 2))
        X = uniforms_to_sphere(U, out_dim=dim)
    elif method == "sfib":
        if dim != 3:
            raise ValueError("--method sfib only valid for dim=3")
        X = spherical_fibonacci(n)
    elif method == "healpix":
        if dim != 3:
            raise ValueError("--method healpix only valid for dim=3")
        if healpix_nside is None:
            raise ValueError("--healpix-nside is required for method=healpix")
        X = healpix_points(healpix_nside)
        if n and X.shape[0] != n:
            print(f"[warn] Ignoring --n={n}; HEALPix with NSIDE={healpix_nside} yields N={X.shape[0]} points.", file=sys.stderr)
    else:
        raise ValueError(f"Unknown method: {method}")

    if relax_steps > 0:
        X = relax_repulsion(X, steps=relax_steps, k=relax_k, lr=relax_lr, seed=seed)
    return X


# =====================
# Repulsion relaxation
# =====================

def relax_repulsion(points: np.ndarray, steps: int = 5, k: int = 16, lr: float = 0.1, seed: Optional[int] = None) -> np.ndarray:
    """A few steps of local repulsive smoothing on S^{d-1} using k-NN per point.
    WARNING: naive neighbor search here is O(N^2); use with moderate N or replace with ANN.
    """
    N, d = points.shape
    X = points.copy()
    rng = np.random.default_rng(seed)

    max_bytes = 256 * 1024 * 1024  # block sizing guard
    B = max(1, min(N, int(max_bytes / (8 * max(1, N)))))

    for _ in range(steps):
        neighbors_idx = np.empty((N, k), dtype=int)
        # blockwise neighbor search by cosine (dot, since unit)
        for s in range(0, N, B):
            e = min(N, s + B)
            Dblk = X[s:e] @ X.T
            for row in range(e - s):
                Dblk[row, s + row] = -np.inf
            idx = np.argpartition(-Dblk, kth=min(k, N-1)-1, axis=1)[:, :k]
            vals = np.take_along_axis(Dblk, idx, axis=1)
            order = np.argsort(-vals, axis=1)
            idx = np.take_along_axis(idx, order, axis=1)
            neighbors_idx[s:e] = idx

        X_new = X.copy()
        for i in range(N):
            nbrs = neighbors_idx[i]
            Xi = X[i]
            diffs = Xi - X[nbrs]
            d2 = np.maximum(1e-12, np.sum(diffs * diffs, axis=1))
            forces = (diffs.T / (d2 ** 1.0)).T  # Riesz-1 like
            f = np.sum(forces, axis=0)
            f -= np.dot(f, Xi) * Xi  # project to tangent plane
            X_new[i] = Xi + lr * f
        X = unit_rows(X_new)
    return X


# ======================
# KNN graph + Dijkstra
# ======================

def build_knn_graph(points: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (indices, angles) where indices[i] lists k neighbor indices for node i,
    and angles[i] the corresponding angular distances (radians). O(N^2) demo implementation.
    """
    X = points
    N = X.shape[0]
    D = X @ X.T
    np.fill_diagonal(D, -np.inf)
    idx = np.argpartition(-D, kth=min(k, N-1)-1, axis=1)[:, :k]
    vals = np.take_along_axis(D, idx, axis=1)
    order = np.argsort(-vals, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    vals = np.take_along_axis(vals, order, axis=1)
    vals = np.clip(vals, -1.0, 1.0)
    ang = np.arccos(vals)
    return idx.astype(int), ang


def shortest_path_knn(points: np.ndarray, graph_idx: np.ndarray, graph_ang: np.ndarray, i: int, j: int) -> Tuple[float, List[int]]:
    """Dijkstra shortest path over a KNN graph using angular edge weights."""
    import heapq
    N = points.shape[0]
    INF = 1e30
    dist = np.full(N, INF, dtype=float)
    prev = np.full(N, -1, dtype=int)
    dist[i] = 0.0
    h = [(0.0, i)]
    visited = np.zeros(N, dtype=bool)
    while h:
        d, u = heapq.heappop(h)
        if visited[u]:
            continue
        visited[u] = True
        if u == j:
            break
        nbrs = graph_idx[u]
        dists = graph_ang[u]
        for v, w in zip(nbrs, dists):
            nd = d + float(w)
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(h, (nd, v))
    if not math.isfinite(dist[j]):
        return float('inf'), []
    path = []
    cur = j
    while cur != -1:
        path.append(cur)
        cur = int(prev[cur])
    path.reverse()
    return float(dist[j]), path


# =========================
# Metrics & visualizations
# =========================

def theoretical_angle_pdf_deg(theta_deg: np.ndarray, dim: int) -> np.ndarray:
    """Theoretical PDF of the angle between two random points on S^{dim-1}, in degrees^{-1}.
    g(θ) ∝ sin^{dim-2}(θ). Uses log-gamma to avoid overflow for high dimensions.
    """
    theta = np.radians(theta_deg)
    n = dim
    lg_c = math.lgamma(n/2.0) - math.lgamma((n-1)/2.0) - 0.5*math.log(math.pi)  # log c
    c = math.exp(lg_c)
    g = c * (np.sin(theta) ** (n - 2))
    return g * (math.pi / 180.0)  # convert to per-degree


def pair_angle_histogram(points: np.ndarray, n_pairs: int = 30000, bins: int = 90) -> Tuple[np.ndarray, np.ndarray]:
    """Empirical angle histogram (degrees) from a random sample of pairs; returns (centers, density)."""
    N = len(points)
    rng = np.random.default_rng(123)
    idx = rng.integers(0, N, size=(n_pairs, 2))
    dots = np.einsum("ij,ij->i", points[idx[:, 0]], points[idx[:, 1]])
    dots = np.clip(dots, -1.0, 1.0)
    ang = np.degrees(np.arccos(dots))
    hist, edges = np.histogram(ang, bins=bins, range=(0.0, 180.0), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist


def nearest_neighbor_angles(points: np.ndarray, anchors: int = 512, cand: int = 4096) -> np.ndarray:
    """Approx. nearest-neighbor angle distribution (degrees) for a subset of anchors vs. subset of candidates."""
    N = len(points)
    rng = np.random.default_rng(42)
    a_idx = rng.choice(N, size=min(anchors, N), replace=False)
    c_idx = rng.choice(N, size=min(cand, N), replace=False)
    Xa = points[a_idx]
    Xc = points[c_idx]
    D = Xa @ Xc.T
    for r, ai in enumerate(a_idx):
        loc = np.nonzero(c_idx == ai)[0]
        if len(loc) > 0:
            D[r, loc[0]] = -np.inf  # avoid self if sampled
    m = np.clip(D.max(axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(m))


def separation_min_deg(points: np.ndarray, anchors: int = 2000, cand: int = 4096) -> float:
    """Approximate minimum separation angle (degrees)."""
    return float(np.min(nearest_neighbor_angles(points, anchors=anchors, cand=cand)))


def covering_radius_est_deg(points: np.ndarray, probes: int = 500, cand: int = 4096) -> float:
    """Approximate covering radius by probing random directions (degrees)."""
    N, d = points.shape
    rng = np.random.default_rng(7)
    U = rng.standard_normal(size=(probes, d))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    c_idx = rng.choice(N, size=min(cand, N), replace=False)
    Xc = points[c_idx]
    D = U @ Xc.T
    m = np.clip(D.max(axis=1), -1.0, 1.0)
    ang = np.degrees(np.arccos(m))
    return float(np.max(ang))


def spherical_cap_discrepancy_S2(points: np.ndarray, n_caps: int = 200) -> Tuple[float, float]:
    """Monte Carlo spherical cap discrepancy on S^2; returns (sup, mean) over random caps."""
    assert points.shape[1] == 3, "cap discrepancy implemented for S^2 only"
    N = len(points)
    rng = np.random.default_rng(202)
    U = rng.standard_normal(size=(n_caps, 3))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    u = rng.random(n_caps)
    cos_alpha = 1.0 - 2.0 * u  # ensures cap area fractions are uniform in [0,1]
    dots = U @ points.T  # (n_caps, N)
    emp = (dots >= cos_alpha[:, None]).mean(axis=1)
    exact = (1.0 - cos_alpha) / 2.0
    err = np.abs(emp - exact)
    return float(np.max(err)), float(np.mean(err))


def riesz_energy_sample(points: np.ndarray, s: float = 1.0, n_pairs: int = 50000) -> float:
    """Sample-based Riesz s-energy on chord distance: E = E_{i!=j}[ 1/||x_i-x_j||^s ]."""
    N = len(points)
    rng = np.random.default_rng(55)
    i = rng.integers(0, N, size=n_pairs)
    j = rng.integers(0, N, size=n_pairs)
    mask = i != j
    i, j = i[mask], j[mask]
    diffs = points[i] - points[j]
    d = np.linalg.norm(diffs, axis=1)
    d = np.clip(d, 1e-12, None)
    return float(np.mean(1.0 / (d ** s)))


# ----------- Plotting (matplotlib; one chart per figure; default colors) -----------

def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except Exception as e:
        raise RuntimeError("matplotlib is required for 'compare' plotting. Try: pip install matplotlib") from e


def plot_pair_angle_vs_theory(points: np.ndarray, method: str, dim: int, out_path: str,
                              n_pairs: int = 30000, bins: int = 90) -> None:
    _require_matplotlib()
    import matplotlib.pyplot as plt
    centers, hist = pair_angle_histogram(points, n_pairs=n_pairs, bins=bins)
    pdf = theoretical_angle_pdf_deg(centers, dim)
    plt.figure()
    plt.plot(centers, hist, label="empirical")
    plt.plot(centers, pdf, label="theory")
    plt.xlabel("angle (degrees)")
    plt.ylabel("density")
    plt.title(f"Pairwise angle density: {method} (dim={dim})")
    plt.legend()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_nn_angle_hist(points: np.ndarray, method: str, dim: int, out_path: str,
                       anchors: int = 512, cand: int = 4096, bins: int = 60) -> None:
    _require_matplotlib()
    import matplotlib.pyplot as plt
    ang = nearest_neighbor_angles(points, anchors=anchors, cand=cand)
    plt.figure()
    plt.hist(ang, bins=bins, density=True)
    plt.xlabel("nearest-neighbor angle (degrees)")
    plt.ylabel("density")
    plt.title(f"Nearest-neighbor angle: {method} (dim={dim})")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_equal_area_heatmap_S2(points: np.ndarray, method: str, out_path: str,
                               bins_lon: int = 72, bins_z: int = 36) -> None:
    _require_matplotlib()
    import matplotlib.pyplot as plt
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    lon = np.degrees(np.arctan2(y, x))  # [-180, 180]
    zz = z                              # [-1, 1]
    H, xe, ye = np.histogram2d(lon, zz, bins=[bins_lon, bins_z], range=[[-180, 180], [-1.0, 1.0]], density=False)
    plt.figure()
    plt.imshow(H.T, origin="lower", aspect="auto", extent=[-180, 180, -1, 1])
    plt.xlabel("longitude (deg)")
    plt.ylabel("z = cos(theta)")
    plt.title(f"Equal-area density heatmap (S^2): {method}")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_metric_bars(rows: List[Dict[str, float]], metric_key: str, title: str, dim: int, out_path: str) -> None:
    _require_matplotlib()
    import matplotlib.pyplot as plt
    methods = [r["method"] for r in rows]
    vals = [float(r.get(metric_key, float("nan"))) for r in rows]
    plt.figure()
    plt.bar(methods, vals)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(metric_key.replace("_", " "))
    plt.title(f"{title} (dim={dim})")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# ======================
# I/O helpers
# ======================

def save_points(points: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True) if os.path.dirname(path) else None
    if path.endswith(".npy"):
        np.save(path, points)
    elif path.endswith(".csv"):
        np.savetxt(path, points, delimiter=",")
    elif path.endswith(".json"):
        with open(path, "w") as f:
            json.dump(points.tolist(), f)
    else:
        np.save(path, points)


def load_points(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path)
    elif path.endswith(".csv"):
        return np.loadtxt(path, delimiter=",")
    elif path.endswith(".json"):
        with open(path, "r") as f:
            return np.array(json.load(f), dtype=float)
    else:
        return np.load(path)


# ======================
# Compare mode (end-to-end)
# ======================

def compare_mode(dim: int, n: int, methods: List[str], plots_dir: str,
                 healpix_nside: Optional[int] = None,
                 n_pairs: int = 30000, bins: int = 90,
                 nn_anchors: int = 512, nn_cand: int = 4096,
                 cap_samples: int = 200, probes: int = 500,
                 write_csv: Optional[str] = None, write_json: Optional[str] = None) -> List[Dict[str, float]]:
    """Build multiple point sets, compute metrics, and write plots + summaries."""
    os.makedirs(plots_dir, exist_ok=True)

    sets: Dict[str, np.ndarray] = {}
    for m in methods:
        if m == "latlong":
            if dim != 3:
                print("[warn] Skipping 'latlong' for dim != 3.", file=sys.stderr)
                continue
            X = latlong_grid(n)
        elif m == "healpix":
            X = generate_points(n=0, dim=dim, method="healpix", healpix_nside=healpix_nside)
            if X.shape[0] > n and n > 0:
                X = X[:n]
        else:
            X = generate_points(n=n, dim=dim, method=m)
        sets[m] = X

    rows: List[Dict[str, float]] = []
    for name, X in sets.items():
        row: Dict[str, float] = {"method": name, "N": float(len(X))}
        # Core metrics
        row["separation_min_deg"] = separation_min_deg(X, anchors=min(nn_anchors, len(X)), cand=min(nn_cand, len(X)))
        row["covering_radius_est_deg"] = covering_radius_est_deg(X, probes=min(probes, 2*len(X)), cand=min(nn_cand, len(X)))
        row["riesz1_energy_sample"] = riesz_energy_sample(X, s=1.0, n_pairs=min(n_pairs, len(X)*10))

        # Cap discrepancy only for S^2
        if dim == 3:
            supd, meand = spherical_cap_discrepancy_S2(X, n_caps=min(cap_samples, 5*len(X)))
            row["cap_discrepancy_sup"] = supd
            row["cap_discrepancy_mean"] = meand

        # Plots
        pa = os.path.join(plots_dir, f"pair_angle_{name}_dim{dim}.png")
        plot_pair_angle_vs_theory(X, name, dim, pa, n_pairs=min(n_pairs, len(X)*50), bins=bins)
        row["pair_angle_plot"] = pa

        nn = os.path.join(plots_dir, f"nn_angle_{name}_dim{dim}.png")
        plot_nn_angle_hist(X, name, dim, nn, anchors=min(nn_anchors, len(X)), cand=min(nn_cand, len(X)), bins=bins//2)
        row["nn_angle_plot"] = nn

        if dim == 3:
            hm = os.path.join(plots_dir, f"heatmap_{name}_S2.png")
            plot_equal_area_heatmap_S2(X, name, hm, bins_lon=72, bins_z=36)
            row["heatmap_plot"] = hm

        rows.append(row)

    # Metric bar charts across methods
    try:
        plot_metric_bars(rows, "separation_min_deg", "Approx. minimum separation (higher is better)", dim,
                         os.path.join(plots_dir, f"bar_separation_min_deg_dim{dim}.png"))
        plot_metric_bars(rows, "covering_radius_est_deg", "Approx. covering radius (lower is better)", dim,
                         os.path.join(plots_dir, f"bar_covering_radius_est_deg_dim{dim}.png"))
        if dim == 3:
            plot_metric_bars(rows, "cap_discrepancy_sup", "Spherical cap discrepancy sup (lower is better)", dim,
                             os.path.join(plots_dir, f"bar_cap_discrepancy_sup_dim{dim}.png"))
    except Exception as e:
        print(f"[warn] Could not create bar charts: {e}", file=sys.stderr)

    # Summaries
    if write_csv:
        keys = sorted({k for r in rows for k in r.keys()})
        with open(write_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(keys)
            for r in rows:
                w.writerow([r.get(k, "") for k in keys])

    if write_json:
        with open(write_json, "w") as f:
            json.dump(rows, f, indent=2)

    return rows


# ==============
# CLI interface
# ==============

def main():
    ap = argparse.ArgumentParser(description="Hypersphere lattices, HEALPix (S^2), and comparison visualizations.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # generate
    ap_gen = sub.add_parser("generate", help="Generate points on S^{n-1}.")
    ap_gen.add_argument("--dim", type=int, required=True, help="Ambient dimension n (points lie on S^{n-1} in R^n).")
    ap_gen.add_argument("--n", type=int, default=0, help="Number of points to generate (ignored for HEALPix).")
    ap_gen.add_argument("--method", type=str, default="rseq",
                        choices=["rseq", "halton", "random", "sfib", "healpix"],
                        help="Construction method.")
    ap_gen.add_argument("--healpix-nside", type=int, default=None, help="NSIDE for HEALPix (dim=3 only).")
    ap_gen.add_argument("--relax-steps", type=int, default=0, help="Optional repulsion relaxation steps (0=off).")
    ap_gen.add_argument("--relax-k", type=int, default=16, help="Neighbors per point for relaxation.")
    ap_gen.add_argument("--relax-lr", type=float, default=0.1, help="Relaxation step size.")
    ap_gen.add_argument("--seed", type=int, default=None, help="Random seed for 'random' and relaxation.")
    ap_gen.add_argument("--output", type=str, required=True, help="Path to save points (.npy/.csv/.json).")

    # angle
    ap_ang = sub.add_parser("angle", help="Angular distance (radians/degrees) between two points by index.")
    ap_ang.add_argument("--points", type=str, required=True, help="Path to points file.")
    ap_ang.add_argument("--i", type=int, required=True)
    ap_ang.add_argument("--j", type=int, required=True)

    # midpoint
    ap_mid = sub.add_parser("midpoint", help="SLERP midpoint between two points by index.")
    ap_mid.add_argument("--points", type=str, required=True, help="Path to points file.")
    ap_mid.add_argument("--i", type=int, required=True)
    ap_mid.add_argument("--j", type=int, required=True)
    ap_mid.add_argument("--output", type=str, required=True, help="Where to save the 1xN midpoint (.npy/.csv/.json).")

    # knn
    ap_knn = sub.add_parser("knn", help="Build a small KNN graph (O(N^2), demo).")
    ap_knn.add_argument("--points", type=str, required=True)
    ap_knn.add_argument("--k", type=int, default=12)
    ap_knn.add_argument("--graph", type=str, required=True, help="Output .npz with indices and angles.")

    # path
    ap_path = sub.add_parser("path", help="Approx shortest path over KNN graph (edge=angular distance).")
    ap_path.add_argument("--points", type=str, required=True)
    ap_path.add_argument("--graph", type=str, required=True, help="Input .npz from 'knn'.")
    ap_path.add_argument("--i", type=int, required=True)
    ap_path.add_argument("--j", type=int, required=True)

    # compare
    ap_cmp = sub.add_parser("compare", help="Compare methods with metrics and plots.")
    ap_cmp.add_argument("--dim", type=int, required=True, help="Ambient dimension n.")
    ap_cmp.add_argument("--n", type=int, required=True, help="Number of points per method (except HEALPix).")
    ap_cmp.add_argument("--methods", type=str, nargs="*", default=None,
                        help="Subset of methods to compare. Defaults: dim=3 -> rseq halton random sfib latlong; else -> rseq halton random. Add 'healpix' if --healpix-nside provided.")
    ap_cmp.add_argument("--healpix-nside", type=int, default=None, help="If set (dim=3), include HEALPix with this NSIDE.")
    ap_cmp.add_argument("--plots-dir", type=str, required=True, help="Directory to save plots.")
    ap_cmp.add_argument("--summary-csv", type=str, default=None, help="Where to save a CSV summary (optional).")
    ap_cmp.add_argument("--summary-json", type=str, default=None, help="Where to save a JSON summary (optional).")
    ap_cmp.add_argument("--pairs", type=int, default=30000, help="Pair samples for angle histogram.")
    ap_cmp.add_argument("--bins", type=int, default=90, help="Histogram bins for pair-angle.")
    ap_cmp.add_argument("--nn-anchors", type=int, default=512, help="Anchors for NN angle estimates.")
    ap_cmp.add_argument("--nn-cand", type=int, default=4096, help="Candidates for NN angle estimates.")
    ap_cmp.add_argument("--caps", type=int, default=200, help="Number of caps for discrepancy (S^2 only).")
    ap_cmp.add_argument("--probes", type=int, default=500, help="Random directions for covering radius.")

    args = ap.parse_args()

    if args.cmd == "generate":
        dim = int(args.dim)
        if dim < 2:
            raise SystemExit("dim must be >= 2")
        if args.method == "healpix":
            if dim != 3:
                raise SystemExit("--method healpix only valid for dim=3")
            if args.healpix_nside is None:
                raise SystemExit("--healpix-nside required for method=healpix")
            X = generate_points(n=0, dim=dim, method="healpix", healpix_nside=args.healpix_nside,
                                relax_steps=args.relax_steps, relax_k=args.relax_k, relax_lr=args.relax_lr, seed=args.seed)
        else:
            if args.n <= 0:
                raise SystemExit("--n must be positive (except for healpix)")
            X = generate_points(n=args.n, dim=dim, method=args.method,
                                healpix_nside=args.healpix_nside,
                                relax_steps=args.relax_steps, relax_k=args.relax_k, relax_lr=args.relax_lr, seed=args.seed)
        save_points(X, args.output)
        print(json.dumps({"saved": args.output, "N": int(X.shape[0]), "dim": int(X.shape[1])}, indent=2))

    elif args.cmd == "angle":
        X = load_points(args.points)
        i, j = int(args.i), int(args.j)
        if not (0 <= i < len(X) and 0 <= j < len(X)):
            raise SystemExit("Indices out of range.")
        d = angular_distance(X[i], X[j])
        print(json.dumps({"i": i, "j": j, "angle_radians": d, "angle_degrees": float(np.degrees(d))}, indent=2))

    elif args.cmd == "midpoint":
        X = load_points(args.points)
        i, j = int(args.i), int(args.j)
        if not (0 <= i < len(X) and 0 <= j < len(X)):
            raise SystemExit("Indices out of range.")
        m = slerp(X[i], X[j], 0.5)              # m has shape (d,)
        mid = m[None, :]                        # (1, d) for saving/consistency
        save_points(mid, args.output)
        print(json.dumps(
            {"saved_midpoint": args.output, "dim": int(mid.shape[1])},  # or use m.size
            indent=2
        ))

    elif args.cmd == "knn":
        X = load_points(args.points)
        k = int(args.k)
        if k <= 0 or k >= len(X):
            raise SystemExit("k must be in [1, N-1]")
        idx, ang = build_knn_graph(X, k)
        os.makedirs(os.path.dirname(os.path.abspath(args.graph)), exist_ok=True) if os.path.dirname(args.graph) else None
        np.savez(args.graph, indices=idx, angles=ang)
        print(json.dumps({"saved_graph": args.graph, "k": k, "N": int(len(X))}, indent=2))

    elif args.cmd == "path":
        X = load_points(args.points)
        data = np.load(args.graph)
        idx = data["indices"]
        ang = data["angles"]
        i, j = int(args.i), int(args.j)
        if not (0 <= i < len(X) and 0 <= j < len(X)):
            raise SystemExit("Indices out of range.")
        dist, path = shortest_path_knn(X, idx, ang, i, j)
        out = {
            "i": i, "j": j,
            "approx_graph_geodesic_radians": float(dist),
            "approx_graph_geodesic_degrees": float(np.degrees(dist)),
            "path_indices": [int(u) for u in path],
            "great_circle_angle_radians": float(angular_distance(X[i], X[j])),
        }
        print(json.dumps(out, indent=2))

    elif args.cmd == "compare":
        dim = int(args.dim)
        if args.methods is None or len(args.methods) == 0:
            if dim == 3:
                methods = ["rseq", "halton", "random", "sfib", "latlong"]
                if args.healpix_nside is not None:
                    methods.append("healpix")
            else:
                methods = ["rseq", "halton", "random"]
        else:
            methods = list(args.methods)
            if "healpix" in methods and (dim != 3 or args.healpix_nside is None):
                raise SystemExit("To use 'healpix', set --dim 3 and --healpix-nside.")

        rows = compare_mode(
            dim=dim, n=int(args.n), methods=methods, plots_dir=args.plots_dir,
            healpix_nside=args.healpix_nside,
            n_pairs=int(args.pairs), bins=int(args.bins),
            nn_anchors=int(args.nn_anchors), nn_cand=int(args.nn_cand),
            cap_samples=int(args.caps), probes=int(args.probes),
            write_csv=args.summary_csv, write_json=args.summary_json
        )
        print(json.dumps({"methods": methods, "plots_dir": args.plots_dir, "summary_csv": args.summary_csv, "summary_json": args.summary_json, "rows": rows}, indent=2))

if __name__ == "__main__":
    main()
