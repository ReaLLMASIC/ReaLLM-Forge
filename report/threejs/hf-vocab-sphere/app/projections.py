from __future__ import annotations

import inspect
import math
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, TSNE

_EPS = 1e-12
_TORCH_SMALL_OP_LOCK = threading.Lock()


PROJECTION_METHODS: dict[str, dict[str, Any]] = {
    "auto": {
        "label": "Auto / fidelity-aware",
        "family": "adaptive",
        "complexity": "Chooses by output geometry, token count, and anchor availability",
        "best_for": "A strong default without hand-tuning.",
        "caveat": "The method can change when the selection size changes.",
        "max_points": 3000,
        "stochastic": False,
    },
    "spherical_pca": {
        "label": "Spherical PCA",
        "family": "linear / global",
        "complexity": "O(n·d·3) randomized SVD",
        "best_for": "Fast, stable global structure and large selections.",
        "caveat": "PCA optimizes variance, not angular-distance fidelity. S² mode additionally discards each projected radius.",
        "max_points": 5000,
        "stochastic": False,
    },
    "tangent_pca": {
        "label": "Tangent-space PCA",
        "family": "spherical / local",
        "complexity": "O(n·d·3) after a spherical log map",
        "best_for": "Semantic neighborhoods concentrated around an anchor or mean direction.",
        "caveat": "The logarithmic map is local and becomes unstable near the antipode of its base point.",
        "max_points": 4000,
        "stochastic": False,
    },
    "cosine_kernel": {
        "label": "Cosine Gram eigenmap",
        "family": "spectral / global",
        "complexity": "O(n²·d + n³)",
        "best_for": "Small-to-medium sets where preserving pairwise dot products matters.",
        "caveat": "The top three eigen-directions cannot preserve a high-rank Gram matrix exactly.",
        "max_points": 1200,
        "stochastic": False,
    },
    "angular_mds": {
        "label": "Classical angular MDS",
        "family": "distance / global",
        "complexity": "O(n²·d + n³)",
        "best_for": "Small sets where pairwise angular distances are the primary object of study.",
        "caveat": "Classical MDS is Euclidean. It is natural in free R³; S² mode adds a separate radial-normalization step.",
        "max_points": 800,
        "stochastic": False,
    },
    "spherical_stress": {
        "label": "Direct spherical stress",
        "family": "constrained / angular",
        "complexity": "O(iterations·n²), with a spectral or PCA initialization",
        "best_for": "Small sets where the displayed great-circle angles should directly approximate original angles.",
        "caveat": "Iterative and non-convex. It is intrinsically S²-constrained, even when its coordinates are shown in free R³.",
        "max_points": 500,
        "stochastic": True,
    },
    "isomap": {
        "label": "Cosine Isomap",
        "family": "manifold / geodesic",
        "complexity": "Neighbor graph + all-pairs graph geodesics + eigensolve",
        "best_for": "Curved local manifolds connected by reliable neighborhood graphs.",
        "caveat": "Sensitive to neighbor count and disconnected or shortcut-prone graphs.",
        "max_points": 1500,
        "stochastic": False,
    },
    "tsne": {
        "label": "3-D cosine t-SNE",
        "family": "stochastic / local",
        "complexity": "Iterative; typically O(n²) for 3-D Barnes-Hut",
        "best_for": "Visual cluster separation and local neighborhoods.",
        "caveat": "Global distances, cluster sizes, and orientation are not directly interpretable; reruns can differ.",
        "max_points": 1800,
        "stochastic": True,
    },
    "umap": {
        "label": "3-D cosine UMAP",
        "family": "stochastic / local-global",
        "complexity": "Approximate neighbor graph + stochastic optimization",
        "best_for": "Large selections with useful local structure and faster nonlinear embedding.",
        "caveat": "Requires optional umap-learn; geometry depends on n_neighbors, min_dist, and seed.",
        "max_points": 5000,
        "stochastic": True,
    },
    "random": {
        "label": "Gaussian random baseline",
        "family": "linear / baseline",
        "complexity": "O(n·d·3)",
        "best_for": "A control condition for judging whether a sophisticated view adds real structure.",
        "caveat": "Three dimensions are far below Johnson–Lindenstrauss regimes for large sets, so distortion can be severe.",
        "max_points": 10000,
        "stochastic": True,
    },
}


@dataclass(slots=True)
class ProjectionOutput:
    coordinates: np.ndarray
    requested_method: str
    actual_method: str
    metrics: dict[str, Any]
    details: dict[str, Any]
    warnings: list[str]


def projection_catalog() -> list[dict[str, Any]]:
    umap_available = _umap_available()
    rows: list[dict[str, Any]] = []
    for key, info in PROJECTION_METHODS.items():
        row = {"key": key, **info, "available": key != "umap" or umap_available}
        if key == "umap" and not umap_available:
            row["availability_note"] = "Install the optional dependency: pip install umap-learn"
        rows.append(row)
    return rows


def _umap_available() -> bool:
    try:
        import umap  # noqa: F401
    except Exception:
        return False
    return True


def _unit_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, _EPS)


def _pad_three(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2:
        raise ValueError("Projection coordinates must be a 2-D matrix.")
    if coords.shape[1] >= 3:
        return coords[:, :3]
    return np.pad(coords, ((0, 0), (0, 3 - coords.shape[1])), mode="constant")


def _stable_axis_signs(coords: np.ndarray) -> np.ndarray:
    coords = coords.copy()
    for axis in range(coords.shape[1]):
        column = coords[:, axis]
        if not np.any(np.abs(column) > _EPS):
            continue
        pivot = int(np.argmax(np.abs(column)))
        if column[pivot] < 0:
            coords[:, axis] *= -1.0
    return coords


def _sphere_normalize(coords: np.ndarray, seed: int) -> tuple[np.ndarray, list[str]]:
    coords = _pad_three(coords)
    warnings: list[str] = []
    norms = np.linalg.norm(coords, axis=1)
    bad = norms <= 1e-10
    if np.any(bad):
        warnings.append(f"{int(np.sum(bad))} projected points landed at the origin and received deterministic micro-jitter.")
        rng = np.random.default_rng(seed)
        coords = coords.copy()
        coords[bad] = rng.normal(size=(int(np.sum(bad)), 3)) * 1e-6
    return _unit_rows(coords), warnings


def _rotation_from_to(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    a = source / max(float(np.linalg.norm(source)), _EPS)
    b = target / max(float(np.linalg.norm(target)), _EPS)
    cross = np.cross(a, b)
    s = float(np.linalg.norm(cross))
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if s < 1e-10:
        if c > 0:
            return np.eye(3)
        basis = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.8 else np.array([0.0, 0.0, 1.0])
        axis = np.cross(a, basis)
        axis /= max(float(np.linalg.norm(axis)), _EPS)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    vx = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    return np.eye(3) + vx + (vx @ vx) * ((1.0 - c) / (s * s))


def _orient_sphere(coords: np.ndarray, anchor_index: int | None) -> np.ndarray:
    """Apply a deterministic rigid orientation to spherical or free 3-D coordinates."""
    coords = _stable_axis_signs(coords)
    if anchor_index is None or not (0 <= anchor_index < len(coords)):
        return coords
    if float(np.linalg.norm(coords[anchor_index])) <= 1e-10:
        return coords

    rotation = _rotation_from_to(coords[anchor_index], np.array([0.0, 1.0, 0.0]))
    coords = coords @ rotation.T

    directions = _unit_rows(coords)
    dots = np.clip(directions @ directions[anchor_index], -1.0, 1.0)
    reference = int(np.argmin(dots))
    x, _, z = coords[reference]
    if abs(x) + abs(z) > 1e-10:
        angle = math.atan2(z, x)
        c, s = math.cos(angle), math.sin(angle)
        rotate_y = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        coords = coords @ rotate_y.T
    return coords


def _euclidean_display_scale(coords: np.ndarray, seed: int) -> tuple[np.ndarray, dict[str, Any], list[str]]:
    """Center and uniformly scale a free 3-D embedding without radial normalization.

    Translation and one global scale do not change the embedding's Euclidean geometry.
    The 95th-percentile radius is mapped to one display unit so outliers may remain
    outside the reference sphere instead of being collapsed onto it.
    """
    values = _pad_three(coords).astype(np.float64, copy=True)
    warnings: list[str] = []
    center = np.mean(values, axis=0, keepdims=True)
    values -= center
    radii = np.linalg.norm(values, axis=1)
    scale = float(np.percentile(radii, 95)) if len(radii) else 0.0
    if scale <= 1e-10:
        warnings.append("The free-space embedding was degenerate and received deterministic micro-jitter.")
        rng = np.random.default_rng(seed)
        values += rng.normal(size=values.shape) * 1e-6
        radii = np.linalg.norm(values, axis=1)
        scale = max(float(np.percentile(radii, 95)), 1e-6)
    values /= scale
    details = {
        "display_center": [float(x) for x in center.reshape(-1)],
        "display_scale_p95_radius": scale,
        "display_max_radius": float(np.max(np.linalg.norm(values, axis=1))) if len(values) else 0.0,
    }
    return values, details, warnings


def _fit_pca(values: np.ndarray, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    n_components = max(1, min(3, values.shape[0], values.shape[1]))
    solver = "randomized" if min(values.shape) > max(12, n_components + 2) else "full"
    pca = PCA(n_components=n_components, svd_solver=solver, random_state=seed)
    coords = pca.fit_transform(values)
    return _pad_three(coords), {
        "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "explained_variance_sum": float(np.sum(pca.explained_variance_ratio_)),
        "svd_solver": solver,
    }


def _spherical_pca(unit_vectors: np.ndarray, seed: int, center_mode: str, anchor_index: int | None) -> tuple[np.ndarray, dict[str, Any]]:
    if center_mode == "none":
        centered = unit_vectors
    elif center_mode == "anchor" and anchor_index is not None:
        centered = unit_vectors - unit_vectors[anchor_index]
    else:
        centered = unit_vectors - np.mean(unit_vectors, axis=0, keepdims=True)
    coords, details = _fit_pca(centered, seed)
    details["center_mode"] = center_mode
    return coords, details


def _tangent_pca(unit_vectors: np.ndarray, seed: int, anchor_index: int | None) -> tuple[np.ndarray, dict[str, Any]]:
    if anchor_index is not None:
        base = unit_vectors[anchor_index]
        base_kind = "anchor"
    else:
        base = np.mean(unit_vectors, axis=0)
        if np.linalg.norm(base) <= 1e-8:
            base = unit_vectors[0]
            base_kind = "first token (mean direction was degenerate)"
        else:
            base_kind = "extrinsic spherical mean"
        base = base / max(float(np.linalg.norm(base)), _EPS)

    cosines = np.clip(unit_vectors @ base, -1.0, 1.0)
    theta = np.arccos(cosines)
    orthogonal = unit_vectors - cosines[:, None] * base[None, :]
    sin_theta = np.sin(theta)
    scale = np.divide(theta, sin_theta, out=np.ones_like(theta), where=np.abs(sin_theta) > 1e-8)
    tangent = orthogonal * scale[:, None]
    coords, details = _fit_pca(tangent, seed)
    details.update(
        {
            "base_kind": base_kind,
            "max_log_map_angle_deg": float(np.degrees(np.max(theta))),
            "median_log_map_angle_deg": float(np.degrees(np.median(theta))),
        }
    )
    return coords, details


def _positive_eigen_embedding(
    matrix: np.ndarray, components: int = 3
) -> tuple[np.ndarray, list[float], np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    order = np.argsort(eigenvalues)[::-1]
    all_eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    positive = all_eigenvalues > 1e-10
    selected_values = all_eigenvalues[positive][:components]
    selected_vectors = eigenvectors[:, positive][:, :components]
    if len(selected_values) == 0:
        return np.zeros((matrix.shape[0], 3), dtype=np.float64), [], all_eigenvalues
    coords = selected_vectors * np.sqrt(selected_values)[None, :]
    return _pad_three(coords), [float(x) for x in selected_values], all_eigenvalues


def _cosine_kernel(unit_vectors: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    gram = np.clip(unit_vectors @ unit_vectors.T, -1.0, 1.0)
    coords, values, spectrum = _positive_eigen_embedding(gram)
    positive_total = float(np.sum(np.clip(spectrum, 0.0, None)))
    captured = float(sum(values) / positive_total) if positive_total > _EPS else 0.0
    return coords, {"top_eigenvalues": values, "positive_spectrum_fraction": captured}


def _angular_mds(unit_vectors: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    cosine = np.clip(unit_vectors @ unit_vectors.T, -1.0, 1.0)
    squared = np.arccos(cosine) ** 2
    row_mean = np.mean(squared, axis=1, keepdims=True)
    col_mean = np.mean(squared, axis=0, keepdims=True)
    grand_mean = float(np.mean(squared))
    gram = -0.5 * (squared - row_mean - col_mean + grand_mean)
    coords, values, spectrum = _positive_eigen_embedding(gram)
    negative_mass = float(np.abs(np.minimum(spectrum, 0.0)).sum())
    return coords, {"top_eigenvalues": values, "negative_eigenvalue_mass": negative_mass}


def _spherical_stress(
    unit_vectors: np.ndarray, seed: int, anchor_index: int | None
) -> tuple[np.ndarray, dict[str, Any]]:
    n = len(unit_vectors)
    if n <= 220:
        initial_raw, _ = _angular_mds(unit_vectors)
        initialization = "classical angular MDS"
    else:
        initial_raw, _ = _spherical_pca(unit_vectors, seed, "mean", anchor_index)
        initialization = "spherical PCA"
    initial, _ = _sphere_normalize(initial_raw, seed)

    pair_i, pair_j = np.triu_indices(n, k=1)
    target_cos = np.einsum("ij,ij->i", unit_vectors[pair_i], unit_vectors[pair_j])
    target_angles_np = np.arccos(np.clip(target_cos, -1.0, 1.0)).astype(np.float32)

    # Tiny pairwise tensor operations can become dramatically slower when a
    # large BLAS/OpenMP thread pool is launched for every optimizer step.  Keep
    # this constrained solve in a small, serialized thread section, then restore
    # the process-wide setting used by large matrix operations elsewhere.
    with _TORCH_SMALL_OP_LOCK:
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(min(previous_threads, 4))
        try:
            torch.manual_seed(int(seed))
            parameters = torch.tensor(initial, dtype=torch.float32, requires_grad=True)
            pair_i_t = torch.from_numpy(pair_i.astype(np.int64))
            pair_j_t = torch.from_numpy(pair_j.astype(np.int64))
            target_angles = torch.from_numpy(target_angles_np)
            target_energy = torch.sum(target_angles * target_angles).clamp_min(1e-8)
            optimizer = torch.optim.Adam([parameters], lr=0.045)
            steps = 240 if n <= 250 else 170
            best_loss = float("inf")
            best = initial.astype(np.float32, copy=True)
            stale = 0
            completed_steps = 0

            for step in range(steps):
                optimizer.zero_grad(set_to_none=True)
                unit = torch.nn.functional.normalize(parameters, dim=1, eps=1e-8)
                pair_cos = torch.sum(unit[pair_i_t] * unit[pair_j_t], dim=1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
                low_angles = torch.arccos(pair_cos)
                residual = low_angles - target_angles
                loss = torch.sum(residual * residual) / target_energy
                loss.backward()
                torch.nn.utils.clip_grad_norm_([parameters], max_norm=5.0)
                optimizer.step()
                with torch.no_grad():
                    parameters.copy_(torch.nn.functional.normalize(parameters, dim=1, eps=1e-8))
                completed_steps = step + 1
                value = float(loss.detach().item())
                if value + 1e-7 < best_loss:
                    best_loss = value
                    best = parameters.detach().cpu().numpy().copy()
                    stale = 0
                else:
                    stale += 1
                if step in {90, 150}:
                    for group in optimizer.param_groups:
                        group["lr"] *= 0.45
                if stale >= 55 and step >= 100:
                    break
        finally:
            torch.set_num_threads(previous_threads)

    initial_cos = np.einsum("ij,ij->i", initial[pair_i], initial[pair_j])
    initial_angles = np.arccos(np.clip(initial_cos, -1.0, 1.0))
    initial_stress = math.sqrt(
        float(np.sum((initial_angles - target_angles_np) ** 2))
        / max(float(np.sum(target_angles_np**2)), _EPS)
    )
    return best, {
        "initialization": initialization,
        "optimization_steps": completed_steps,
        "initial_stress_1": float(initial_stress),
        "optimized_stress_squared": float(best_loss),
    }


def _isomap(unit_vectors: np.ndarray, neighbors: int) -> tuple[np.ndarray, dict[str, Any]]:
    n_components = max(1, min(3, len(unit_vectors) - 1, unit_vectors.shape[1]))
    n_neighbors = max(2, min(int(neighbors), len(unit_vectors) - 1))
    model = Isomap(n_neighbors=n_neighbors, n_components=n_components, metric="cosine")
    coords = model.fit_transform(unit_vectors)
    reconstruction_error = None
    try:
        reconstruction_error = float(model.reconstruction_error())
    except Exception:
        pass
    return _pad_three(coords), {"n_neighbors": n_neighbors, "reconstruction_error": reconstruction_error}


def _tsne(unit_vectors: np.ndarray, seed: int, perplexity: float) -> tuple[np.ndarray, dict[str, Any]]:
    n = len(unit_vectors)
    effective_perplexity = max(2.0, min(float(perplexity), max(2.0, (n - 1) / 3.0)))
    kwargs: dict[str, Any] = {
        "n_components": 3,
        "perplexity": effective_perplexity,
        "metric": "cosine",
        "init": "random",
        "learning_rate": "auto",
        "method": "barnes_hut",
        "random_state": seed,
        "angle": 0.5,
    }
    if "max_iter" in inspect.signature(TSNE).parameters:
        kwargs["max_iter"] = 750
    else:  # pragma: no cover - compatibility with older sklearn
        kwargs["n_iter"] = 750
    model = TSNE(**kwargs)
    coords = model.fit_transform(unit_vectors)
    return coords, {"perplexity": effective_perplexity, "kl_divergence": float(model.kl_divergence_)}


def _umap(unit_vectors: np.ndarray, seed: int, neighbors: int, min_dist: float) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        import umap
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("UMAP is unavailable. Install the optional 'umap-learn' dependency.") from exc
    n_neighbors = max(2, min(int(neighbors), len(unit_vectors) - 1))
    reducer = umap.UMAP(
        n_components=3,
        metric="cosine",
        n_neighbors=n_neighbors,
        min_dist=float(min_dist),
        random_state=seed,
        transform_seed=seed,
    )
    coords = reducer.fit_transform(unit_vectors)
    return coords, {"n_neighbors": n_neighbors, "min_dist": float(min_dist)}


def _random_projection(unit_vectors: np.ndarray, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0.0, 1.0 / math.sqrt(3.0), size=(unit_vectors.shape[1], 3))
    return unit_vectors @ matrix, {"seed": seed, "distribution": "N(0, 1/3)"}


def _choose_auto_method(n: int, anchor_index: int | None, geometry_mode: str = "sphere") -> str:
    if geometry_mode == "euclidean":
        if n <= 800:
            return "angular_mds"
        if anchor_index is not None and n <= 2000:
            return "tangent_pca"
        return "spherical_pca"
    if n <= 140:
        return "spherical_stress"
    if anchor_index is not None and n <= 2000:
        return "tangent_pca"
    return "spherical_pca"


def _sample_pairs(n: int, max_pairs: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    total = n * (n - 1) // 2
    if total <= max_pairs:
        return np.triu_indices(n, k=1)
    rng = np.random.default_rng(seed)
    pairs: set[tuple[int, int]] = set()
    while len(pairs) < max_pairs:
        batch = min(max_pairs - len(pairs), 8192)
        left = rng.integers(0, n, size=batch * 2)
        right = rng.integers(0, n, size=batch * 2)
        for a, b in zip(left, right):
            if a == b:
                continue
            i, j = (int(a), int(b)) if a < b else (int(b), int(a))
            pairs.add((i, j))
            if len(pairs) >= max_pairs:
                break
    ordered = np.asarray(sorted(pairs), dtype=np.int64)
    return ordered[:, 0], ordered[:, 1]


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return None
    value = spearmanr(a, b, nan_policy="omit").statistic
    return None if not np.isfinite(value) else float(value)


def projection_metrics(
    high_unit: np.ndarray,
    low_coords: np.ndarray,
    *,
    seed: int,
    anchor_index: int | None,
    geometry_mode: str = "sphere",
    max_pairs: int = 80_000,
    knn_k: int = 10,
) -> dict[str, Any]:
    n = len(high_unit)
    mode = str(geometry_mode or "sphere").strip().casefold()
    if mode not in {"sphere", "euclidean"}:
        raise ValueError("geometry_mode must be 'sphere' or 'euclidean'.")
    if n < 2:
        return {
            "pair_sample_count": 0,
            "angular_spearman_rho": None,
            "stress_1": 0.0,
            "mean_abs_angle_error_deg": 0.0,
            "p95_abs_angle_error_deg": 0.0,
            "knn_recall_at_k": None,
            "knn_k": 0,
            "anchor_angle_spearman_rho": None,
            "metric_geometry": mode,
        }

    rows, cols = _sample_pairs(n, max_pairs, seed)
    high_cos = np.einsum("ij,ij->i", high_unit[rows], high_unit[cols])
    high_angles = np.arccos(np.clip(high_cos, -1.0, 1.0))

    low_values = np.asarray(low_coords, dtype=np.float64)
    low_distance_scale = 1.0
    if mode == "sphere":
        low_directions = _unit_rows(low_values)
        low_cos = np.einsum("ij,ij->i", low_directions[rows], low_directions[cols])
        low_raw_distances = np.arccos(np.clip(low_cos, -1.0, 1.0))
        low_equivalent_angles = low_raw_distances
    else:
        low_raw_distances = np.linalg.norm(low_values[rows] - low_values[cols], axis=1)
        denom_scale = float(np.dot(low_raw_distances, low_raw_distances))
        if denom_scale > _EPS:
            low_distance_scale = float(np.dot(high_angles, low_raw_distances) / denom_scale)
        low_equivalent_angles = low_raw_distances * low_distance_scale

    error_deg = np.degrees(np.abs(low_equivalent_angles - high_angles))
    denom = float(np.sum(high_angles * high_angles))
    stress = math.sqrt(float(np.sum((low_equivalent_angles - high_angles) ** 2)) / max(denom, _EPS))

    k = min(max(1, int(knn_k)), n - 1)
    rng = np.random.default_rng(seed + 911)
    if n <= 128:
        probe_rows = np.arange(n)
    else:
        probe_count = 96 if n <= 1000 else 48
        probe_rows = np.sort(rng.choice(n, size=min(probe_count, n), replace=False))
    high_probe_similarity = high_unit[probe_rows] @ high_unit.T
    if mode == "sphere":
        low_directions = _unit_rows(low_values)
        low_probe_similarity = low_directions[probe_rows] @ low_directions.T
    else:
        differences = low_values[probe_rows, None, :] - low_values[None, :, :]
        low_probe_similarity = -np.linalg.norm(differences, axis=2)
    recalls: list[float] = []
    for probe_index, row in enumerate(probe_rows):
        high_similarity = high_probe_similarity[probe_index].copy()
        low_similarity = low_probe_similarity[probe_index].copy()
        high_similarity[row] = -np.inf
        low_similarity[row] = -np.inf
        high_neighbors = set(np.argpartition(high_similarity, -k)[-k:].tolist())
        low_neighbors = set(np.argpartition(low_similarity, -k)[-k:].tolist())
        recalls.append(len(high_neighbors.intersection(low_neighbors)) / k)

    anchor_rho = None
    if anchor_index is not None and 0 <= anchor_index < n:
        mask = np.arange(n) != anchor_index
        high_anchor_angles = np.arccos(np.clip(high_unit @ high_unit[anchor_index], -1.0, 1.0))[mask]
        if mode == "sphere":
            directions = _unit_rows(low_values)
            low_anchor_distances = np.arccos(np.clip(directions @ directions[anchor_index], -1.0, 1.0))[mask]
        else:
            low_anchor_distances = np.linalg.norm(low_values - low_values[anchor_index], axis=1)[mask]
        anchor_rho = _safe_spearman(high_anchor_angles, low_anchor_distances)

    return {
        "pair_sample_count": int(len(high_angles)),
        "angular_spearman_rho": _safe_spearman(high_angles, low_raw_distances),
        "stress_1": float(stress),
        "mean_abs_angle_error_deg": float(np.mean(error_deg)),
        "p95_abs_angle_error_deg": float(np.percentile(error_deg, 95)),
        "knn_recall_at_k": float(np.mean(recalls)) if recalls else None,
        "knn_k": int(k),
        "anchor_angle_spearman_rho": anchor_rho,
        "metric_geometry": mode,
        "low_distance_scale_to_radians": float(low_distance_scale),
    }


def project_vectors(
    vectors: np.ndarray,
    *,
    method: str = "auto",
    seed: int = 42,
    anchor_index: int | None = None,
    center_mode: str = "mean",
    manifold_neighbors: int = 15,
    tsne_perplexity: float = 30.0,
    umap_min_dist: float = 0.1,
    align_anchor: bool = True,
    geometry_mode: str = "sphere",
) -> ProjectionOutput:
    started = time.perf_counter()
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("vectors must have shape [token_count, hidden_dim].")
    n, d = values.shape
    if n < 2:
        raise ValueError("Select at least two tokens to build a projection.")
    if d < 1:
        raise ValueError("The vector matrix has no hidden dimensions.")
    if not np.all(np.isfinite(values)):
        raise ValueError("The selected vectors contain NaN or infinite values.")

    mode = str(geometry_mode or "sphere").strip().casefold()
    if mode not in {"sphere", "euclidean"}:
        raise ValueError("geometry_mode must be 'sphere' or 'euclidean'.")

    requested = str(method or "auto").strip().casefold()
    if requested not in PROJECTION_METHODS:
        raise ValueError(f"Unknown projection method {requested!r}.")
    actual = _choose_auto_method(n, anchor_index, mode) if requested == "auto" else requested
    max_points = int(PROJECTION_METHODS[actual]["max_points"])
    if n > max_points:
        raise ValueError(f"{PROJECTION_METHODS[actual]['label']} supports at most {max_points:,} selected tokens in this app; got {n:,}.")
    if actual == "umap" and not _umap_available():
        raise RuntimeError("UMAP is unavailable. Install the optional 'umap-learn' dependency.")

    unit_vectors = _unit_rows(values)
    warnings: list[str] = []
    details: dict[str, Any]

    if actual == "spherical_pca":
        raw, details = _spherical_pca(unit_vectors, seed, center_mode, anchor_index)
    elif actual == "tangent_pca":
        raw, details = _tangent_pca(unit_vectors, seed, anchor_index)
        if details.get("max_log_map_angle_deg", 0.0) > 165.0:
            warnings.append("Some points are close to the tangent-map antipode; interpret their placement cautiously.")
    elif actual == "cosine_kernel":
        raw, details = _cosine_kernel(unit_vectors)
    elif actual == "angular_mds":
        raw, details = _angular_mds(unit_vectors)
    elif actual == "spherical_stress":
        raw, details = _spherical_stress(unit_vectors, seed, anchor_index)
    elif actual == "isomap":
        raw, details = _isomap(unit_vectors, manifold_neighbors)
    elif actual == "tsne":
        raw, details = _tsne(unit_vectors, seed, tsne_perplexity)
    elif actual == "umap":
        raw, details = _umap(unit_vectors, seed, manifold_neighbors, umap_min_dist)
    elif actual == "random":
        raw, details = _random_projection(unit_vectors, seed)
    else:  # pragma: no cover - guarded above
        raise ValueError(f"Unsupported method {actual!r}.")

    raw = _stable_axis_signs(_pad_three(raw))
    if mode == "euclidean" and actual == "spherical_stress":
        warnings.append(
            "Direct spherical stress is intrinsically sphere-constrained; choose angular MDS, PCA, Isomap, t-SNE, or UMAP for a genuinely unconstrained R³ embedding."
        )

    if mode == "sphere":
        coordinates, mapping_warnings = _sphere_normalize(raw, seed)
        warnings.extend(mapping_warnings)
        details["sphere_mapping"] = "radial normalization of the 3-D embedding"
        details["coordinate_geometry"] = "unit sphere S²"
    else:
        coordinates, scale_details, mapping_warnings = _euclidean_display_scale(raw, seed)
        details.update(scale_details)
        warnings.extend(mapping_warnings)
        details["sphere_mapping"] = "none"
        details["coordinate_geometry"] = "free Euclidean R³ with one global display scale"

    anchor_was_alignable = (
        anchor_index is not None
        and 0 <= anchor_index < len(coordinates)
        and float(np.linalg.norm(coordinates[anchor_index])) > 1e-10
    )
    if align_anchor and anchor_was_alignable:
        coordinates = _orient_sphere(coordinates, anchor_index)
    elif align_anchor and anchor_index is not None and not anchor_was_alignable:
        warnings.append("The anchor projected to the origin, so north-pole alignment was skipped.")

    metrics = projection_metrics(
        unit_vectors,
        coordinates,
        seed=seed,
        anchor_index=anchor_index,
        geometry_mode=mode,
    )
    metrics["runtime_ms"] = float((time.perf_counter() - started) * 1000.0)
    metrics["token_count"] = int(n)
    metrics["hidden_dim"] = int(d)
    metrics["radial_normalization"] = mode == "sphere"
    details["geometry_mode"] = mode
    details["anchor_aligned_to_north"] = bool(align_anchor and anchor_was_alignable)

    return ProjectionOutput(
        coordinates=coordinates.astype(np.float32),
        requested_method=requested,
        actual_method=actual,
        metrics=metrics,
        details=details,
        warnings=warnings,
    )


def nearest_neighbor_edges(unit_vectors: np.ndarray, *, k: int = 2, max_edges: int = 4000) -> list[dict[str, Any]]:
    raw = np.asarray(unit_vectors, dtype=np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    values = raw / np.maximum(norms, 1e-12)
    n = len(values)
    if n < 2 or k <= 0:
        return []
    k = min(int(k), n - 1)
    edges: dict[tuple[int, int], float] = {}
    block_size = 768
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        similarities = values[start:end] @ values.T
        local_rows = np.arange(end - start)
        similarities[local_rows, np.arange(start, end)] = -np.inf
        candidates = np.argpartition(similarities, -k, axis=1)[:, -k:]
        for local_source, source in enumerate(range(start, end)):
            for target in candidates[local_source]:
                a, b = (source, int(target)) if source < target else (int(target), source)
                cosine = float(np.clip(similarities[local_source, target], -1.0, 1.0))
                previous = edges.get((a, b))
                if previous is None or cosine > previous:
                    edges[(a, b)] = cosine
    ordered = sorted(edges.items(), key=lambda item: (-item[1], item[0]))[: int(max_edges)]
    return [
        {
            "source_index": int(pair[0]),
            "target_index": int(pair[1]),
            "cosine_similarity": float(cosine),
            "angle_deg": float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))),
        }
        for pair, cosine in ordered
    ]
