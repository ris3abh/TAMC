"""Persistent homology summaries for delay-embedded point clouds.

Backed by GUDHI's Vietoris-Rips implementation. Functions accept a point
cloud (n_points, n_dims) and return persistence diagrams or scalar/vector
summaries suitable for comparing topology across sliding windows.
"""

from __future__ import annotations

import numpy as np

try:
    import gudhi
except (
    ImportError
) as exc:  # pragma: no cover - dependency documented in requirements.txt
    raise ImportError(
        "gudhi is required for topology_metrics; install via requirements.txt"
    ) from exc


def vietoris_rips_persistence(
    point_cloud: np.ndarray,
    max_dimension: int = 1,
    max_edge_length: float | None = None,
) -> list[tuple[int, tuple[float, float]]]:
    """Compute persistence pairs (homology_dim, (birth, death)) via Vietoris-Rips.

    Restricted to H0 and H1 by default per the project's cost/accuracy tradeoff
    (see paper_notes/research_brief.md, Risk 4).
    """
    point_cloud = np.asarray(point_cloud, dtype=float)

    if point_cloud.ndim != 2:
        raise ValueError("point_cloud must be 2D: (n_points, n_features)")
    if point_cloud.shape[0] < 2:
        raise ValueError("point_cloud must contain at least two points")
    if max_dimension < 0:
        raise ValueError("max_dimension must be >= 0")

    edge_length = float("inf") if max_edge_length is None else float(max_edge_length)

    rips = gudhi.RipsComplex(
        points=point_cloud,
        max_edge_length=edge_length,
    )
    simplex_tree = rips.create_simplex_tree(max_dimension=max_dimension + 1)
    return simplex_tree.persistence()


def betti_numbers_at(
    persistence: list[tuple[int, tuple[float, float]]], threshold: float
) -> dict[int, int]:
    """Count features alive at a given filtration threshold, per homology dimension."""
    counts: dict[int, int] = {}
    for dim, (birth, death) in persistence:
        if birth <= threshold < death:
            counts[dim] = counts.get(dim, 0) + 1
    return counts


def betti_curve(
    persistence: list[tuple[int, tuple[float, float]]],
    dimension: int,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Evaluate the Betti curve for one homology dimension over a grid of thresholds."""
    pairs = [(b, d) for dim, (b, d) in persistence if dim == dimension]
    curve = np.zeros(len(thresholds))
    for i, t in enumerate(thresholds):
        curve[i] = sum(1 for b, d in pairs if b <= t < d)
    return curve


def persistence_lifetimes(
    persistence: list[tuple[int, tuple[float, float]]], dimension: int
) -> np.ndarray:
    """Extract finite lifetimes (death - birth) for one homology dimension."""
    lifetimes = [
        death - birth
        for dim, (birth, death) in persistence
        if dim == dimension and np.isfinite(death)
    ]
    return np.array(lifetimes)


def total_persistence(
    persistence: list[tuple[int, tuple[float, float]]],
    dimension: int,
    power: float = 1.0,
) -> float:
    """Sum of lifetime^power for one homology dimension."""
    lifetimes = persistence_lifetimes(persistence, dimension)
    if lifetimes.size == 0:
        return 0.0
    return float(np.sum(lifetimes**power))


def max_persistence(
    persistence: list[tuple[int, tuple[float, float]]], dimension: int
) -> float:
    """Longest lifetime for one homology dimension (0.0 if none)."""
    lifetimes = persistence_lifetimes(persistence, dimension)
    return float(lifetimes.max()) if lifetimes.size else 0.0


def persistence_entropy(
    persistence: list[tuple[int, tuple[float, float]]], dimension: int
) -> float:
    """Shannon entropy of normalized lifetimes for one homology dimension."""
    lifetimes = persistence_lifetimes(persistence, dimension)
    total = lifetimes.sum()
    if total <= 0:
        return 0.0
    probs = lifetimes / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def diagram_for_dimension(
    persistence: list[tuple[int, tuple[float, float]]], dimension: int
) -> np.ndarray:
    """Extract an (n_features, 2) birth-death array for one homology dimension.

    Infinite deaths (essential classes) are dropped, matching the convention
    expected by persim's bottleneck/Wasserstein distance functions.
    """
    pairs = [
        (birth, death)
        for dim, (birth, death) in persistence
        if dim == dimension and np.isfinite(death)
    ]
    return np.array(pairs) if pairs else np.empty((0, 2))


def bottleneck_distance(diagram_a: np.ndarray, diagram_b: np.ndarray) -> float:
    """Bottleneck distance between two persistence diagrams (same homology dim)."""
    return float(gudhi.bottleneck_distance(diagram_a, diagram_b))


def wasserstein_distance(
    diagram_a: np.ndarray, diagram_b: np.ndarray, order: int = 2
) -> float:
    """Wasserstein distance between two persistence diagrams via persim.

    `order` is accepted for API stability but persim's wasserstein() (0.3.x)
    only computes the order-2 (Euclidean ground metric) distance.
    """
    try:
        from persim import wasserstein
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "persim is required for wasserstein_distance; install via requirements.txt"
        ) from exc
    return float(wasserstein(diagram_a, diagram_b))


def topology_summary_vector(
    point_cloud: np.ndarray, max_dimension: int = 1
) -> dict[str, float]:
    """Compute a compact scalar summary (entropy/total/max persistence per dim).

    Intended as the fast per-window feature vector that tamic_signal.py
    compares across windows, instead of recomputing full diagrams downstream.
    """
    persistence = vietoris_rips_persistence(point_cloud, max_dimension=max_dimension)
    summary: dict[str, float] = {}
    for dim in range(max_dimension + 1):
        summary[f"h{dim}_entropy"] = persistence_entropy(persistence, dim)
        summary[f"h{dim}_total_persistence"] = total_persistence(persistence, dim)
        summary[f"h{dim}_max_persistence"] = max_persistence(persistence, dim)
    return summary
