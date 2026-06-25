"""RG-style statistical/distributional regime-similarity baseline.

Implements a transparent, from-scratch statistical/distributional
regime-similarity signal as a baseline against TAMC's topological drift.
This is conceptually the closest neighbor to RG-TTA (Regime-Guided
Test-Time Adaptation), which uses distributional regime similarity to
control TTA intensity -- no RG-TTA code is used or copied here; this
module computes regime similarity from first principles (moment-based
signatures, a two-sample KS statistic, Wasserstein distance, and a
variance ratio) and exposes it through the same
`history -> z(drift) -> sigmoid(z - threshold)` control law as
`drift_gates.py::ScalarDriftSignal`, so it can be compared against TAMC
under an identical gate mechanism (see methodology.md, Section 3,
"Fair Comparison Against Non-Topological Gates").
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class RegimeSignature:
    """A compact moment-based fingerprint of one window's distribution."""

    mean: float
    std: float
    skewness: float
    kurtosis: float
    lag1_autocorr: float


def compute_regime_signature(window: np.ndarray) -> RegimeSignature:
    """Compute a `RegimeSignature` for `window`.

    Robust to constant and very short windows: skewness/kurtosis are 0
    when std is 0 (a constant window has no shape to describe), and
    lag-1 autocorrelation is 0 when undefined (fewer than 2 points, or a
    zero-variance window). Never returns NaN/inf.
    """
    window = np.asarray(window, dtype=float)
    mean = float(window.mean())
    std = float(window.std())

    if std == 0.0:
        skewness = 0.0
        kurtosis = 0.0
    else:
        skewness = float(stats.skew(window))
        kurtosis = float(stats.kurtosis(window))
        if not np.isfinite(skewness):
            skewness = 0.0
        if not np.isfinite(kurtosis):
            kurtosis = 0.0

    if len(window) < 2 or std == 0.0:
        lag1_autocorr = 0.0
    else:
        centered = window - mean
        denom = float(np.sum(centered**2))
        lag1_autocorr = (
            float(np.sum(centered[:-1] * centered[1:]) / denom) if denom > 0 else 0.0
        )
        if not np.isfinite(lag1_autocorr):
            lag1_autocorr = 0.0

    return RegimeSignature(
        mean=mean,
        std=std,
        skewness=skewness,
        kurtosis=kurtosis,
        lag1_autocorr=lag1_autocorr,
    )


def signature_distance(a: RegimeSignature, b: RegimeSignature) -> float:
    """Normalized Euclidean distance between two five-dimensional signatures.

    Each component is scaled conservatively before differencing so no
    single feature (e.g. a large mean) dominates the distance.
    """
    mean_scale = abs(a.mean) + abs(b.mean) + 1e-8
    std_scale = abs(a.std) + abs(b.std) + 1e-8

    def _scaled_term(x: float, y: float, scale: float) -> float:
        return ((x - y) / scale) ** 2

    components = [
        _scaled_term(a.mean, b.mean, mean_scale),
        _scaled_term(a.std, b.std, std_scale),
        _scaled_term(a.skewness, b.skewness, 1.0 + abs(a.skewness) + abs(b.skewness)),
        _scaled_term(a.kurtosis, b.kurtosis, 1.0 + abs(a.kurtosis) + abs(b.kurtosis)),
        _scaled_term(
            a.lag1_autocorr,
            b.lag1_autocorr,
            1.0 + abs(a.lag1_autocorr) + abs(b.lag1_autocorr),
        ),
    ]
    return float(np.sqrt(sum(components)))


def feature_similarity(source: RegimeSignature, current: RegimeSignature) -> float:
    """Convert `signature_distance` to a [0, 1] similarity."""
    distance = signature_distance(source, current)
    similarity = 1.0 / (1.0 + distance)
    return float(np.clip(similarity, 0.0, 1.0))


def ks_similarity(source: np.ndarray, current: np.ndarray) -> float:
    """Two-sample KS-statistic-based similarity, clipped to [0, 1]."""
    source = np.asarray(source, dtype=float)
    current = np.asarray(current, dtype=float)
    if len(source) < 2 or len(current) < 2:
        return 1.0
    statistic = float(stats.ks_2samp(source, current).statistic)
    if not np.isfinite(statistic):
        return 1.0
    return float(np.clip(1.0 - statistic, 0.0, 1.0))


def wasserstein_similarity(source: np.ndarray, current: np.ndarray) -> float:
    """Wasserstein-distance-based similarity, clipped to [0, 1]."""
    source = np.asarray(source, dtype=float)
    current = np.asarray(current, dtype=float)
    distance = float(stats.wasserstein_distance(source, current))
    if not np.isfinite(distance):
        return 1.0
    similarity = 1.0 / (1.0 + distance)
    return float(np.clip(similarity, 0.0, 1.0))


def variance_ratio_similarity(source: np.ndarray, current: np.ndarray) -> float:
    """Symmetric variance-ratio similarity, robust to near-zero variance."""
    source_var = float(np.asarray(source, dtype=float).var())
    current_var = float(np.asarray(current, dtype=float).var())

    eps = 1e-12
    source_near_zero = source_var < eps
    current_near_zero = current_var < eps

    if source_near_zero and current_near_zero:
        return 1.0
    if source_near_zero or current_near_zero:
        return 0.0

    ratio = current_var / source_var
    similarity = min(ratio, 1.0 / ratio)
    return float(np.clip(similarity, 0.0, 1.0))


def regime_similarity(source: np.ndarray, current: np.ndarray) -> dict[str, float]:
    """Combine feature/KS/Wasserstein/variance-ratio similarity into one dict.

    `combined_similarity` is the mean of the four component similarities.
    """
    source = np.asarray(source, dtype=float)
    current = np.asarray(current, dtype=float)

    source_signature = compute_regime_signature(source)
    current_signature = compute_regime_signature(current)

    feat_sim = feature_similarity(source_signature, current_signature)
    ks_sim = ks_similarity(source, current)
    wass_sim = wasserstein_similarity(source, current)
    var_sim = variance_ratio_similarity(source, current)

    combined = float(np.mean([feat_sim, ks_sim, wass_sim, var_sim]))

    return {
        "feature_similarity": feat_sim,
        "ks_similarity": ks_sim,
        "wasserstein_similarity": wass_sim,
        "variance_ratio_similarity": var_sim,
        "combined_similarity": combined,
    }


@dataclass
class RegimeSimilaritySignal:
    """Online RG-style regime-similarity monitor, gated like `ScalarDriftSignal`.

    Holds a fixed source/reference window and, for each new causally
    observed window, computes combined regime similarity, converts it to
    drift (`1 - combined_similarity`), and z-scores that drift against its
    own running history using the same control law TAMC uses:
    `gate = sigmoid(z(drift) - threshold)`.

    Only the scalar drift history is retained, never the windows
    themselves.
    """

    source_window: np.ndarray
    history: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.source_window = np.asarray(self.source_window, dtype=float)

    def score_components(self, current_window: np.ndarray) -> dict[str, float]:
        """All four similarities, the combined similarity, and the drift."""
        components = regime_similarity(self.source_window, current_window)
        components["drift"] = 1.0 - components["combined_similarity"]
        return components

    def score(self, current_window: np.ndarray) -> float:
        """Compute and record the drift for one causally observed window."""
        drift = self.score_components(current_window)["drift"]
        self.history.append(drift)
        return drift

    def drift_zscore(self, drift: float, min_history: int = 8) -> float:
        """Z-score `drift` against the monitor's own running history."""
        if len(self.history) < min_history:
            return 0.0
        baseline = (
            np.array(self.history[:-1])
            if self.history[-1] == drift
            else np.array(self.history)
        )
        mean = baseline.mean()
        std = baseline.std()
        if std == 0.0:
            return 0.0
        return float((drift - mean) / std)

    def gate(
        self,
        current_window: np.ndarray,
        threshold: float = 2.0,
        min_history: int = 8,
    ) -> float:
        """Score `current_window`, record it, and return the [0, 1] gate.

        Calls `score()` internally -- do not also call `score()`
        separately for the same window before calling `gate()`, or the
        drift will be recorded twice in history.
        """
        drift = self.score(current_window)
        z = self.drift_zscore(drift, min_history=min_history)
        return float(1.0 / (1.0 + np.exp(-(z - threshold))))
