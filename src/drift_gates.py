"""Generic z-scored sigmoid gate for non-topological drift scores.

Mirrors TamicSignal's drift_zscore/gate control law (tamic_signal.py) but
decoupled from persistent homology, so any scalar drift score (mean/
variance distance, autocorrelation distance, spectral distance, ...) can
be turned into a [0, 1] gate using the same rule TAMC uses for topological
drift. This lets non-topological baselines be compared fairly against
TAMC's gate under an identical control law.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ScalarDriftSignal:
    """Online monitor that z-scores a scalar drift score against its own history.

    Causal by construction: `gate` at step `t` only ever depends on scores
    observed at or before `t` (via `history`), never on future values.
    """

    history: list[float] = field(default_factory=list)

    def score(self, distance: float) -> None:
        """Record one new causally-observed drift score."""
        self.history.append(distance)

    def drift_zscore(self, distance: float, min_history: int = 8) -> float:
        """Z-score `distance` against the monitor's own running history.

        Returns 0.0 until enough history has accumulated, matching
        TamicSignal.drift_zscore.
        """
        if len(self.history) < min_history:
            return 0.0
        baseline = (
            np.array(self.history[:-1])
            if self.history[-1] == distance
            else np.array(self.history)
        )
        mean = baseline.mean()
        std = baseline.std()
        if std == 0.0:
            return 0.0
        return float((distance - mean) / std)

    def gate(
        self, distance: float, threshold: float = 2.0, min_history: int = 8
    ) -> float:
        """Smooth [0, 1] gate: ramps from 0 to 1 as the drift z-score crosses threshold."""
        z = self.drift_zscore(distance, min_history=min_history)
        return float(1.0 / (1.0 + np.exp(-(z - threshold))))
