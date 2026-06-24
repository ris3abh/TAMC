"""Topological drift signal construction for TAMC meta-control.

Builds source-regime topology prototypes, scores incoming windows against
them, and exposes the drift score as a gate / control signal for adapters
in adapters.py. See paper_notes/research_biref.md sections 6-7 for the
control-signal design rationale (drift must not be used as a constant term
in a candidate-dependent optimization objective).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from delay_embedding import EmbeddingConfig, sliding_attractor_point_clouds
from topology_metrics import (
    diagram_for_dimension,
    vietoris_rips_persistence,
    wasserstein_distance,
)


@dataclass
class SourcePrototype:
    """A single reference topology computed from a source-regime window."""

    diagrams: dict[int, np.ndarray]
    label: str = "source"


@dataclass
class DriftScore:
    """Per-step topological drift relative to the closest source prototype."""

    distance: float
    prototype_label: str


@dataclass
class TamicSignal:
    """Online topological drift monitor.

    Holds one or more source prototypes (built once, offline, from
    known-stationary data) and scores each incoming window's persistence
    diagram against the closest prototype via Wasserstein distance.
    """

    config: EmbeddingConfig
    max_dimension: int = 1
    drift_dimension: int = 1
    prototypes: list[SourcePrototype] = field(default_factory=list)
    history: list[float] = field(default_factory=list)

    def add_source_prototype(self, series: np.ndarray, label: str = "source") -> None:
        """Build and store a source-regime topology prototype from a reference window."""
        if series.shape[0] != self.config.window:
            raise ValueError(
                f"reference series length {series.shape[0]} must equal "
                f"config.window {self.config.window}"
            )
        point_cloud = next(
            iter(
                sliding_attractor_point_clouds(
                    series, self.config, stride=self.config.window
                )
            )
        )
        persistence = vietoris_rips_persistence(
            point_cloud, max_dimension=self.max_dimension
        )
        diagrams = {
            dim: diagram_for_dimension(persistence, dim)
            for dim in range(self.max_dimension + 1)
        }
        self.prototypes.append(SourcePrototype(diagrams=diagrams, label=label))

    def score_window(self, window: np.ndarray) -> DriftScore:
        """Score one raw-signal window against the closest stored source prototype."""
        if not self.prototypes:
            raise RuntimeError(
                "no source prototypes registered; call add_source_prototype first"
            )
        if window.shape[0] != self.config.window:
            raise ValueError(
                f"window length {window.shape[0]} must equal config.window {self.config.window}"
            )

        point_cloud = next(
            iter(
                sliding_attractor_point_clouds(
                    window, self.config, stride=self.config.window
                )
            )
        )
        persistence = vietoris_rips_persistence(
            point_cloud, max_dimension=self.max_dimension
        )
        diagram = diagram_for_dimension(persistence, self.drift_dimension)

        best_distance = float("inf")
        best_label = self.prototypes[0].label
        for prototype in self.prototypes:
            distance = wasserstein_distance(
                diagram, prototype.diagrams[self.drift_dimension]
            )
            if distance < best_distance:
                best_distance = distance
                best_label = prototype.label

        self.history.append(best_distance)
        return DriftScore(distance=best_distance, prototype_label=best_label)

    def drift_zscore(self, distance: float, min_history: int = 8) -> float:
        """Z-score a drift distance against the monitor's own running history.

        Returns 0.0 until enough history has accumulated, matching the
        MAD/z-score thresholding scheme described in the research brief.
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
        """Smooth [0, 1] gate: ramps from 0 to 1 as drift z-score crosses threshold.

        Intended as `g_t` in TAMC-Lite (research brief section 7.1): multiplies
        a residual correction so adaptation only engages once drift is
        meaningfully above the monitor's own historical baseline.
        """
        z = self.drift_zscore(distance, min_history=min_history)
        return float(1.0 / (1.0 + np.exp(-(z - threshold))))

    def update_rate(
        self,
        distance: float,
        eta_min: float = 0.0,
        eta_max: float = 1.0,
        min_history: int = 8,
    ) -> float:
        """Topology-controlled adapter update rate (TAMC-Rate, research brief 7.2)."""
        z = self.drift_zscore(distance, min_history=min_history)
        sigmoid = 1.0 / (1.0 + np.exp(-z))
        return float(eta_min + (eta_max - eta_min) * sigmoid)
