"""TAMC-Lite end-to-end pipeline: frozen forecast + topology-gated residual.

Wires a frozen forecaster, a TamicSignal topological drift monitor, and a
residual adapter into a single forward-only prediction step:

    observed context -> frozen forecast
                      -> topological drift score (TamicSignal.score_window)
                      -> topology-derived gate (TamicSignal.gate)
                      -> residual correction
                      -> adapted forecast = base_forecast + gate * correction

The drift score only ever multiplies the residual correction; it is never
added as a term in an optimization objective (research brief section 6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from tamic_signal import TamicSignal


class Forecaster(Protocol):
    def predict(self, context: np.ndarray) -> np.ndarray: ...


class Residual(Protocol):
    def correction(self, context: np.ndarray) -> np.ndarray: ...


@dataclass
class TamicLitePipeline:
    """Reusable TAMC-Lite prediction step around a frozen forecaster."""

    forecaster: Forecaster
    tamic_signal: TamicSignal
    residual_adapter: Residual
    context_length: int
    topology_window: int
    horizon: int
    gate_threshold: float = 2.0
    min_history: int = 8

    def predict(self, context: np.ndarray, topology_window_values: np.ndarray) -> dict:
        """Run one forward-only TAMC-Lite prediction step.

        `context` (length `context_length`) feeds the frozen forecaster and
        the residual adapter; `topology_window_values` (length
        `topology_window`) feeds the topological drift monitor. They are
        kept separate because the forecaster's context and the topology
        monitor's window need not be the same length.
        """
        base_forecast = np.asarray(self.forecaster.predict(context), dtype=float)

        score = self.tamic_signal.score_window(topology_window_values)
        gate = self.tamic_signal.gate(
            score.distance,
            threshold=self.gate_threshold,
            min_history=self.min_history,
        )

        correction = np.asarray(self.residual_adapter.correction(context), dtype=float)
        adapted_forecast = base_forecast + gate * correction

        return {
            "base_forecast": base_forecast,
            "adapted_forecast": adapted_forecast,
            "gate": gate,
            "topological_distance": score.distance,
            "prototype_label": score.prototype_label,
        }


@dataclass
class TamicBlendPipeline:
    """Topology-gated blend of a frozen forecaster and an adaptive forecaster.

        forecast = (1 - gate) * frozen_forecast + gate * adaptive_forecast

    Unlike TamicLitePipeline (which gates an additive residual), this gates
    a blend between two full forecasts: a source-trained frozen forecaster
    and a forward-only adaptive forecaster that only ever sees the current
    context (no labels, no future data, no gradient updates). `gate` still
    comes from TamicSignal.gate(...) and only ever multiplies/blends; it is
    never added as a term in an optimization objective.
    """

    frozen_forecaster: Forecaster
    adaptive_forecaster: Forecaster
    tamic_signal: TamicSignal
    context_length: int
    topology_window: int
    horizon: int
    gate_threshold: float = 2.0
    min_history: int = 8

    def predict(self, context: np.ndarray, topology_window_values: np.ndarray) -> dict:
        """Run one forward-only TAMC-Blend prediction step.

        `context` (length `context_length`) feeds both forecasters;
        `topology_window_values` (length `topology_window`) feeds the
        topological drift monitor.
        """
        frozen_forecast = np.asarray(
            self.frozen_forecaster.predict(context), dtype=float
        )
        adaptive_forecast = np.asarray(
            self.adaptive_forecaster.predict(context), dtype=float
        )

        score = self.tamic_signal.score_window(topology_window_values)
        gate = self.tamic_signal.gate(
            score.distance,
            threshold=self.gate_threshold,
            min_history=self.min_history,
        )

        blended_forecast = (1.0 - gate) * frozen_forecast + gate * adaptive_forecast

        return {
            "frozen_forecast": frozen_forecast,
            "adaptive_forecast": adaptive_forecast,
            "blended_forecast": blended_forecast,
            "gate": gate,
            "topological_distance": score.distance,
            "prototype_label": score.prototype_label,
        }
