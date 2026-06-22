"""Forward-only adapters for frozen time-series forecasters.

Implements TAMC-Lite (topology-gated residual output correction, research
brief section 7.1) plus a generic interface for zeroth-order/forward-only
parameter search (CMA-ES/SPSA-compatible) over small adapter parameter
vectors, per research brief sections 6 and 7.4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np


class FrozenForecaster(Protocol):
    """Interface a frozen forecasting model must satisfy to be wrapped by an adapter."""

    def __call__(self, context: np.ndarray) -> np.ndarray:
        """Map a context window to a forecast; weights are never updated here."""
        ...


@dataclass
class FrozenForecasterWrapper:
    """Read-only wrapper around a frozen forecaster, used so adapters never touch weights."""

    model: FrozenForecaster

    def predict(self, context: np.ndarray) -> np.ndarray:
        return np.asarray(self.model(context))


@dataclass
class ResidualAdapter:
    """Lightweight learned residual correction r_phi(context) added to a frozen forecast.

    `params` holds phi: a small parameter vector (e.g. linear weights) the
    forward-only optimizer in adapters.py's ForwardOnlyOptimizer is allowed
    to search over. The correction function is supplied by the caller so
    this class stays agnostic to whether phi parameterizes a bias, a linear
    map, or something else.
    """

    correction_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
    params: np.ndarray

    def correction(self, context: np.ndarray) -> np.ndarray:
        return self.correction_fn(context, self.params)


@dataclass
class TamicLiteAdapter:
    """Topology-gated output adapter: y_hat_TAMC = y_hat + gate * residual_correction.

    `gate` is expected to come from TamicSignal.gate(...) in tamic_signal.py,
    i.e. a topological-drift-derived scalar in [0, 1]. This keeps the
    topological term acting as a control multiplier rather than an additive
    constant in any downstream optimization objective (research brief
    section 6: "Methodological Warning").
    """

    forecaster: FrozenForecasterWrapper
    residual: ResidualAdapter

    def predict(self, context: np.ndarray, gate: float) -> np.ndarray:
        base_forecast = self.forecaster.predict(context)
        correction = self.residual.correction(context)
        return base_forecast + gate * correction


@dataclass
class TamicRateAdapter:
    """Output adapter whose residual update rate is controlled by topological drift.

    Unlike TamicLiteAdapter (which gates the *correction magnitude* at
    prediction time), this controls how fast `residual.params` itself moves
    toward a target update each step (TAMC-Rate, research brief 7.2).
    """

    forecaster: FrozenForecasterWrapper
    residual: ResidualAdapter

    def predict(self, context: np.ndarray) -> np.ndarray:
        return self.forecaster.predict(context) + self.residual.correction(context)

    def apply_update(self, update: np.ndarray, eta: float) -> None:
        """Move residual.params toward `update` at topology-controlled rate eta."""
        self.residual.params = (1.0 - eta) * self.residual.params + eta * update


@dataclass
class PrototypeAdapter:
    """Selects among per-regime adapters by nearest topological source prototype.

    TAMC-Prototype (research brief 7.3): each registered prototype label maps
    to its own ResidualAdapter, calibration bias, or similar; the caller looks
    up the closest prototype label via TamicSignal.score_window(...) and
    routes through the matching adapter here.
    """

    forecaster: FrozenForecasterWrapper
    adapters_by_label: dict[str, ResidualAdapter] = field(default_factory=dict)

    def register(self, label: str, residual: ResidualAdapter) -> None:
        self.adapters_by_label[label] = residual

    def predict(self, context: np.ndarray, prototype_label: str) -> np.ndarray:
        base_forecast = self.forecaster.predict(context)
        residual = self.adapters_by_label.get(prototype_label)
        if residual is None:
            return base_forecast
        return base_forecast + residual.correction(context)


@dataclass
class ForwardOnlyOptimizer:
    """Generic forward-only/zeroth-order search over an adapter's parameter vector.

    Wraps any scalar `fitness_fn(params) -> float` (lower is better) and
    performs simple antithetic-sampling SPSA-style perturbation search,
    avoiding gradients entirely. Intended for small parameter vectors
    (output bias, residual scale, normalization affine params, prompt
    vectors) per research brief section 7.4 / FOA-style adaptation.
    """

    fitness_fn: Callable[[np.ndarray], float]
    sigma: float = 0.01
    n_samples: int = 8
    seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def step(self, params: np.ndarray, learning_rate: float) -> np.ndarray:
        """One antithetic forward-only update step; returns the new parameter vector.

        Only forward evaluations of fitness_fn are used (no backprop), matching
        the FOA/FOZO-style forward-only constraint discussed in the brief.
        """
        gradient_estimate = np.zeros_like(params)
        for _ in range(self.n_samples):
            perturbation = self._rng.standard_normal(params.shape)
            fitness_plus = self.fitness_fn(params + self.sigma * perturbation)
            fitness_minus = self.fitness_fn(params - self.sigma * perturbation)
            gradient_estimate += (fitness_plus - fitness_minus) * perturbation
        gradient_estimate /= 2.0 * self.sigma * self.n_samples
        return params - learning_rate * gradient_estimate
