"""Frozen forecasters used as the base model TAMC-Lite adapts around.

These models are trained once (or not at all) and never updated at
inference time; only the adapters in adapters.py touch their output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class NaiveLastValueForecaster:
    """Repeats the last observed context value for the full forecast horizon."""

    horizon: int

    def predict(self, context: np.ndarray) -> np.ndarray:
        context = np.asarray(context, dtype=float)
        last_value = context[-1]
        return np.full(self.horizon, last_value)

    def __call__(self, context: np.ndarray) -> np.ndarray:
        return self.predict(context)


@dataclass
class RecentPatternForecaster:
    """Forecasts by continuing the most recent observed local pattern.

    Estimates a dominant lag from the context's autocorrelation (the first
    local maximum beyond short-range smoothness, not the raw global
    maximum, which is always near lag 0 for a smooth signal) and forecasts
    via seasonal-naive-with-drift: it anchors at the current level and adds
    the change the signal showed `horizon` steps after the same phase one
    lag ago, rather than splicing in raw historical amplitude (which drifts
    under amplitude/beat modulation in quasi-periodic signals). Falls back
    to a simple continuation when no reliable periodic structure is found,
    or when the detected lag is too short to support that continuation.
    Uses only the given context: no labels, no future data, no gradient
    updates.
    """

    horizon: int
    min_lag: int = 4
    max_lag: int | None = None
    fallback: str = "last_value"
    score_threshold: float = 0.3

    def _dominant_lag(self, context: np.ndarray) -> int | None:
        n = len(context)
        max_lag = self.max_lag if self.max_lag is not None else n // 2
        max_lag = min(max_lag, n - 1)
        if max_lag < self.min_lag + 1:
            return None

        centered = context - context.mean()
        denom = np.sum(centered**2)
        if denom == 0:
            return None

        lags = np.arange(self.min_lag, max_lag + 1)
        correlations = (
            np.array([np.sum(centered[:-lag] * centered[lag:]) for lag in lags]) / denom
        )

        best_lag = None
        best_score = 0.0
        for i in range(1, len(correlations) - 1):
            is_local_peak = (
                correlations[i] > correlations[i - 1]
                and correlations[i] > correlations[i + 1]
            )
            if is_local_peak and correlations[i] > best_score:
                best_score = correlations[i]
                best_lag = int(lags[i])

        if best_lag is None or best_score < self.score_threshold:
            return None
        return best_lag

    def _fallback_forecast(self, context: np.ndarray) -> np.ndarray:
        if self.fallback == "linear":
            x = np.arange(len(context))
            slope, intercept = np.polyfit(x, context, 1)
            future_x = np.arange(len(context), len(context) + self.horizon)
            return slope * future_x + intercept
        return np.full(self.horizon, context[-1])

    def predict(self, context: np.ndarray) -> np.ndarray:
        context = np.asarray(context, dtype=float)
        lag = self._dominant_lag(context)
        if lag is None or lag <= self.horizon:
            return self._fallback_forecast(context)

        last_value = context[-1]
        reference = context[-(lag + 1)]
        future_phase = context[-lag : -lag + self.horizon]
        return last_value + (future_phase - reference)

    def __call__(self, context: np.ndarray) -> np.ndarray:
        return self.predict(context)


@dataclass
class LinearARForecaster:
    """Linear autoregressive forecaster fit once on source-regime data only.

    Predicts all `horizon` steps jointly from a `context_length` window via
    ridge-regularized least squares (no sklearn dependency).
    """

    context_length: int
    horizon: int
    ridge_lambda: float = 1e-3
    weights: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    bias: np.ndarray = field(default_factory=lambda: np.empty(0))

    def fit(
        self, series: np.ndarray, context_length: int, horizon: int
    ) -> "LinearARForecaster":
        """Fit on a source-regime series via ridge regression (closed-form)."""
        series = np.asarray(series, dtype=float)
        self.context_length = context_length
        self.horizon = horizon

        n_samples = len(series) - context_length - horizon + 1
        if n_samples < 1:
            raise ValueError("series too short for given context_length and horizon")

        X = np.stack([series[i : i + context_length] for i in range(n_samples)])
        Y = np.stack(
            [
                series[i + context_length : i + context_length + horizon]
                for i in range(n_samples)
            ]
        )

        X_design = np.concatenate([X, np.ones((n_samples, 1))], axis=1)
        n_features = X_design.shape[1]
        regularizer = self.ridge_lambda * np.eye(n_features)
        regularizer[-1, -1] = 0.0  # do not regularize the bias term

        coeffs = np.linalg.solve(X_design.T @ X_design + regularizer, X_design.T @ Y)
        self.weights = coeffs[:-1]
        self.bias = coeffs[-1]
        return self

    def predict(self, context: np.ndarray) -> np.ndarray:
        context = np.asarray(context, dtype=float)
        if context.shape[0] != self.context_length:
            raise ValueError(
                f"context length {context.shape[0]} must equal "
                f"context_length {self.context_length}"
            )
        return context @ self.weights + self.bias

    def __call__(self, context: np.ndarray) -> np.ndarray:
        return self.predict(context)
