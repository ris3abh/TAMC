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
