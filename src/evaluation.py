"""Causal forecast-evaluation utilities for frozen forecasters and adapters."""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np


def make_supervised_windows(
    series: np.ndarray, context_length: int, horizon: int, stride: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Build (context, target) supervised pairs from a 1D series.

    Returns (contexts, targets) of shapes (n_windows, context_length) and
    (n_windows, horizon), where targets[i] immediately follows contexts[i].
    """
    series = np.asarray(series, dtype=float)
    n_samples = (len(series) - context_length - horizon) // stride + 1
    if n_samples < 1:
        raise ValueError("series too short for given context_length and horizon")

    contexts = np.stack(
        [series[i * stride : i * stride + context_length] for i in range(n_samples)]
    )
    targets = np.stack(
        [
            series[i * stride + context_length : i * stride + context_length + horizon]
            for i in range(n_samples)
        ]
    )
    return contexts, targets


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


class PredictFn(Protocol):
    def __call__(self, context: np.ndarray) -> np.ndarray: ...


def rolling_forecast_evaluation(
    series: np.ndarray,
    forecaster: PredictFn | Callable[[np.ndarray], np.ndarray],
    context_length: int,
    horizon: int,
    start_index: int,
    end_index: int | None = None,
    stride: int = 1,
) -> dict[str, np.ndarray]:
    """Causally roll a forecaster forward, predicting only from past context.

    At each time `t` in `range(start_index, end_index, stride)`, predicts
    `series[t:t+horizon]` from `series[t-context_length:t]`; never looks
    past `t`. `forecaster` may be any callable with a `predict(context)`
    method or a plain `__call__(context)` callable.

    Returns a dict with `times`, `predictions` (n_windows, horizon), and
    `targets` (n_windows, horizon).
    """
    series = np.asarray(series, dtype=float)
    if start_index < context_length:
        raise ValueError("start_index must be >= context_length")
    if end_index is None:
        end_index = len(series) - horizon
    if end_index > len(series) - horizon:
        raise ValueError("end_index too large: targets would run past series end")

    predict = getattr(forecaster, "predict", forecaster)

    times = []
    predictions = []
    targets = []
    for t in range(start_index, end_index, stride):
        context = series[t - context_length : t]
        target = series[t : t + horizon]
        prediction = predict(context)

        times.append(t)
        predictions.append(np.asarray(prediction, dtype=float))
        targets.append(target)

    return {
        "times": np.array(times),
        "predictions": np.stack(predictions),
        "targets": np.stack(targets),
    }
