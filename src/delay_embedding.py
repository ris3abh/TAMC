"""Delay-coordinate (Takens) embedding utilities for streaming time series."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EmbeddingConfig:
    """Parameters for a Takens delay embedding."""

    dimension: int
    delay: int
    window: int

    def __post_init__(self) -> None:
        if self.dimension < 1:
            raise ValueError("dimension must be >= 1")
        if self.delay < 1:
            raise ValueError("delay must be >= 1")
        min_window = (self.dimension - 1) * self.delay + 1
        if self.window < min_window:
            raise ValueError(
                f"window must be >= {min_window} for dimension={self.dimension}, "
                f"delay={self.delay}"
            )


def takens_embedding(series: np.ndarray, dimension: int, delay: int) -> np.ndarray:
    """Embed a 1D series into delay-coordinate space.

    Returns an array of shape (n_points, dimension) where row i is
    [x_i, x_{i+delay}, ..., x_{i+(dimension-1)*delay}].
    """
    series = np.asarray(series, dtype=float)
    if series.ndim != 1:
        raise ValueError("series must be 1D; embed each channel separately")

    n = series.shape[0]
    n_points = n - (dimension - 1) * delay
    if n_points < 1:
        raise ValueError("series too short for given dimension/delay")

    indices = np.arange(n_points)[:, None] + np.arange(dimension)[None, :] * delay
    return series[indices]


def multivariate_delay_embedding(
    channels: np.ndarray, dimension: int, delay: int
) -> np.ndarray:
    """Delay-embed each channel of a multivariate series and concatenate coordinates.

    channels: array of shape (n_samples, n_channels).
    Returns an array of shape (n_points, dimension * n_channels).
    """
    channels = np.asarray(channels, dtype=float)
    if channels.ndim != 2:
        raise ValueError("channels must be 2D: (n_samples, n_channels)")

    embeddings = [
        takens_embedding(channels[:, c], dimension, delay)
        for c in range(channels.shape[1])
    ]
    n_points = min(e.shape[0] for e in embeddings)
    return np.concatenate([e[:n_points] for e in embeddings], axis=1)


def sliding_windows(series: np.ndarray, window: int, stride: int = 1) -> np.ndarray:
    """Extract overlapping sliding windows from a 1D series.

    Returns an array of shape (n_windows, window).
    """
    series = np.asarray(series, dtype=float)
    if series.ndim != 1:
        raise ValueError("series must be 1D")
    if window > series.shape[0]:
        raise ValueError("window larger than series length")

    n_windows = (series.shape[0] - window) // stride + 1
    indices = np.arange(n_windows)[:, None] * stride + np.arange(window)[None, :]
    return series[indices]


def fit_train_normalization(series: np.ndarray) -> tuple[float, float]:
    """Compute mean/std from a training-only segment for later z-normalization."""
    series = np.asarray(series, dtype=float)
    mean = float(series.mean())
    std = float(series.std())
    if std == 0.0:
        std = 1.0
    return mean, std


def apply_normalization(series: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Apply a previously fit train-only normalization to new data."""
    return (np.asarray(series, dtype=float) - mean) / std


def sliding_attractor_point_clouds(
    series: np.ndarray, config: EmbeddingConfig, stride: int = 1
) -> list[np.ndarray]:
    """Build a sequence of delay-embedded point clouds from successive windows.

    Each window of `config.window` raw samples is delay-embedded with
    `config.dimension` and `config.delay` to produce one point cloud. The
    windows themselves slide over the raw series with the given stride,
    yielding the sequence of point clouds a streaming topology monitor would
    consume.
    """
    windows = sliding_windows(series, config.window, stride=stride)
    return [
        takens_embedding(window, config.dimension, config.delay) for window in windows
    ]
