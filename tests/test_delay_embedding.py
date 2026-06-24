import numpy as np

from delay_embedding import (
    EmbeddingConfig,
    sliding_attractor_point_clouds,
    sliding_windows,
    takens_embedding,
)


def test_takens_embedding_shape():
    series = np.arange(10, dtype=float)
    embedded = takens_embedding(series, dimension=3, delay=2)
    assert embedded.shape == (6, 3)


def test_sliding_windows_shape():
    series = np.arange(10, dtype=float)
    windows = sliding_windows(series, window=4, stride=2)
    assert windows.shape == (4, 4)


def test_sliding_attractor_point_clouds():
    series = np.arange(20, dtype=float)
    config = EmbeddingConfig(dimension=3, delay=2, window=10)
    clouds = sliding_attractor_point_clouds(series, config, stride=5)
    assert len(clouds) == 3
    assert clouds[0].shape == (6, 3)
