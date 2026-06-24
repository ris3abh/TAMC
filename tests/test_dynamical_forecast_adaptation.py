import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "experiments"))

from dynamical_forecast_adaptation import RollingLinearARForecaster


def test_rolling_linear_ar_returns_horizon_shape():
    rng = np.random.default_rng(0)
    context = np.sin(np.linspace(0, 8 * np.pi, 128)) + rng.normal(0, 0.01, 128)

    forecaster = RollingLinearARForecaster(horizon=8, train_context_length=64)
    forecast = forecaster.predict(context)

    assert forecast.shape == (8,)
    assert np.all(np.isfinite(forecast))


def test_rolling_linear_ar_falls_back_on_short_context():
    context = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    forecaster = RollingLinearARForecaster(horizon=3, train_context_length=64)
    forecast = forecaster.predict(context)

    assert forecast.shape == (3,)
    assert np.all(forecast == context[-1])


def test_rolling_linear_ar_is_deterministic():
    rng = np.random.default_rng(1)
    context = np.sin(np.linspace(0, 8 * np.pi, 128)) + rng.normal(0, 0.02, 128)

    forecaster = RollingLinearARForecaster(horizon=8, train_context_length=64)
    forecast_a = forecaster.predict(context)
    forecast_b = forecaster.predict(context)

    assert np.array_equal(forecast_a, forecast_b)


def test_rolling_linear_ar_uses_only_given_context():
    series = np.sin(np.linspace(0, 8 * np.pi, 200))
    context = series[:128]

    forecaster = RollingLinearARForecaster(horizon=8, train_context_length=64)
    forecast_a = forecaster.predict(context)

    corrupted = series.copy()
    corrupted[128:] = 999.0
    forecast_b = forecaster.predict(corrupted[:128])

    assert np.allclose(forecast_a, forecast_b)
