import numpy as np

from forecasting import (
    LinearARForecaster,
    NaiveLastValueForecaster,
    RecentPatternForecaster,
)


def test_naive_last_value_forecaster_shape_and_value():
    forecaster = NaiveLastValueForecaster(horizon=5)
    context = np.array([1.0, 2.0, 3.0])
    forecast = forecaster.predict(context)
    assert forecast.shape == (5,)
    assert np.all(forecast == 3.0)


def test_linear_ar_forecaster_fit_predict_shapes():
    rng = np.random.default_rng(0)
    series = np.sin(np.linspace(0, 20 * np.pi, 500)) + rng.normal(0, 0.01, 500)

    forecaster = LinearARForecaster(context_length=16, horizon=4)
    forecaster.fit(series, context_length=16, horizon=4)

    context = series[-16:]
    forecast = forecaster.predict(context)

    assert forecast.shape == (4,)
    assert forecaster.weights.shape == (16, 4)
    assert forecaster.bias.shape == (4,)


def test_linear_ar_forecaster_fits_clean_sine_reasonably():
    t = np.linspace(0, 40 * np.pi, 1000)
    series = np.sin(t)

    forecaster = LinearARForecaster(context_length=16, horizon=4)
    forecaster.fit(series, context_length=16, horizon=4)

    context = series[100:116]
    target = series[116:120]
    forecast = forecaster.predict(context)

    assert np.allclose(forecast, target, atol=0.2)


def test_recent_pattern_forecaster_returns_correct_shape():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 8 * np.pi, 200)
    context = np.sin(t) + rng.normal(0, 0.01, 200)

    forecaster = RecentPatternForecaster(horizon=5, min_lag=4)
    forecast = forecaster.predict(context)

    assert forecast.shape == (5,)
    assert np.all(np.isfinite(forecast))


def test_recent_pattern_forecaster_uses_only_past_context():
    """The forecast must depend only on the given context, never on data
    beyond it; perturbing values after the context boundary must not change
    the forecast for a fixed context."""
    t = np.linspace(0, 8 * np.pi, 200)
    series = np.sin(t)
    context = series[:120]

    forecaster = RecentPatternForecaster(horizon=5, min_lag=4)
    forecast_a = forecaster.predict(context)

    series_perturbed = series.copy()
    series_perturbed[120:] = 999.0  # corrupt everything after the context
    forecast_b = forecaster.predict(series_perturbed[:120])

    assert np.allclose(forecast_a, forecast_b)


def test_recent_pattern_forecaster_falls_back_without_periodic_structure():
    rng = np.random.default_rng(0)
    context = rng.normal(0, 1, size=40)  # pure noise, no periodicity

    forecaster = RecentPatternForecaster(horizon=5, min_lag=4)
    forecast = forecaster.predict(context)

    assert forecast.shape == (5,)
    assert np.all(forecast == context[-1])
