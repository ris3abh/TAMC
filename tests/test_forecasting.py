import numpy as np

from forecasting import LinearARForecaster, NaiveLastValueForecaster


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
