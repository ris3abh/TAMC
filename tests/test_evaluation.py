import numpy as np

from evaluation import (
    mae,
    make_supervised_windows,
    rmse,
    rolling_forecast_evaluation,
)
from forecasting import NaiveLastValueForecaster


def test_make_supervised_windows_shapes():
    series = np.arange(20, dtype=float)
    contexts, targets = make_supervised_windows(
        series, context_length=5, horizon=3, stride=2
    )
    assert contexts.shape[1] == 5
    assert targets.shape[1] == 3
    assert contexts.shape[0] == targets.shape[0]


def test_mae_and_rmse_zero_for_identical_inputs():
    y = np.array([1.0, 2.0, 3.0])
    assert mae(y, y) == 0.0
    assert rmse(y, y) == 0.0


def test_mae_and_rmse_basic_values():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([3.0, 4.0])
    assert mae(y_true, y_pred) == 3.5
    assert rmse(y_true, y_pred) == 2.5 * np.sqrt(2)


def test_rolling_forecast_evaluation_runs_causally():
    series = np.arange(50, dtype=float)
    forecaster = NaiveLastValueForecaster(horizon=3)

    result = rolling_forecast_evaluation(
        series=series,
        forecaster=forecaster,
        context_length=5,
        horizon=3,
        start_index=10,
        end_index=20,
        stride=1,
    )

    assert result["predictions"].shape == (10, 3)
    assert result["targets"].shape == (10, 3)
    assert result["times"].shape == (10,)
    # Naive last-value forecast at time t should equal series[t - 1].
    assert np.allclose(result["predictions"][0], series[result["times"][0] - 1])
