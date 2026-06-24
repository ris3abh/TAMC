import numpy as np

from adapters import AnalogResidualAdapter, MeanShiftResidual
from delay_embedding import EmbeddingConfig
from forecasting import NaiveLastValueForecaster, RecentPatternForecaster
from tamc_pipeline import TamicBlendPipeline, TamicLitePipeline
from tamic_signal import TamicSignal


def _build_pipeline(
    topology_window: int = 32, context_length: int = 8, horizon: int = 4
):
    config = EmbeddingConfig(dimension=3, delay=2, window=topology_window)
    signal = TamicSignal(config=config, max_dimension=1, drift_dimension=1)

    rng = np.random.default_rng(0)
    source_window = np.sin(np.linspace(0, 4 * np.pi, topology_window))
    source_window += rng.normal(0, 0.01, topology_window)
    signal.add_source_prototype(source_window, label="source")

    forecaster = NaiveLastValueForecaster(horizon=horizon)
    residual = MeanShiftResidual(source_mean=0.0, horizon=horizon)

    pipeline = TamicLitePipeline(
        forecaster=forecaster,
        tamic_signal=signal,
        residual_adapter=residual,
        context_length=context_length,
        topology_window=topology_window,
        horizon=horizon,
    )
    return pipeline, topology_window, context_length, horizon


def test_pipeline_predict_returns_expected_keys():
    pipeline, topology_window, context_length, horizon = _build_pipeline()

    context = np.linspace(0, 1, context_length)
    topology_window_values = np.sin(np.linspace(0, 4 * np.pi, topology_window))

    result = pipeline.predict(context, topology_window_values)

    expected_keys = {
        "base_forecast",
        "adapted_forecast",
        "gate",
        "topological_distance",
        "prototype_label",
    }
    assert expected_keys == set(result.keys())


def test_pipeline_adapted_forecast_shape_matches_base_forecast():
    pipeline, topology_window, context_length, horizon = _build_pipeline()

    context = np.linspace(0, 1, context_length)
    topology_window_values = np.sin(np.linspace(0, 4 * np.pi, topology_window))

    result = pipeline.predict(context, topology_window_values)

    assert result["base_forecast"].shape == (horizon,)
    assert result["adapted_forecast"].shape == result["base_forecast"].shape


def test_pipeline_gate_is_between_zero_and_one():
    pipeline, topology_window, context_length, horizon = _build_pipeline()

    context = np.linspace(0, 1, context_length)

    for offset in range(5):
        topology_window_values = np.sin(
            np.linspace(offset, offset + 4 * np.pi, topology_window)
        )
        result = pipeline.predict(context, topology_window_values)
        assert 0.0 <= result["gate"] <= 1.0


def test_pipeline_can_use_analog_residual_adapter():
    topology_window, context_length, horizon = 32, 8, 4
    config = EmbeddingConfig(dimension=3, delay=2, window=topology_window)
    signal = TamicSignal(config=config, max_dimension=1, drift_dimension=1)

    rng = np.random.default_rng(0)
    source_window = np.sin(np.linspace(0, 4 * np.pi, topology_window))
    source_window += rng.normal(0, 0.01, topology_window)
    signal.add_source_prototype(source_window, label="source")

    forecaster = NaiveLastValueForecaster(horizon=horizon)

    n_memory = 20
    contexts = rng.normal(0, 1, size=(n_memory, context_length))
    targets = rng.normal(0, 1, size=(n_memory, horizon))
    base_predictions = np.stack([forecaster.predict(c) for c in contexts])
    residual = AnalogResidualAdapter(horizon=horizon, k=5)
    residual.fit(contexts, targets, base_predictions)

    pipeline = TamicLitePipeline(
        forecaster=forecaster,
        tamic_signal=signal,
        residual_adapter=residual,
        context_length=context_length,
        topology_window=topology_window,
        horizon=horizon,
    )

    context = np.linspace(0, 1, context_length)
    topology_window_values = np.sin(np.linspace(0, 4 * np.pi, topology_window))
    result = pipeline.predict(context, topology_window_values)

    assert result["adapted_forecast"].shape == (horizon,)
    assert np.all(np.isfinite(result["adapted_forecast"]))


def _build_blend_pipeline(
    topology_window: int = 32, context_length: int = 8, horizon: int = 4
):
    config = EmbeddingConfig(dimension=3, delay=2, window=topology_window)
    signal = TamicSignal(config=config, max_dimension=1, drift_dimension=1)

    rng = np.random.default_rng(0)
    source_window = np.sin(np.linspace(0, 4 * np.pi, topology_window))
    source_window += rng.normal(0, 0.01, topology_window)
    signal.add_source_prototype(source_window, label="source")

    frozen_forecaster = NaiveLastValueForecaster(horizon=horizon)
    adaptive_forecaster = RecentPatternForecaster(horizon=horizon, min_lag=4)

    pipeline = TamicBlendPipeline(
        frozen_forecaster=frozen_forecaster,
        adaptive_forecaster=adaptive_forecaster,
        tamic_signal=signal,
        context_length=context_length,
        topology_window=topology_window,
        horizon=horizon,
    )
    return pipeline, topology_window, context_length, horizon


def test_blend_pipeline_predict_returns_expected_keys():
    pipeline, topology_window, context_length, horizon = _build_blend_pipeline()

    context = np.linspace(0, 1, context_length)
    topology_window_values = np.sin(np.linspace(0, 4 * np.pi, topology_window))

    result = pipeline.predict(context, topology_window_values)

    expected_keys = {
        "frozen_forecast",
        "adaptive_forecast",
        "blended_forecast",
        "gate",
        "topological_distance",
        "prototype_label",
    }
    assert expected_keys == set(result.keys())


def test_blend_pipeline_blended_forecast_shape_matches_frozen_forecast():
    pipeline, topology_window, context_length, horizon = _build_blend_pipeline()

    context = np.linspace(0, 1, context_length)
    topology_window_values = np.sin(np.linspace(0, 4 * np.pi, topology_window))

    result = pipeline.predict(context, topology_window_values)

    assert result["frozen_forecast"].shape == (horizon,)
    assert result["adaptive_forecast"].shape == (horizon,)
    assert result["blended_forecast"].shape == result["frozen_forecast"].shape


def test_blend_pipeline_gate_is_between_zero_and_one():
    pipeline, topology_window, context_length, horizon = _build_blend_pipeline()

    context = np.linspace(0, 1, context_length)

    for offset in range(5):
        topology_window_values = np.sin(
            np.linspace(offset, offset + 4 * np.pi, topology_window)
        )
        result = pipeline.predict(context, topology_window_values)
        assert 0.0 <= result["gate"] <= 1.0


def test_blend_pipeline_blend_is_convex_combination():
    pipeline, topology_window, context_length, horizon = _build_blend_pipeline()

    context = np.linspace(0, 1, context_length)
    topology_window_values = np.sin(np.linspace(0, 4 * np.pi, topology_window))

    result = pipeline.predict(context, topology_window_values)
    gate = result["gate"]
    expected = (1.0 - gate) * result["frozen_forecast"] + gate * result[
        "adaptive_forecast"
    ]

    assert np.allclose(result["blended_forecast"], expected)
