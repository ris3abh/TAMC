import numpy as np

from adapters import AnalogResidualAdapter


def _toy_memory(n_memory: int = 12, context_length: int = 6, horizon: int = 3):
    rng = np.random.default_rng(0)
    contexts = rng.normal(0, 1, size=(n_memory, context_length))
    targets = rng.normal(0, 1, size=(n_memory, horizon))
    base_predictions = rng.normal(0, 0.1, size=(n_memory, horizon))
    return contexts, targets, base_predictions


def test_analog_residual_adapter_fit_stores_memory():
    contexts, targets, base_predictions = _toy_memory()
    horizon = targets.shape[1]

    adapter = AnalogResidualAdapter(horizon=horizon, k=5)
    adapter.fit(contexts, targets, base_predictions)

    assert adapter.memory_contexts_.shape == contexts.shape
    assert adapter.memory_residuals_.shape == targets.shape
    assert np.allclose(adapter.memory_residuals_, targets - base_predictions)


def test_analog_residual_adapter_correction_shape_and_finite():
    contexts, targets, base_predictions = _toy_memory()
    horizon = targets.shape[1]

    adapter = AnalogResidualAdapter(horizon=horizon, k=5)
    adapter.fit(contexts, targets, base_predictions)

    query_context = np.zeros(contexts.shape[1])
    correction = adapter.correction(query_context)

    assert correction.shape == (horizon,)
    assert np.all(np.isfinite(correction))


def test_analog_residual_adapter_respects_max_correction_norm():
    contexts, targets, base_predictions = _toy_memory()
    horizon = targets.shape[1]

    adapter = AnalogResidualAdapter(horizon=horizon, k=5, max_correction_norm=0.01)
    adapter.fit(contexts, targets, base_predictions)

    query_context = np.zeros(contexts.shape[1])
    correction = adapter.correction(query_context)

    assert np.linalg.norm(correction) <= 0.01 + 1e-9


def test_analog_residual_adapter_raises_before_fit():
    adapter = AnalogResidualAdapter(horizon=3, k=5)
    try:
        adapter.correction(np.zeros(6))
        assert False, "expected RuntimeError before fit"
    except RuntimeError:
        pass
