import numpy as np

from drift_gates import ScalarDriftSignal


def test_gate_uses_zero_zscore_before_min_history():
    signal = ScalarDriftSignal()
    for distance in [1.0, 1.1, 0.9, 1.05, 0.95]:
        signal.score(distance)
        z = signal.drift_zscore(distance, min_history=8)
        gate = signal.gate(distance, threshold=2.0, min_history=8)
        assert z == 0.0
        assert gate == 1.0 / (1.0 + np.exp(2.0))


def test_gate_rises_for_large_deviation_after_min_history():
    signal = ScalarDriftSignal()
    rng = np.random.default_rng(0)
    baseline_scores = 1.0 + rng.normal(0, 0.01, size=10)
    for distance in baseline_scores:
        signal.score(float(distance))
        signal.gate(float(distance), threshold=2.0, min_history=8)

    low_gate = signal.gate(1.0, threshold=2.0, min_history=8)
    high_gate = signal.gate(50.0, threshold=2.0, min_history=8)

    assert low_gate < high_gate
    assert 0.0 <= low_gate <= 1.0
    assert 0.0 <= high_gate <= 1.0
    assert high_gate > 0.9


def test_gate_is_causal_and_does_not_mutate_on_query():
    signal = ScalarDriftSignal()
    for distance in [1.0] * 10:
        signal.score(distance)

    history_before = list(signal.history)
    signal.gate(1.0, threshold=2.0, min_history=8)
    signal.drift_zscore(1.0, min_history=8)

    assert signal.history == history_before
