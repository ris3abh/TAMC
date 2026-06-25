import numpy as np

from regime_similarity import (
    RegimeSimilaritySignal,
    compute_regime_signature,
    regime_similarity,
    variance_ratio_similarity,
)


def _assert_finite_signature(signature) -> None:
    for value in (
        signature.mean,
        signature.std,
        signature.skewness,
        signature.kurtosis,
        signature.lag1_autocorr,
    ):
        assert np.isfinite(value)


def test_compute_regime_signature_random_window():
    rng = np.random.default_rng(0)
    window = rng.normal(0, 1, size=64)
    _assert_finite_signature(compute_regime_signature(window))


def test_compute_regime_signature_constant_window():
    window = np.full(32, 5.0)
    signature = compute_regime_signature(window)
    _assert_finite_signature(signature)
    assert signature.std == 0.0
    assert signature.skewness == 0.0
    assert signature.kurtosis == 0.0
    assert signature.lag1_autocorr == 0.0


def test_compute_regime_signature_very_short_window():
    _assert_finite_signature(compute_regime_signature(np.array([1.0])))
    _assert_finite_signature(compute_regime_signature(np.array([1.0, 2.0])))


def test_regime_similarity_identical_source_is_high():
    rng = np.random.default_rng(0)
    source = rng.normal(0, 1, size=128)
    result = regime_similarity(source, source)
    assert result["combined_similarity"] > 0.99
    for key in (
        "feature_similarity",
        "ks_similarity",
        "wasserstein_similarity",
        "variance_ratio_similarity",
    ):
        assert 0.0 <= result[key] <= 1.0


def test_regime_similarity_distribution_shift_is_lower():
    rng = np.random.default_rng(0)
    source = rng.normal(0, 1, size=200)
    same_regime = rng.normal(0, 1, size=200)
    shifted_regime = rng.normal(8, 3, size=200)

    same_result = regime_similarity(source, same_regime)
    shifted_result = regime_similarity(source, shifted_regime)

    assert shifted_result["combined_similarity"] < same_result["combined_similarity"]


def test_variance_ratio_similarity_zero_variance_cases():
    constant_a = np.full(16, 3.0)
    constant_b = np.full(16, -7.0)
    rng = np.random.default_rng(0)
    varying = rng.normal(0, 1, size=16)

    assert variance_ratio_similarity(constant_a, constant_b) == 1.0
    assert variance_ratio_similarity(constant_a, varying) == 0.0
    assert variance_ratio_similarity(varying, constant_a) == 0.0


def test_regime_similarity_signal_gate_in_unit_interval():
    rng = np.random.default_rng(0)
    source_window = rng.normal(0, 1, size=64)
    signal = RegimeSimilaritySignal(source_window=source_window)

    for offset in range(12):
        window = rng.normal(offset * 0.5, 1, size=64)
        gate = signal.gate(window, threshold=2.0, min_history=8)
        assert 0.0 <= gate <= 1.0


def test_regime_similarity_signal_gate_does_not_fire_strongly_before_min_history():
    rng = np.random.default_rng(0)
    source_window = rng.normal(0, 1, size=64)
    signal = RegimeSimilaritySignal(source_window=source_window)

    for _ in range(5):  # fewer than min_history=8
        window = rng.normal(0, 1, size=64)
        gate = signal.gate(window, threshold=2.0, min_history=8)
        assert gate < 0.3
