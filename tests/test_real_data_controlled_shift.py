import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "experiments"))

from real_data_controlled_shift import inject_shift


def test_amplitude_shift_scales_values():
    post = np.array([1.0, -2.0, 3.0])
    shifted = inject_shift(post, "amplitude", seed=0, amplitude_factor=1.35)
    assert np.allclose(shifted, post * 1.35)


def test_trend_shift_adds_ramp_ending_at_magnitude():
    post = np.zeros(10)
    shifted = inject_shift(post, "trend", seed=0, trend_magnitude=1.0)
    assert shifted.shape == post.shape
    assert np.isclose(shifted[0], 0.0)
    assert np.isclose(shifted[-1], 1.0)
    assert np.all(np.diff(shifted) >= 0)


def test_noise_shift_is_deterministic_given_seed():
    post = np.zeros(20)
    shifted_a = inject_shift(post, "noise", seed=0, noise_std=0.35)
    shifted_b = inject_shift(post, "noise", seed=0, noise_std=0.35)
    shifted_c = inject_shift(post, "noise", seed=1, noise_std=0.35)
    assert np.allclose(shifted_a, shifted_b)
    assert not np.allclose(shifted_a, shifted_c)
    assert shifted_a.shape == post.shape


def test_seasonality_break_preserves_length_and_changes_values():
    post = np.sin(np.linspace(0, 8 * np.pi, 64))
    shifted = inject_shift(post, "seasonality_break", seed=0)
    assert shifted.shape == post.shape
    assert not np.allclose(shifted, post)


def test_frequency_proxy_preserves_length():
    post = np.sin(np.linspace(0, 8 * np.pi, 64))
    for factor in [0.5, 1.0, 1.3, 2.0]:
        shifted = inject_shift(post, "frequency_proxy", seed=0, frequency_factor=factor)
        assert shifted.shape == post.shape
        assert np.all(np.isfinite(shifted))


def test_unknown_shift_type_raises():
    post = np.zeros(10)
    try:
        inject_shift(post, "not_a_real_shift", seed=0)
        assert False, "expected ValueError"
    except ValueError:
        pass
