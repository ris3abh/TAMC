"""Synthetic regime-shift experiment for TAMC.

Goal:
    Show that topological attractor drift can detect a structural shift
    from periodic to quasi-periodic dynamics while simple first-order
    statistics remain comparatively weak.

Run:
    uv run python experiments/synthetic_regime_shift.py --seed 0
    uv run python experiments/synthetic_regime_shift.py --multi-seed 10
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from delay_embedding import EmbeddingConfig, sliding_windows
from tamic_signal import TamicSignal


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    std = x.std()
    if std == 0:
        return x - x.mean()
    return (x - x.mean()) / std


def minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo = np.nanmin(x)
    hi = np.nanmax(x)
    if hi - lo == 0:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def generate_sine_to_quasiperiodic(
    n_per_regime: int = 700,
    noise_std: float = 0.03,
    seed: int = 0,
) -> tuple[np.ndarray, int]:
    """Generate periodic -> quasi-periodic shift with matched mean/variance."""
    rng = np.random.default_rng(seed)

    t1 = np.linspace(0, 28 * np.pi, n_per_regime)
    t2 = np.linspace(0, 28 * np.pi, n_per_regime)

    regime_1 = np.sin(t1)
    regime_2 = np.sin(t2) + 0.55 * np.sin(np.sqrt(2) * t2 + 0.3)

    regime_1 = zscore(regime_1)
    regime_2 = zscore(regime_2)

    regime_1 = regime_1 + rng.normal(0, noise_std, size=n_per_regime)
    regime_2 = regime_2 + rng.normal(0, noise_std, size=n_per_regime)

    # Re-normalize after noise so mean/variance remain closely matched.
    regime_1 = zscore(regime_1)
    regime_2 = zscore(regime_2)

    series = np.concatenate([regime_1, regime_2])
    shift_index = n_per_regime
    return series, shift_index


def autocorrelation_signature(window: np.ndarray, max_lag: int = 32) -> np.ndarray:
    """Autocorrelation vector for lags 1..max_lag."""
    window = zscore(window)
    values = []
    for lag in range(1, max_lag + 1):
        if lag >= len(window):
            values.append(0.0)
            continue
        values.append(float(np.mean(window[:-lag] * window[lag:])))
    return np.array(values)


def spectral_signature(window: np.ndarray) -> np.ndarray:
    """Normalized FFT power spectrum signature."""
    window = zscore(window)
    spectrum = np.abs(np.fft.rfft(window)) ** 2
    total = spectrum.sum()
    if total == 0:
        return spectrum
    return spectrum / total


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    return float(np.linalg.norm(a[:n] - b[:n]))


def compute_baseline_drifts(
    series: np.ndarray,
    source_window: np.ndarray,
    window: int,
    stride: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    windows = sliding_windows(series, window=window, stride=stride)
    times = np.arange(len(windows)) * stride + window - 1

    source_mean = source_window.mean()
    source_var = source_window.var()
    source_acf = autocorrelation_signature(source_window)
    source_spec = spectral_signature(source_window)

    mean_drift = []
    var_drift = []
    acf_drift = []
    spec_drift = []

    for w in windows:
        mean_drift.append(abs(w.mean() - source_mean))
        var_drift.append(abs(w.var() - source_var))
        acf_drift.append(l2_distance(autocorrelation_signature(w), source_acf))
        spec_drift.append(l2_distance(spectral_signature(w), source_spec))

    drifts = {
        "Rolling mean drift": np.array(mean_drift),
        "Rolling variance drift": np.array(var_drift),
        "Autocorrelation drift": np.array(acf_drift),
        "Spectral drift": np.array(spec_drift),
    }
    return times, drifts


def compute_topological_drift(
    series: np.ndarray,
    source_window: np.ndarray,
    window: int,
    stride: int,
) -> np.ndarray:
    config = EmbeddingConfig(dimension=3, delay=8, window=window)
    signal = TamicSignal(config=config, max_dimension=1, drift_dimension=1)
    signal.add_source_prototype(source_window, label="periodic_source")

    windows = sliding_windows(series, window=window, stride=stride)
    distances = []
    for w in windows:
        score = signal.score_window(w)
        distances.append(score.distance)

    return np.array(distances)


def plot_results(
    series: np.ndarray,
    times: np.ndarray,
    shift_index: int,
    drifts: dict[str, np.ndarray],
    topological_drift: np.ndarray,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(6, 1, figsize=(12, 11), sharex=True)

    axes[0].plot(np.arange(len(series)), series)
    axes[0].axvline(shift_index, linestyle="--")
    axes[0].set_ylabel("Signal")
    axes[0].set_title("Synthetic regime shift: periodic to quasi-periodic")

    plot_items = list(drifts.items()) + [("TAMC topological drift", topological_drift)]

    for ax, (name, values) in zip(axes[1:], plot_items):
        ax.plot(times, minmax(values))
        ax.axvline(shift_index, linestyle="--")
        ax.set_ylabel(name)

    axes[-1].set_xlabel("Time index")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def detection_metrics(
    drift: np.ndarray,
    times: np.ndarray,
    shift_index: int,
    n_std: float = 3.0,
) -> dict[str, float]:
    """Threshold-based detection metrics for one drift signal.

    Threshold is set from the pre-shift segment only (pre_mean + n_std *
    pre_std), matching a standard anomaly-detection convention where the
    "normal" regime defines what counts as alarm-worthy.
    """
    pre_mask = times < shift_index
    post_mask = ~pre_mask

    pre = drift[pre_mask]
    post = drift[post_mask]
    post_times = times[post_mask]

    pre_mean = float(pre.mean())
    pre_std = float(pre.std())
    threshold = pre_mean + n_std * pre_std

    exceed = np.where(post > threshold)[0]
    delay = float(post_times[exceed[0]] - shift_index) if exceed.size else float("nan")
    false_alarms = int(np.sum(pre > threshold))

    auroc = float("nan")
    try:
        from sklearn.metrics import roc_auc_score

        labels = (times >= shift_index).astype(int)
        auroc = float(roc_auc_score(labels, drift))
    except ImportError:
        pass

    post_mean = float(post.mean()) if post.size else float("nan")
    separation = (post_mean - pre_mean) / (pre_std + 1e-8)

    return {
        "Pre Mean": pre_mean,
        "Pre Std": pre_std,
        "Post Mean": post_mean,
        "Max Post": float(post.max()) if post.size else float("nan"),
        "Delay": delay,
        "False Alarms": false_alarms,
        "AUROC": auroc,
        "Separation": separation,
    }


def build_metrics_table(
    times: np.ndarray,
    signals: dict[str, np.ndarray],
    shift_index: int,
    n_std: float = 3.0,
) -> pd.DataFrame:
    """Build a comparison table of detection metrics across all drift signals."""
    rows = {
        name: detection_metrics(values, times, shift_index, n_std=n_std)
        for name, values in signals.items()
    }
    return pd.DataFrame(rows).T


def run_experiment(
    seed: int, window: int = 128, stride: int = 8
) -> tuple[pd.DataFrame, dict]:
    """Run one full seed of the experiment; return its metrics table and raw arrays."""
    series, shift_index = generate_sine_to_quasiperiodic(seed=seed)
    source_window = series[:window]

    times, drifts = compute_baseline_drifts(
        series=series,
        source_window=source_window,
        window=window,
        stride=stride,
    )
    topo_drift = compute_topological_drift(
        series=series,
        source_window=source_window,
        window=window,
        stride=stride,
    )

    all_signals = {**drifts, "TAMC topological drift": topo_drift}
    metrics_table = build_metrics_table(times, all_signals, shift_index)

    raw = {
        "series": series,
        "shift_index": shift_index,
        "times": times,
        "drifts": drifts,
        "topo_drift": topo_drift,
    }
    return metrics_table, raw


def run_single_seed(seed: int, figures_dir: Path) -> None:
    metrics_table, raw = run_experiment(seed=seed)

    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figures_dir / "synthetic_regime_shift_drift.png"
    plot_results(
        series=raw["series"],
        times=raw["times"],
        shift_index=raw["shift_index"],
        drifts=raw["drifts"],
        topological_drift=raw["topo_drift"],
        output_path=figure_path,
    )

    metrics_path = figures_dir / "synthetic_regime_shift_metrics.csv"
    metrics_table.to_csv(metrics_path, index_label="Method")

    series, shift_index = raw["series"], raw["shift_index"]
    print(f"Saved figure to {figure_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Shift index: {shift_index}")
    print("Mean/variance check:")
    print(f"  pre mean={series[:shift_index].mean():.4f}, pre var={series[:shift_index].var():.4f}")
    print(f"  post mean={series[shift_index:].mean():.4f}, post var={series[shift_index:].var():.4f}")
    print("\nDetection metrics (threshold = pre_mean + 3*pre_std):")
    print(metrics_table.to_string(float_format=lambda v: f"{v:.4f}"))


def run_multi_seed(n_seeds: int, figures_dir: Path) -> None:
    tables = []
    for seed in range(n_seeds):
        metrics_table, _ = run_experiment(seed=seed)
        metrics_table.index.name = "Method"
        tables.append(metrics_table.assign(Seed=seed))

    combined = pd.concat(tables)
    summary = combined.groupby("Method")[["AUROC", "Delay", "False Alarms", "Separation"]].agg(
        ["mean", "std"]
    )
    summary.columns = [f"{metric} {stat}" for metric, stat in summary.columns]

    figures_dir.mkdir(parents=True, exist_ok=True)
    summary_path = figures_dir / "synthetic_regime_shift_multiseed_metrics.csv"
    summary.to_csv(summary_path)

    print(f"Ran {n_seeds} seeds (0..{n_seeds - 1})")
    print(f"Saved multi-seed summary to {summary_path}")
    print("\nMulti-seed detection metrics (mean +/- std):")
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", type=int, default=0, help="Single-seed run (default: 0)."
    )
    parser.add_argument(
        "--multi-seed",
        type=int,
        default=None,
        metavar="N",
        help="Run seeds 0..N-1 and report mean +/- std detection metrics instead "
        "of a single-seed figure run.",
    )
    args = parser.parse_args()

    figures_dir = Path(__file__).resolve().parents[1] / "figures"

    if args.multi_seed is not None:
        run_multi_seed(args.multi_seed, figures_dir)
    else:
        run_single_seed(args.seed, figures_dir)


if __name__ == "__main__":
    main()
