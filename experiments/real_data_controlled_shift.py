"""Real-data controlled-shift experiment for TAMC-Lite.

Goal:
    Run TAMC-Lite forecast blending on a real time series (loaded from a
    local CSV, never downloaded) with a controlled, deterministic
    perturbation injected into the second half only, and compare it
    against an always-on blend and several non-topological gated blends
    under an identical "z-score crosses threshold -> sigmoid gate" control
    law (src/drift_gates.py::ScalarDriftSignal).

Protocol (see paper_notes/problem_statement.md for the formal version):
    - The series is causally normalized using only the source/pre-shift
      half; the shifted/post-shift half is never used to fit that
      normalization.
    - The frozen forecaster and the TAMC/baseline source prototypes are
      fit only on the source/pre-shift half.
    - At time t, every forecast and every gate uses only series[:t]; no
      future values (including the injected shift itself) ever leak into
      a prediction made before they occur.

Run:
    uv run python experiments/real_data_controlled_shift.py \\
        --csv-path data/ETTh1.csv --value-column OT --timestamp-column date \\
        --shift-type seasonality_break --seed 0

    uv run python experiments/real_data_controlled_shift.py \\
        --csv-path data/ETTh1.csv --value-column OT --timestamp-column date \\
        --shift-type seasonality_break --multi-seed 10
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from delay_embedding import (
    EmbeddingConfig,
    apply_normalization,
    fit_train_normalization,
)
from drift_gates import ScalarDriftSignal
from evaluation import mae, rmse
from forecasting import LinearARForecaster, RecentPatternForecaster
from tamc_pipeline import TamicBlendPipeline
from tamic_signal import TamicSignal

from synthetic_regime_shift import (
    autocorrelation_signature,
    l2_distance,
    spectral_signature,
)
from tamc_lite_synthetic_forecast import compute_tradeoff_metrics

SHIFT_TYPES = ("amplitude", "trend", "noise", "seasonality_break", "frequency_proxy")

VARIANT_NAMES = (
    "Frozen forecaster",
    "Adaptive recent-pattern forecaster",
    "Always-on 50/50 blend",
    "TAMC-gated blend",
    "Mean/variance-gated blend",
    "Autocorrelation-gated blend",
    "Spectral-gated blend",
)


def load_and_normalize_series(
    csv_path: str,
    value_column: str,
    timestamp_column: str | None,
    n_points: int,
) -> tuple[np.ndarray, int]:
    """Load a local CSV, select+clean one column, and causally normalize it.

    Sorts by `timestamp_column` if given, drops missing values, takes a
    contiguous prefix of length `n_points`, splits it in half, and fits
    normalization on the first (source/pre-shift) half only -- the second
    half is normalized with that same fitted mean/std, never its own.

    Returns (normalized_series, shift_index) with no shift injected yet.
    """
    columns = (
        [value_column] if timestamp_column is None else [timestamp_column, value_column]
    )
    frame = pd.read_csv(csv_path, usecols=columns)

    if timestamp_column is not None:
        frame = frame.sort_values(timestamp_column)

    frame = frame.dropna(subset=[value_column])
    series = frame[value_column].to_numpy(dtype=float)[:n_points]

    shift_index = len(series) // 2
    mean, std = fit_train_normalization(series[:shift_index])
    normalized = apply_normalization(series, mean, std)
    return normalized, shift_index


def inject_shift(
    post: np.ndarray,
    shift_type: str,
    seed: int,
    amplitude_factor: float = 1.35,
    trend_magnitude: float = 1.0,
    noise_std: float = 0.35,
    frequency_factor: float = 1.3,
) -> np.ndarray:
    """Apply one deterministic, controlled perturbation to a post-shift segment.

    Only `noise` actually uses `seed`; the other perturbation types are
    fully deterministic given the input segment, so multi-seed runs on
    those types correctly show zero across-seed variance.
    """
    post = np.asarray(post, dtype=float).copy()
    n = len(post)

    if shift_type == "amplitude":
        return post * amplitude_factor

    if shift_type == "trend":
        ramp = np.linspace(0.0, trend_magnitude, n)
        return post + ramp

    if shift_type == "noise":
        rng = np.random.default_rng(seed)
        return post + rng.normal(0.0, noise_std, size=n)

    if shift_type == "seasonality_break":
        # Mixing each value with its mirrored (reverse-time) counterpart
        # inverts the phase of any recurrence structure while keeping the
        # same underlying value distribution.
        return 0.5 * post + 0.5 * post[::-1]

    if shift_type == "frequency_proxy":
        # Resample the local time axis to mimic a frequency change, then
        # crop/pad back to the original length so downstream windowing is
        # unaffected.
        new_n = max(2, int(round(n / frequency_factor)))
        old_x = np.linspace(0.0, 1.0, n)
        new_x = np.linspace(0.0, 1.0, new_n)
        resampled = np.interp(new_x, old_x, post)
        if new_n >= n:
            return resampled[:n]
        pad_width = n - new_n
        return np.concatenate([resampled, np.full(pad_width, resampled[-1])])

    raise ValueError(
        f"unknown shift_type {shift_type!r}; expected one of {SHIFT_TYPES}"
    )


def run_experiment(
    base_series: np.ndarray,
    shift_index: int,
    shift_type: str,
    seed: int,
    context_length: int = 128,
    horizon: int = 8,
    window: int = 128,
    stride: int = 4,
    gate_threshold: float = 2.0,
    min_history: int = 8,
    amplitude_factor: float = 1.35,
    trend_magnitude: float = 1.0,
    noise_std: float = 0.35,
    frequency_factor: float = 1.3,
) -> tuple[pd.DataFrame, dict]:
    """Run one full seed; return the per-segment metrics table and raw arrays."""
    post = base_series[shift_index:]
    shifted_post = inject_shift(
        post,
        shift_type,
        seed,
        amplitude_factor=amplitude_factor,
        trend_magnitude=trend_magnitude,
        noise_std=noise_std,
        frequency_factor=frequency_factor,
    )
    series = np.concatenate([base_series[:shift_index], shifted_post])
    source_series = series[:shift_index]

    forecaster = LinearARForecaster(context_length=context_length, horizon=horizon)
    forecaster.fit(source_series, context_length=context_length, horizon=horizon)

    adaptive_forecaster = RecentPatternForecaster(horizon=horizon)

    config = EmbeddingConfig(dimension=3, delay=8, window=window)
    signal = TamicSignal(config=config, max_dimension=1, drift_dimension=1)
    signal.add_source_prototype(source_series[:window], label="source")

    blend_pipeline = TamicBlendPipeline(
        frozen_forecaster=forecaster,
        adaptive_forecaster=adaptive_forecaster,
        tamic_signal=signal,
        context_length=context_length,
        topology_window=window,
        horizon=horizon,
        gate_threshold=gate_threshold,
        min_history=min_history,
    )

    source_window = series[:window]
    source_mean = source_window.mean()
    source_var = source_window.var()
    source_acf = autocorrelation_signature(source_window)
    source_spec = spectral_signature(source_window)

    meanvar_signal = ScalarDriftSignal()
    acf_signal = ScalarDriftSignal()
    spec_signal = ScalarDriftSignal()

    start_index = max(context_length, window)
    end_index = len(series) - horizon

    times: list[int] = []
    gates: list[float] = []
    frozen_point: list[float] = []
    tamc_point: list[float] = []
    errors: dict[str, list[tuple[float, float]]] = {name: [] for name in VARIANT_NAMES}

    for t in range(start_index, end_index, stride):
        context = series[t - context_length : t]
        topology_window_values = series[t - window : t]
        target = series[t : t + horizon]

        # One pipeline call advances the shared topology-monitor history and
        # gives the frozen/adaptive forecasts and TAMC's gate.
        result = blend_pipeline.predict(context, topology_window_values)
        frozen_forecast = result["frozen_forecast"]
        adaptive_forecast = result["adaptive_forecast"]
        tamc_gate = result["gate"]
        tamc_blend_forecast = result["blended_forecast"]
        always_on_forecast = 0.5 * frozen_forecast + 0.5 * adaptive_forecast

        meanvar_drift = abs(topology_window_values.mean() - source_mean) + abs(
            topology_window_values.var() - source_var
        )
        acf_drift = l2_distance(
            autocorrelation_signature(topology_window_values), source_acf
        )
        spec_drift = l2_distance(
            spectral_signature(topology_window_values), source_spec
        )

        meanvar_signal.score(meanvar_drift)
        acf_signal.score(acf_drift)
        spec_signal.score(spec_drift)

        meanvar_gate = meanvar_signal.gate(
            meanvar_drift, threshold=gate_threshold, min_history=min_history
        )
        acf_gate = acf_signal.gate(
            acf_drift, threshold=gate_threshold, min_history=min_history
        )
        spec_gate = spec_signal.gate(
            spec_drift, threshold=gate_threshold, min_history=min_history
        )

        forecasts = {
            "Frozen forecaster": frozen_forecast,
            "Adaptive recent-pattern forecaster": adaptive_forecast,
            "Always-on 50/50 blend": always_on_forecast,
            "TAMC-gated blend": tamc_blend_forecast,
            "Mean/variance-gated blend": (1 - meanvar_gate) * frozen_forecast
            + meanvar_gate * adaptive_forecast,
            "Autocorrelation-gated blend": (1 - acf_gate) * frozen_forecast
            + acf_gate * adaptive_forecast,
            "Spectral-gated blend": (1 - spec_gate) * frozen_forecast
            + spec_gate * adaptive_forecast,
        }

        times.append(t)
        gates.append(tamc_gate)
        frozen_point.append(frozen_forecast[0])
        tamc_point.append(tamc_blend_forecast[0])

        for name, forecast in forecasts.items():
            errors[name].append((mae(target, forecast), rmse(target, forecast)))

    times_arr = np.array(times)
    pre_mask = times_arr < shift_index
    post_mask = ~pre_mask

    rows = []
    for name in VARIANT_NAMES:
        values = np.array(errors[name])
        mae_values, rmse_values = values[:, 0], values[:, 1]
        for segment_name, mask in [("pre-shift", pre_mask), ("post-shift", post_mask)]:
            rows.append(
                {
                    "Variant": name,
                    "Segment": segment_name,
                    "MAE": float(mae_values[mask].mean()),
                    "RMSE": float(rmse_values[mask].mean()),
                }
            )
    metrics_table = pd.DataFrame(rows)

    raw = {
        "series": series,
        "shift_index": shift_index,
        "times": times_arr,
        "gates": np.array(gates),
        "frozen_point": np.array(frozen_point),
        "tamc_point": np.array(tamc_point),
    }
    return metrics_table, raw


def plot_results(raw: dict, shift_type: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    series = raw["series"]
    shift_index = raw["shift_index"]
    times = raw["times"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(np.arange(len(series)), series, alpha=0.5, label="True series")
    axes[0].plot(times, raw["frozen_point"], "--", label="Frozen forecast")
    axes[0].plot(times, raw["tamc_point"], "--", label="TAMC-gated blend forecast")
    axes[0].axvline(shift_index, linestyle=":", color="black")
    axes[0].set_ylabel("Value (causally normalized)")
    axes[0].set_title(f"Real-data controlled shift: {shift_type}")
    axes[0].legend(loc="upper left")

    axes[1].plot(times, raw["gates"], color="tab:red")
    axes[1].axvline(shift_index, linestyle=":", color="black")
    axes[1].set_ylabel("TAMC gate")
    axes[1].set_xlabel("Time index")
    axes[1].set_ylim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def run_single_seed(
    base_series: np.ndarray, shift_index: int, args, figures_dir: Path
) -> None:
    metrics_table, raw = run_experiment(
        base_series,
        shift_index,
        shift_type=args.shift_type,
        seed=args.seed,
        context_length=args.context_length,
        horizon=args.horizon,
        window=args.window,
        stride=args.stride,
        gate_threshold=args.gate_threshold,
        amplitude_factor=args.amplitude_factor,
        trend_magnitude=args.trend_magnitude,
        noise_std=args.noise_std,
        frequency_factor=args.frequency_factor,
    )
    tradeoff_table = compute_tradeoff_metrics(metrics_table)

    tradeoff_columns = [
        "Variant",
        "Pre Harm",
        "Post Gain",
        "Net Adaptation Score",
        "Post Improvement %",
    ]
    merged_metrics_table = metrics_table.merge(
        tradeoff_table[tradeoff_columns], on="Variant", how="left"
    )

    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figures_dir / "real_data_controlled_shift.png"
    plot_results(raw, args.shift_type, figure_path)

    metrics_path = figures_dir / "real_data_controlled_shift_metrics.csv"
    merged_metrics_table.to_csv(metrics_path, index=False)

    print(f"Saved figure to {figure_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Shift index: {raw['shift_index']} (shift type: {args.shift_type})")
    print(
        "\nForecast error by variant and segment (pre-shift is in-sample for the AR fit):"
    )
    print(
        merged_metrics_table.to_string(index=False, float_format=lambda v: f"{v:.4f}")
    )
    print("\nAdaptation tradeoff vs frozen forecaster:")
    print(tradeoff_table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def run_multi_seed(
    base_series: np.ndarray, shift_index: int, args, figures_dir: Path
) -> None:
    tables = []
    tradeoff_tables = []
    for seed in range(args.multi_seed):
        metrics_table, _ = run_experiment(
            base_series,
            shift_index,
            shift_type=args.shift_type,
            seed=seed,
            context_length=args.context_length,
            horizon=args.horizon,
            window=args.window,
            stride=args.stride,
            gate_threshold=args.gate_threshold,
            amplitude_factor=args.amplitude_factor,
            trend_magnitude=args.trend_magnitude,
            noise_std=args.noise_std,
            frequency_factor=args.frequency_factor,
        )
        tables.append(metrics_table.assign(Seed=seed))
        tradeoff_tables.append(
            compute_tradeoff_metrics(metrics_table).assign(Seed=seed)
        )

    combined = pd.concat(tables)
    summary = combined.groupby(["Variant", "Segment"])[["MAE", "RMSE"]].agg(
        ["mean", "std"]
    )
    summary.columns = [f"{metric} {stat}" for metric, stat in summary.columns]

    tradeoff_combined = pd.concat(tradeoff_tables)
    tradeoff_value_columns = [
        "Pre-shift MAE",
        "Post-shift MAE",
        "Pre Harm",
        "Post Gain",
        "Net Adaptation Score",
        "Post Improvement %",
    ]
    tradeoff_summary = tradeoff_combined.groupby("Variant")[tradeoff_value_columns].agg(
        ["mean", "std"]
    )
    tradeoff_summary.columns = [
        f"{metric} {stat}" for metric, stat in tradeoff_summary.columns
    ]

    tradeoff_csv_columns = [
        "Pre Harm mean",
        "Pre Harm std",
        "Post Gain mean",
        "Post Gain std",
        "Net Adaptation Score mean",
        "Net Adaptation Score std",
        "Post Improvement % mean",
        "Post Improvement % std",
    ]
    merged_summary = summary.join(tradeoff_summary[tradeoff_csv_columns], on="Variant")

    figures_dir.mkdir(parents=True, exist_ok=True)
    summary_path = figures_dir / "real_data_controlled_shift_multiseed_metrics.csv"
    merged_summary.to_csv(summary_path)

    print(
        f"Ran {args.multi_seed} seeds (0..{args.multi_seed - 1}), shift type: {args.shift_type}"
    )
    print(f"Saved multi-seed summary to {summary_path}")
    print("\nMulti-seed forecast error by variant and segment (mean +/- std):")
    print(merged_summary.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nAdaptation tradeoff vs frozen forecaster (mean +/- std across seeds):")
    print(tradeoff_summary.to_string(float_format=lambda v: f"{v:.4f}"))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-path", required=True, help="Path to a local CSV file.")
    parser.add_argument(
        "--value-column", required=True, help="Column to use as the univariate series."
    )
    parser.add_argument(
        "--timestamp-column",
        default=None,
        help="Column to sort by before selecting --value-column (optional).",
    )
    parser.add_argument(
        "--shift-type",
        required=True,
        choices=SHIFT_TYPES,
        help="Controlled perturbation to inject into the post-shift half.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Single-seed run (default: 0)."
    )
    parser.add_argument(
        "--multi-seed",
        type=int,
        default=None,
        metavar="N",
        help="Run seeds 0..N-1 and report mean +/- std instead of a single-seed figure run.",
    )
    parser.add_argument("--n-points", type=int, default=3000)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--gate-threshold", type=float, default=2.0)
    parser.add_argument("--amplitude-factor", type=float, default=1.35)
    parser.add_argument("--trend-magnitude", type=float, default=1.0)
    parser.add_argument("--noise-std", type=float, default=0.35)
    parser.add_argument("--frequency-factor", type=float, default=1.3)
    args = parser.parse_args()

    base_series, shift_index = load_and_normalize_series(
        args.csv_path, args.value_column, args.timestamp_column, args.n_points
    )

    figures_dir = Path(__file__).resolve().parents[1] / "figures"

    if args.multi_seed is not None:
        run_multi_seed(base_series, shift_index, args, figures_dir)
    else:
        run_single_seed(base_series, shift_index, args, figures_dir)


if __name__ == "__main__":
    main()
