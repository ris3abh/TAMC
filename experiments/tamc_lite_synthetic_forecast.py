"""TAMC-Lite end-to-end forecasting adaptation demo.

Goal:
    Show that a topology-gated residual adapter (TAMC-Lite) can recover
    forecast accuracy after a regime shift that degrades a frozen
    forecaster, without corrupting pre-shift performance the way an
    always-on residual adapter does.

Compares eight forecasting strategies on the same sine -> quasi-periodic
regime shift used in synthetic_regime_shift.py. The main comparison is the
topology-gated forecast blend:
    1. frozen forecaster alone (no adaptation)
    2. adaptive recent-pattern forecaster alone
    3. always-on 50/50 blend of frozen and adaptive
    4. TAMC-gated blend (gate driven by topological drift)
Plus the earlier residual-correction variants, kept for context:
    5. always-on MeanShiftResidual
    6. TAMC-gated MeanShiftResidual
    7. always-on AnalogResidualAdapter
    8. TAMC-gated AnalogResidualAdapter

The frozen forecaster (LinearARForecaster) is fit once on source/pre-shift
data only and never updated; the AnalogResidualAdapter's source-memory bank
is likewise built only from source/pre-shift supervised windows. The
adaptive forecaster (RecentPatternForecaster) only ever sees the current
context: no labels, no future data, no gradient updates, at any point.

Run:
    uv run python experiments/tamc_lite_synthetic_forecast.py --seed 0
    uv run python experiments/tamc_lite_synthetic_forecast.py --multi-seed 10
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from adapters import AnalogResidualAdapter, MeanShiftResidual
from delay_embedding import EmbeddingConfig
from evaluation import mae, make_supervised_windows, rmse
from forecasting import LinearARForecaster, RecentPatternForecaster
from tamc_pipeline import TamicBlendPipeline
from tamic_signal import TamicSignal

from synthetic_regime_shift import generate_sine_to_quasiperiodic

# CONTEXT_LENGTH matches TOPOLOGY_WINDOW so the adaptive forecaster's
# autocorrelation-based lag search has enough room to resolve the signal's
# ~50-sample period; a much shorter context cannot detect that lag at all.
CONTEXT_LENGTH = 128
HORIZON = 8
TOPOLOGY_WINDOW = 128
STRIDE = 4
ANALOG_K = 5
ANALOG_TEMPERATURE = 1.0
ADAPTIVE_MIN_LAG = 4


def run_experiment(seed: int) -> tuple[pd.DataFrame, dict]:
    """Run one full seed; return the per-segment metrics table and raw arrays."""
    series, shift_index = generate_sine_to_quasiperiodic(seed=seed)
    source_series = series[:shift_index]

    forecaster = LinearARForecaster(context_length=CONTEXT_LENGTH, horizon=HORIZON)
    forecaster.fit(source_series, context_length=CONTEXT_LENGTH, horizon=HORIZON)

    adaptive_forecaster = RecentPatternForecaster(
        horizon=HORIZON, min_lag=ADAPTIVE_MIN_LAG
    )

    config = EmbeddingConfig(dimension=3, delay=8, window=TOPOLOGY_WINDOW)
    signal = TamicSignal(config=config, max_dimension=1, drift_dimension=1)
    signal.add_source_prototype(
        source_series[:TOPOLOGY_WINDOW], label="periodic_source"
    )

    mean_shift_residual = MeanShiftResidual(
        source_mean=float(source_series.mean()), horizon=HORIZON
    )

    # AnalogResidualAdapter is fit only on source/pre-shift supervised
    # windows; the frozen forecaster's own predictions on those windows
    # define the residuals it memorizes, so no post-shift labels are used.
    source_contexts, source_targets = make_supervised_windows(
        source_series, context_length=CONTEXT_LENGTH, horizon=HORIZON, stride=1
    )
    source_base_predictions = np.stack([forecaster.predict(c) for c in source_contexts])
    analog_residual = AnalogResidualAdapter(
        horizon=HORIZON, k=ANALOG_K, temperature=ANALOG_TEMPERATURE
    )
    analog_residual.fit(source_contexts, source_targets, source_base_predictions)

    blend_pipeline = TamicBlendPipeline(
        frozen_forecaster=forecaster,
        adaptive_forecaster=adaptive_forecaster,
        tamic_signal=signal,
        context_length=CONTEXT_LENGTH,
        topology_window=TOPOLOGY_WINDOW,
        horizon=HORIZON,
    )

    start_index = TOPOLOGY_WINDOW
    end_index = len(series) - HORIZON

    times = []
    gates = []
    distances = []
    frozen_point = []
    adaptive_point = []
    tamc_blend_point = []
    errors: dict[str, list[tuple[float, float]]] = {
        "Frozen forecaster": [],
        "Adaptive recent-pattern forecaster": [],
        "Always-on 50/50 blend": [],
        "TAMC-gated blend": [],
        "Always-on MeanShiftResidual": [],
        "TAMC-gated MeanShiftResidual": [],
        "Always-on AnalogResidualAdapter": [],
        "TAMC-gated AnalogResidualAdapter": [],
    }

    for t in range(start_index, end_index, STRIDE):
        context = series[t - CONTEXT_LENGTH : t]
        topology_window_values = series[t - TOPOLOGY_WINDOW : t]
        target = series[t : t + HORIZON]

        # One pipeline call advances the shared topology-monitor history and
        # gives the frozen/adaptive forecasts and the gate; the gate is
        # reused for the residual variants below so topology is only ever
        # scored once per step.
        result = blend_pipeline.predict(context, topology_window_values)
        frozen_forecast = result["frozen_forecast"]
        adaptive_forecast = result["adaptive_forecast"]
        gate = result["gate"]
        tamc_blend_forecast = result["blended_forecast"]
        always_on_blend_forecast = 0.5 * frozen_forecast + 0.5 * adaptive_forecast

        mean_shift_correction = mean_shift_residual.correction(context)
        analog_correction = analog_residual.correction(context)

        forecasts = {
            "Frozen forecaster": frozen_forecast,
            "Adaptive recent-pattern forecaster": adaptive_forecast,
            "Always-on 50/50 blend": always_on_blend_forecast,
            "TAMC-gated blend": tamc_blend_forecast,
            "Always-on MeanShiftResidual": frozen_forecast + mean_shift_correction,
            "TAMC-gated MeanShiftResidual": frozen_forecast
            + gate * mean_shift_correction,
            "Always-on AnalogResidualAdapter": frozen_forecast + analog_correction,
            "TAMC-gated AnalogResidualAdapter": frozen_forecast
            + gate * analog_correction,
        }

        times.append(t)
        gates.append(gate)
        distances.append(result["topological_distance"])
        frozen_point.append(frozen_forecast[0])
        adaptive_point.append(adaptive_forecast[0])
        tamc_blend_point.append(tamc_blend_forecast[0])

        for variant_name, forecast in forecasts.items():
            errors[variant_name].append((mae(target, forecast), rmse(target, forecast)))

    times = np.array(times)
    pre_mask = times < shift_index
    post_mask = ~pre_mask

    rows = []
    for variant_name, values in errors.items():
        values_array = np.array(values)
        mae_values, rmse_values = values_array[:, 0], values_array[:, 1]
        for segment_name, mask in [("pre-shift", pre_mask), ("post-shift", post_mask)]:
            rows.append(
                {
                    "Variant": variant_name,
                    "Segment": segment_name,
                    "MAE": float(mae_values[mask].mean()),
                    "RMSE": float(rmse_values[mask].mean()),
                }
            )
    metrics_table = pd.DataFrame(rows)

    raw = {
        "series": series,
        "shift_index": shift_index,
        "times": times,
        "gates": np.array(gates),
        "distances": np.array(distances),
        "frozen_point": np.array(frozen_point),
        "adaptive_point": np.array(adaptive_point),
        "tamc_point": np.array(tamc_blend_point),
    }
    return metrics_table, raw


def plot_results(raw: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    series = raw["series"]
    shift_index = raw["shift_index"]
    times = raw["times"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(np.arange(len(series)), series, alpha=0.5, label="True series")
    axes[0].plot(times, raw["frozen_point"], "--", label="Frozen forecast")
    axes[0].plot(
        times, raw["adaptive_point"], "--", label="Adaptive recent-pattern forecast"
    )
    axes[0].plot(times, raw["tamc_point"], "--", label="TAMC-gated blend forecast")
    axes[0].axvline(shift_index, linestyle=":", color="black")
    axes[0].set_ylabel("Value")
    axes[0].set_title("TAMC-Lite forecasting adaptation: sine to quasi-periodic shift")
    axes[0].legend(loc="upper left")

    axes[1].plot(times, raw["gates"], color="tab:red")
    axes[1].axvline(shift_index, linestyle=":", color="black")
    axes[1].set_ylabel("Topology gate")
    axes[1].set_xlabel("Time index")
    axes[1].set_ylim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def run_single_seed(seed: int, figures_dir: Path) -> None:
    metrics_table, raw = run_experiment(seed=seed)

    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figures_dir / "tamc_lite_synthetic_forecast.png"
    plot_results(raw, figure_path)

    metrics_path = figures_dir / "tamc_lite_synthetic_forecast_metrics.csv"
    metrics_table.to_csv(metrics_path, index=False)

    print(f"Saved figure to {figure_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Shift index: {raw['shift_index']}")
    print(
        "\nForecast error by variant and segment (pre-shift is in-sample for the AR fit):"
    )
    print(metrics_table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def run_multi_seed(n_seeds: int, figures_dir: Path) -> None:
    tables = []
    for seed in range(n_seeds):
        metrics_table, _ = run_experiment(seed=seed)
        tables.append(metrics_table.assign(Seed=seed))

    combined = pd.concat(tables)
    summary = combined.groupby(["Variant", "Segment"])[["MAE", "RMSE"]].agg(
        ["mean", "std"]
    )
    summary.columns = [f"{metric} {stat}" for metric, stat in summary.columns]

    figures_dir.mkdir(parents=True, exist_ok=True)
    summary_path = figures_dir / "tamc_lite_synthetic_forecast_multiseed_metrics.csv"
    summary.to_csv(summary_path)

    print(f"Ran {n_seeds} seeds (0..{n_seeds - 1})")
    print(f"Saved multi-seed summary to {summary_path}")
    print("\nMulti-seed forecast error by variant and segment (mean +/- std):")
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
        help="Run seeds 0..N-1 and report mean +/- std forecast errors instead "
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
