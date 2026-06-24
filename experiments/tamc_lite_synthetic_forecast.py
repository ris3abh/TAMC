"""TAMC-Lite end-to-end forecasting adaptation demo.

Goal:
    Show that a topology-gated residual adapter (TAMC-Lite) can recover
    forecast accuracy after a regime shift that degrades a frozen
    forecaster, without corrupting pre-shift performance the way an
    always-on residual adapter does.

Compares three forecasting strategies on the same sine -> quasi-periodic
regime shift used in synthetic_regime_shift.py:
    1. frozen forecaster only (no adaptation)
    2. always-on residual adapter (gate = 1 at every step)
    3. TAMC-gated residual adapter (gate driven by topological drift)

The forecaster (LinearARForecaster) is fit once on source/pre-shift data
only and never updated; only the residual correction and its gate change
at inference time (forward-only, no gradients).

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

from adapters import MeanShiftResidual
from delay_embedding import EmbeddingConfig
from evaluation import mae, rmse
from forecasting import LinearARForecaster
from tamc_pipeline import TamicLitePipeline
from tamic_signal import TamicSignal

from synthetic_regime_shift import generate_sine_to_quasiperiodic

CONTEXT_LENGTH = 32
HORIZON = 8
TOPOLOGY_WINDOW = 128
STRIDE = 4


def run_experiment(seed: int) -> tuple[pd.DataFrame, dict]:
    """Run one full seed; return the per-segment metrics table and raw arrays."""
    series, shift_index = generate_sine_to_quasiperiodic(seed=seed)
    source_series = series[:shift_index]

    forecaster = LinearARForecaster(context_length=CONTEXT_LENGTH, horizon=HORIZON)
    forecaster.fit(source_series, context_length=CONTEXT_LENGTH, horizon=HORIZON)

    config = EmbeddingConfig(dimension=3, delay=8, window=TOPOLOGY_WINDOW)
    signal = TamicSignal(config=config, max_dimension=1, drift_dimension=1)
    signal.add_source_prototype(
        source_series[:TOPOLOGY_WINDOW], label="periodic_source"
    )

    residual = MeanShiftResidual(
        source_mean=float(source_series.mean()), horizon=HORIZON
    )

    pipeline = TamicLitePipeline(
        forecaster=forecaster,
        tamic_signal=signal,
        residual_adapter=residual,
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
    tamc_point = []
    frozen_mae, frozen_rmse_ = [], []
    always_on_mae, always_on_rmse_ = [], []
    tamc_mae, tamc_rmse_ = [], []

    for t in range(start_index, end_index, STRIDE):
        context = series[t - CONTEXT_LENGTH : t]
        topology_window_values = series[t - TOPOLOGY_WINDOW : t]
        target = series[t : t + HORIZON]

        result = pipeline.predict(context, topology_window_values)
        base_forecast = result["base_forecast"]
        adapted_forecast = result["adapted_forecast"]
        always_on_forecast = base_forecast + residual.correction(context)

        times.append(t)
        gates.append(result["gate"])
        distances.append(result["topological_distance"])
        frozen_point.append(base_forecast[0])
        tamc_point.append(adapted_forecast[0])

        frozen_mae.append(mae(target, base_forecast))
        frozen_rmse_.append(rmse(target, base_forecast))
        always_on_mae.append(mae(target, always_on_forecast))
        always_on_rmse_.append(rmse(target, always_on_forecast))
        tamc_mae.append(mae(target, adapted_forecast))
        tamc_rmse_.append(rmse(target, adapted_forecast))

    times = np.array(times)
    pre_mask = times < shift_index
    post_mask = ~pre_mask

    variants = {
        "Frozen forecaster": (np.array(frozen_mae), np.array(frozen_rmse_)),
        "Always-on residual adapter": (
            np.array(always_on_mae),
            np.array(always_on_rmse_),
        ),
        "TAMC-gated residual adapter": (np.array(tamc_mae), np.array(tamc_rmse_)),
    }

    rows = []
    for variant_name, (mae_values, rmse_values) in variants.items():
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
        "tamc_point": np.array(tamc_point),
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
    axes[0].plot(times, raw["tamc_point"], "--", label="TAMC adapted forecast")
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
