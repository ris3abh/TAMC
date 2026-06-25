"""Conference-scale benchmark: TAMC vs. RG-style regime-similarity control.

Goal:
    First step toward a stronger, benchmarked comparison: run TAMC's
    topological gate against a transparent RG-style (Regime-Guided)
    statistical/distributional regime-similarity gate -- plus the existing
    mean/variance, autocorrelation, and spectral gates -- under an
    identical frozen/adaptive forecast-blending framework, across multiple
    real datasets with controlled injected shifts. This is benchmark
    *infrastructure*: it establishes the comparison harness and runs
    whatever local datasets are available, not a claim that TAMC beats
    RG-style control across the board.

No RG-TTA code is used or copied; `src/regime_similarity.py` implements
regime similarity from first principles (see that module's docstring).

All gated blends use the *same* adaptive forecaster (`RecentPatternForecaster`)
so the comparison isolates the gate/control signal, not the adaptive model.

Causal protocol (same as `real_data_controlled_shift.py`):
    - Normalization is fit on the source/pre-shift half only.
    - The shift is injected into the post-shift half only.
    - The frozen forecaster and every gate's source reference are built
      only from the source segment.
    - No future labels are used anywhere.

Run:
    uv run python experiments/benchmark_regime_control.py \\
        --datasets ETTh1 --shift-type seasonality_break --seed 0

    uv run python experiments/benchmark_regime_control.py \\
        --datasets ETTh1 ETTh2 ETTm1 ETTm2 Weather \\
        --shift-type seasonality_break --multi-seed 3
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from delay_embedding import EmbeddingConfig
from drift_gates import ScalarDriftSignal
from evaluation import mae, rmse
from forecasting import LinearARForecaster, RecentPatternForecaster
from regime_similarity import RegimeSimilaritySignal
from tamc_pipeline import TamicBlendPipeline
from tamic_signal import TamicSignal

from real_data_controlled_shift import (
    SHIFT_TYPES,
    inject_shift,
    load_and_normalize_series,
)
from synthetic_regime_shift import (
    autocorrelation_signature,
    l2_distance,
    spectral_signature,
)
from tamc_lite_synthetic_forecast import compute_tradeoff_metrics

DATASET_FILENAMES = {
    "ETTh1": "ETTh1.csv",
    "ETTh2": "ETTh2.csv",
    "ETTm1": "ETTm1.csv",
    "ETTm2": "ETTm2.csv",
    "Weather": "weather.csv",
}
DEFAULT_DATASETS = tuple(DATASET_FILENAMES.keys())
TIMESTAMP_COLUMN = "date"

VARIANT_NAMES = (
    "Frozen forecaster",
    "Adaptive recent-pattern forecaster",
    "Always-on 50/50 blend",
    "TAMC-gated blend",
    "RG-style regime-similarity-gated blend",
    "Mean/variance-gated blend",
    "Autocorrelation-gated blend",
    "Spectral-gated blend",
)


def resolve_value_column(
    csv_path: Path, timestamp_column: str, override: str | None
) -> str | None:
    """Pick the value column: an explicit override, else 'OT' if present,
    else the first numeric non-timestamp column. Returns None if no usable
    column is found.
    """
    if override is not None:
        return override

    header = pd.read_csv(csv_path, nrows=64)
    if "OT" in header.columns:
        return "OT"
    for column in header.columns:
        if column == timestamp_column:
            continue
        if pd.api.types.is_numeric_dtype(header[column]):
            return column
    return None


def run_experiment(
    dataset: str,
    csv_path: Path,
    value_column_override: str | None,
    shift_type: str,
    seed: int,
    n_points: int,
    context_length: int,
    horizon: int,
    window: int,
    stride: int,
    delay: int,
    drift_dimension: int,
    gate_threshold: float,
    min_history: int,
    embedding_dimension: int = 3,
) -> tuple[pd.DataFrame, dict] | None:
    """Run one full seed for one dataset; return (metrics_table, raw), or
    None if the dataset/column is unavailable (skip, do not raise).
    """
    if not csv_path.exists():
        print(f"  SKIP: {dataset} -- file not found: {csv_path}")
        return None

    value_column = resolve_value_column(
        csv_path, TIMESTAMP_COLUMN, value_column_override
    )
    if value_column is None:
        print(
            f"  SKIP: {dataset} -- no usable value column found "
            f"(looked for 'OT' or first numeric non-timestamp column)."
        )
        return None

    try:
        base_series, shift_index = load_and_normalize_series(
            str(csv_path), value_column, TIMESTAMP_COLUMN, n_points
        )
    except (ValueError, KeyError) as exc:
        print(f"  SKIP: {dataset} -- failed to load column '{value_column}': {exc}")
        return None

    if len(base_series) < 2 * max(context_length, window) + horizon:
        print(
            f"  SKIP: {dataset} -- series too short ({len(base_series)} points) "
            f"for context_length={context_length}, window={window}, horizon={horizon}."
        )
        return None

    post = base_series[shift_index:]
    shifted_post = inject_shift(post, shift_type, seed)
    series = np.concatenate([base_series[:shift_index], shifted_post])
    source_series = series[:shift_index]

    frozen_forecaster = LinearARForecaster(
        context_length=context_length, horizon=horizon
    )
    frozen_forecaster.fit(source_series, context_length=context_length, horizon=horizon)

    adaptive_forecaster = RecentPatternForecaster(horizon=horizon)

    config = EmbeddingConfig(dimension=embedding_dimension, delay=delay, window=window)
    tamc_signal = TamicSignal(
        config=config, max_dimension=1, drift_dimension=drift_dimension
    )
    tamc_signal.add_source_prototype(source_series[:window], label="source")

    blend_pipeline = TamicBlendPipeline(
        frozen_forecaster=frozen_forecaster,
        adaptive_forecaster=adaptive_forecaster,
        tamic_signal=tamc_signal,
        context_length=context_length,
        topology_window=window,
        horizon=horizon,
        gate_threshold=gate_threshold,
        min_history=min_history,
    )

    rg_signal = RegimeSimilaritySignal(source_window=series[:window])

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
    tamc_gates: list[float] = []
    rg_gates: list[float] = []
    frozen_point: list[float] = []
    tamc_point: list[float] = []
    rg_point: list[float] = []
    errors: dict[str, list[tuple[float, float]]] = {name: [] for name in VARIANT_NAMES}

    for t in range(start_index, end_index, stride):
        context = series[t - context_length : t]
        topology_window_values = series[t - window : t]
        target = series[t : t + horizon]

        # One pipeline call advances TAMC's topology-monitor history and
        # gives the frozen/adaptive forecasts plus TAMC's gate; every
        # other gate below is scored exactly once per step too.
        result = blend_pipeline.predict(context, topology_window_values)
        frozen_forecast = result["frozen_forecast"]
        adaptive_forecast = result["adaptive_forecast"]
        tamc_gate = result["gate"]
        tamc_blend = result["blended_forecast"]

        rg_gate = rg_signal.gate(
            topology_window_values, threshold=gate_threshold, min_history=min_history
        )
        rg_blend = (1 - rg_gate) * frozen_forecast + rg_gate * adaptive_forecast

        always_on = 0.5 * frozen_forecast + 0.5 * adaptive_forecast

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
            "Always-on 50/50 blend": always_on,
            "TAMC-gated blend": tamc_blend,
            "RG-style regime-similarity-gated blend": rg_blend,
            "Mean/variance-gated blend": (1 - meanvar_gate) * frozen_forecast
            + meanvar_gate * adaptive_forecast,
            "Autocorrelation-gated blend": (1 - acf_gate) * frozen_forecast
            + acf_gate * adaptive_forecast,
            "Spectral-gated blend": (1 - spec_gate) * frozen_forecast
            + spec_gate * adaptive_forecast,
        }

        times.append(t)
        tamc_gates.append(tamc_gate)
        rg_gates.append(rg_gate)
        frozen_point.append(frozen_forecast[0])
        tamc_point.append(tamc_blend[0])
        rg_point.append(rg_blend[0])

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
        "tamc_gates": np.array(tamc_gates),
        "rg_gates": np.array(rg_gates),
        "frozen_point": np.array(frozen_point),
        "tamc_point": np.array(tamc_point),
        "rg_point": np.array(rg_point),
        "value_column": value_column,
    }
    return metrics_table, raw


def plot_results(raw: dict, dataset: str, shift_type: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    series = raw["series"]
    shift_index = raw["shift_index"]
    times = raw["times"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(np.arange(len(series)), series, alpha=0.5, label="True series")
    axes[0].plot(times, raw["frozen_point"], "--", label="Frozen forecast")
    axes[0].plot(times, raw["tamc_point"], "--", label="TAMC-gated forecast")
    axes[0].plot(times, raw["rg_point"], "--", label="RG-style-gated forecast")
    axes[0].axvline(shift_index, linestyle=":", color="black")
    axes[0].set_ylabel("Value (causally normalized)")
    axes[0].set_title(f"Benchmark: {dataset} ({shift_type})")
    axes[0].legend(loc="upper left", fontsize=8)

    axes[1].plot(times, raw["tamc_gates"], color="tab:blue", label="TAMC gate")
    axes[1].plot(times, raw["rg_gates"], color="tab:orange", label="RG-style gate")
    axes[1].axvline(shift_index, linestyle=":", color="black")
    axes[1].set_ylabel("Gate")
    axes[1].set_xlabel("Time index")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def run_single_seed(args, figures_dir: Path) -> None:
    for dataset in args.datasets:
        csv_path = Path(args.data_dir) / DATASET_FILENAMES[dataset]
        print(f"\n=== {dataset} (seed={args.seed}) ===")
        result = run_experiment(
            dataset=dataset,
            csv_path=csv_path,
            value_column_override=args.value_column,
            shift_type=args.shift_type,
            seed=args.seed,
            n_points=args.n_points,
            context_length=args.context_length,
            horizon=args.horizon,
            window=args.window,
            stride=args.stride,
            delay=args.delay,
            drift_dimension=args.drift_dimension,
            gate_threshold=args.gate_threshold,
            min_history=args.min_history,
        )
        if result is None:
            continue
        metrics_table, raw = result

        tradeoff_table = compute_tradeoff_metrics(metrics_table)
        tradeoff_columns = [
            "Variant",
            "Pre Harm",
            "Post Gain",
            "Net Adaptation Score",
            "Post Improvement %",
        ]
        merged = metrics_table.merge(
            tradeoff_table[tradeoff_columns], on="Variant", how="left"
        )

        figures_dir.mkdir(parents=True, exist_ok=True)
        figure_path = (
            figures_dir / f"benchmark_regime_control_{dataset}_{args.shift_type}.png"
        )
        plot_results(raw, dataset, args.shift_type, figure_path)
        print(f"Saved figure to {figure_path}")
        print(merged.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def run_multi_seed(args, figures_dir: Path) -> None:
    tables = []
    for dataset in args.datasets:
        csv_path = Path(args.data_dir) / DATASET_FILENAMES[dataset]
        for seed in range(args.multi_seed):
            result = run_experiment(
                dataset=dataset,
                csv_path=csv_path,
                value_column_override=args.value_column,
                shift_type=args.shift_type,
                seed=seed,
                n_points=args.n_points,
                context_length=args.context_length,
                horizon=args.horizon,
                window=args.window,
                stride=args.stride,
                delay=args.delay,
                drift_dimension=args.drift_dimension,
                gate_threshold=args.gate_threshold,
                min_history=args.min_history,
            )
            if result is None:
                break  # this dataset is unavailable; no point retrying other seeds
            metrics_table, _ = result
            tradeoff_table = compute_tradeoff_metrics(metrics_table)
            tradeoff_columns = [
                "Variant",
                "Pre Harm",
                "Post Gain",
                "Net Adaptation Score",
                "Post Improvement %",
            ]
            merged = metrics_table.merge(
                tradeoff_table[tradeoff_columns], on="Variant", how="left"
            )
            merged.insert(0, "Dataset", dataset)
            merged.insert(1, "Shift Type", args.shift_type)
            merged.insert(2, "Seed", seed)
            tables.append(merged)
            print(f"[{dataset}] seed {seed} done", flush=True)

    if not tables:
        print("\nNo datasets produced results; nothing to summarize.")
        return

    combined = pd.concat(tables, ignore_index=True)

    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = figures_dir / "benchmark_regime_control_metrics.csv"
    combined.to_csv(metrics_path, index=False)
    print(f"\nSaved long-format metrics to {metrics_path}")

    summary_columns = [
        "MAE",
        "RMSE",
        "Pre Harm",
        "Post Gain",
        "Net Adaptation Score",
        "Post Improvement %",
    ]
    summary = combined.groupby(["Dataset", "Shift Type", "Variant", "Segment"])[
        summary_columns
    ].agg(["mean", "std"])
    summary.columns = [f"{metric} {stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary_path = figures_dir / "benchmark_regime_control_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")

    tradeoff_columns_only = [
        "Pre Harm",
        "Post Gain",
        "Net Adaptation Score",
        "Post Improvement %",
    ]
    # Tradeoff metrics are identical across both segment rows for a given
    # (Dataset, Shift Type, Seed, Variant); de-duplicate before aggregating
    # so the across-seed std is not biased (see other benchmark scripts).
    tradeoff_compact = combined.drop_duplicates(
        subset=["Dataset", "Shift Type", "Seed", "Variant"]
    )
    tradeoff_summary = tradeoff_compact.groupby(["Dataset", "Shift Type", "Variant"])[
        tradeoff_columns_only
    ].agg(["mean", "std"])
    tradeoff_summary.columns = [
        f"{metric} {stat}" for metric, stat in tradeoff_summary.columns
    ]
    tradeoff_summary = tradeoff_summary.reset_index()
    tradeoff_path = figures_dir / "benchmark_regime_control_tradeoff_summary.csv"
    tradeoff_summary.to_csv(tradeoff_path, index=False)
    print(f"Saved tradeoff summary to {tradeoff_path}")

    print_interpretation(tradeoff_summary)


def print_interpretation(tradeoff_summary: pd.DataFrame) -> None:
    print("\n=== Honest interpretation ===")
    for dataset in tradeoff_summary["Dataset"].unique():
        subset = tradeoff_summary[tradeoff_summary["Dataset"] == dataset]
        best_row = subset.loc[subset["Net Adaptation Score mean"].idxmax()]

        def _score(variant: str) -> float | None:
            rows = subset[subset["Variant"] == variant]
            return (
                float(rows["Net Adaptation Score mean"].iloc[0])
                if not rows.empty
                else None
            )

        tamc_score = _score("TAMC-gated blend")
        rg_score = _score("RG-style regime-similarity-gated blend")
        acf_score = _score("Autocorrelation-gated blend")
        spec_score = _score("Spectral-gated blend")
        meanvar_score = _score("Mean/variance-gated blend")
        always_on_score = _score("Always-on 50/50 blend")

        print(f"\n[{dataset}]")
        print(
            f"  Best variant by mean Net Adaptation Score: "
            f"{best_row['Variant']} ({best_row['Net Adaptation Score mean']:.4f})"
        )
        print(f"  TAMC-gated:              {tamc_score:.4f}")
        print(f"  RG-style regime-similarity-gated: {rg_score:.4f}")
        print(f"  Autocorrelation-gated:   {acf_score:.4f}")
        print(f"  Spectral-gated:          {spec_score:.4f}")
        print(f"  Mean/variance-gated:     {meanvar_score:.4f}")
        print(f"  Always-on:               {always_on_score:.4f}")

        if tamc_score is not None and rg_score is not None:
            if tamc_score > rg_score:
                print(
                    f"  TAMC beats the RG-style gate on {dataset} "
                    f"({tamc_score:.4f} > {rg_score:.4f})."
                )
            elif tamc_score < rg_score:
                print(
                    f"  HONEST NOTE: TAMC does NOT beat the RG-style gate on "
                    f"{dataset} ({tamc_score:.4f} < {rg_score:.4f})."
                )
            else:
                print(f"  TAMC ties the RG-style gate on {dataset}.")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        choices=list(DATASET_FILENAMES.keys()),
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--shift-type", default="seasonality_break", choices=list(SHIFT_TYPES)
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--multi-seed", type=int, default=None, metavar="N")
    parser.add_argument("--n-points", type=int, default=3000)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--delay", type=int, default=8)
    parser.add_argument("--drift-dimension", type=int, default=0, choices=[0, 1])
    parser.add_argument("--gate-threshold", type=float, default=2.0)
    parser.add_argument("--min-history", type=int, default=8)
    parser.add_argument(
        "--value-column",
        default=None,
        help="Override the value column for every selected dataset (default: "
        "per-dataset 'OT', or auto-detected first numeric non-timestamp column).",
    )
    args = parser.parse_args()

    figures_dir = Path(__file__).resolve().parents[1] / "figures"

    if args.multi_seed is not None:
        run_multi_seed(args, figures_dir)
    else:
        run_single_seed(args, figures_dir)


if __name__ == "__main__":
    main()
