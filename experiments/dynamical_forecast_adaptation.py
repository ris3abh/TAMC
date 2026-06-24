"""Dynamical-system forecast-adaptation experiment for TAMC-Lite.

Goal:
    The forecast-adaptation evidence so far (`tamc_lite_synthetic_forecast.py`,
    `real_data_controlled_shift.py`) comes from one synthetic shift (sine to
    quasi-periodic) and one real series under controlled injection. TAMC
    *detection* already works on three controlled dynamical systems (sine,
    logistic map, Lorenz); this experiment asks whether topology-gated
    forecast *adaptation* generalizes the same way, or whether the existing
    adaptation evidence is mostly limited to the sine/quasi-periodic case.
    This is a limitation-removal experiment, not a leaderboard: every
    variant is evaluated and reported as-is, including weak or negative
    results, and no gate/adaptive-model choice is made by looking at
    post-shift performance.

Design decision -- "best adaptive forecaster" for non-topological gates:
    Per-system validation-based selection of the adaptive forecaster for
    the non-topological gates would need its own held-out split of the
    source regime and its own selection criterion -- a second judgment
    call on top of the one this experiment is trying to avoid. Per the
    task spec's explicit fallback, we instead fix `RollingLinearARForecaster`
    as the adaptive forecaster for the three non-topological gates
    (mean/variance, autocorrelation, spectral) on every system, decided
    before looking at any post-shift result. The TAMC-gated comparison
    still covers both adaptive forecasters paired with the topological
    gate, so the experiment can still observe whether `RecentPatternForecaster`
    or `RollingLinearARForecaster` is the better TAMC-gated partner.

Run:
    uv run python experiments/dynamical_forecast_adaptation.py --seed 0
    uv run python experiments/dynamical_forecast_adaptation.py --multi-seed 10
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from delay_embedding import EmbeddingConfig
from drift_gates import ScalarDriftSignal
from evaluation import mae, rmse
from forecasting import LinearARForecaster, RecentPatternForecaster
from tamc_pipeline import TamicBlendPipeline
from tamic_signal import TamicSignal

from logistic_map_shift import generate_logistic_map_regime_shift
from lorenz_shift import generate_lorenz_regime_shift
from synthetic_regime_shift import (
    autocorrelation_signature,
    generate_sine_to_quasiperiodic,
    l2_distance,
    spectral_signature,
)
from tamc_lite_synthetic_forecast import compute_tradeoff_metrics

GENERATORS = {
    "sine_quasiperiodic": generate_sine_to_quasiperiodic,
    "logistic_map": generate_logistic_map_regime_shift,
    "lorenz": generate_lorenz_regime_shift,
}
DEFAULT_SYSTEMS = tuple(GENERATORS.keys())

# Per-system topology defaults, fixed before running this experiment:
# H0 (drift_dimension=0) throughout, per the topology ablation
# (experiments/topology_ablation.py) showing H0 is the more robust
# default across the full tested delay/window grid for every system,
# including sine (which originally used H1 in the detection experiment).
# Delay values match each system's existing detection setup.
SYSTEM_TOPOLOGY_DEFAULTS = {
    "sine_quasiperiodic": {"delay": 8, "drift_dimension": 0, "window": 128},
    "logistic_map": {"delay": 2, "drift_dimension": 0, "window": 128},
    "lorenz": {"delay": 6, "drift_dimension": 0, "window": 128},
}

VARIANT_NAMES = (
    "Frozen forecaster",
    "Adaptive recent-pattern forecaster",
    "Adaptive rolling LinearAR forecaster",
    "Always-on blend (recent-pattern)",
    "Always-on blend (rolling LinearAR)",
    "TAMC-gated blend (recent-pattern)",
    "TAMC-gated blend (rolling LinearAR)",
    "Mean/variance-gated blend",
    "Autocorrelation-gated blend",
    "Spectral-gated blend",
)


@dataclass
class RollingLinearARForecaster:
    """Causal adaptive forecaster: refits a small LinearAR model from the
    current context window alone, at every prediction call.

    No future labels and no post-shift-specific tuning: at prediction time
    it only ever sees the given `context` (length `context_length` as
    passed by the caller), fits a fresh `LinearARForecaster` using
    supervised (sub-context, sub-target) pairs built entirely from inside
    that context (via `LinearARForecaster.fit`'s own windowing, applied to
    `context` rather than to the full series), then predicts from the most
    recent `train_context_length`-length slice of that same context. Falls
    back to repeating the last observed value when the context is too
    short to form at least `min_train_windows` such pairs.
    """

    horizon: int
    train_context_length: int
    ridge_lambda: float = 1e-3
    min_train_windows: int = 4

    def predict(self, context: np.ndarray) -> np.ndarray:
        context = np.asarray(context, dtype=float)
        n = len(context)
        train_len = self.train_context_length

        n_samples = n - train_len - self.horizon + 1
        if train_len < 2 or n_samples < self.min_train_windows:
            return np.full(self.horizon, context[-1])

        forecaster = LinearARForecaster(
            context_length=train_len,
            horizon=self.horizon,
            ridge_lambda=self.ridge_lambda,
        )
        forecaster.fit(context, context_length=train_len, horizon=self.horizon)
        return forecaster.predict(context[-train_len:])

    def __call__(self, context: np.ndarray) -> np.ndarray:
        return self.predict(context)


def run_experiment(
    system: str,
    seed: int,
    context_length: int,
    horizon: int,
    stride: int,
    window: int,
    delay: int,
    drift_dimension: int,
    rolling_train_context: int,
    gate_threshold: float,
    min_history: int,
) -> tuple[pd.DataFrame, dict]:
    """Run one full seed for one system; return metrics table and raw arrays."""
    generator = GENERATORS[system]
    series, shift_index = generator(seed=seed)
    source_series = series[:shift_index]

    frozen_forecaster = LinearARForecaster(
        context_length=context_length, horizon=horizon
    )
    frozen_forecaster.fit(source_series, context_length=context_length, horizon=horizon)

    recent_pattern_forecaster = RecentPatternForecaster(horizon=horizon)
    rolling_ar_forecaster = RollingLinearARForecaster(
        horizon=horizon, train_context_length=rolling_train_context
    )

    config = EmbeddingConfig(dimension=3, delay=delay, window=window)
    signal = TamicSignal(
        config=config, max_dimension=1, drift_dimension=drift_dimension
    )
    signal.add_source_prototype(source_series[:window], label="source")

    blend_pipeline = TamicBlendPipeline(
        frozen_forecaster=frozen_forecaster,
        adaptive_forecaster=recent_pattern_forecaster,
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
    recent_point: list[float] = []
    rolling_point: list[float] = []
    errors: dict[str, list[tuple[float, float]]] = {name: [] for name in VARIANT_NAMES}

    for t in range(start_index, end_index, stride):
        context = series[t - context_length : t]
        topology_window_values = series[t - window : t]
        target = series[t : t + horizon]

        # One pipeline call advances the shared topology-monitor history
        # and gives the frozen/recent-pattern forecasts plus TAMC's gate;
        # the gate is reused below for the rolling-AR blend so topology is
        # only ever scored once per step.
        result = blend_pipeline.predict(context, topology_window_values)
        frozen_forecast = result["frozen_forecast"]
        recent_forecast = result["adaptive_forecast"]
        tamc_gate = result["gate"]
        tamc_recent_blend = result["blended_forecast"]

        rolling_forecast = rolling_ar_forecaster.predict(context)
        tamc_rolling_blend = (
            1 - tamc_gate
        ) * frozen_forecast + tamc_gate * rolling_forecast

        always_on_recent = 0.5 * frozen_forecast + 0.5 * recent_forecast
        always_on_rolling = 0.5 * frozen_forecast + 0.5 * rolling_forecast

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

        # Non-topological gates pair with RollingLinearARForecaster (the
        # designated "best adaptive forecaster" for these gates; see the
        # module docstring for why).
        forecasts = {
            "Frozen forecaster": frozen_forecast,
            "Adaptive recent-pattern forecaster": recent_forecast,
            "Adaptive rolling LinearAR forecaster": rolling_forecast,
            "Always-on blend (recent-pattern)": always_on_recent,
            "Always-on blend (rolling LinearAR)": always_on_rolling,
            "TAMC-gated blend (recent-pattern)": tamc_recent_blend,
            "TAMC-gated blend (rolling LinearAR)": tamc_rolling_blend,
            "Mean/variance-gated blend": (1 - meanvar_gate) * frozen_forecast
            + meanvar_gate * rolling_forecast,
            "Autocorrelation-gated blend": (1 - acf_gate) * frozen_forecast
            + acf_gate * rolling_forecast,
            "Spectral-gated blend": (1 - spec_gate) * frozen_forecast
            + spec_gate * rolling_forecast,
        }

        times.append(t)
        gates.append(tamc_gate)
        frozen_point.append(frozen_forecast[0])
        recent_point.append(tamc_recent_blend[0])
        rolling_point.append(tamc_rolling_blend[0])

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
        "recent_point": np.array(recent_point),
        "rolling_point": np.array(rolling_point),
    }
    return metrics_table, raw


def plot_results(raw: dict, system: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    series = raw["series"]
    shift_index = raw["shift_index"]
    times = raw["times"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(np.arange(len(series)), series, alpha=0.5, label="True series")
    axes[0].plot(times, raw["frozen_point"], "--", label="Frozen forecast")
    axes[0].plot(
        times, raw["recent_point"], "--", label="TAMC-gated recent-pattern forecast"
    )
    axes[0].plot(
        times, raw["rolling_point"], "--", label="TAMC-gated rolling LinearAR forecast"
    )
    axes[0].axvline(shift_index, linestyle=":", color="black")
    axes[0].set_ylabel("Value")
    axes[0].set_title(f"Dynamical forecast adaptation: {system}")
    axes[0].legend(loc="upper left", fontsize=8)

    axes[1].plot(times, raw["gates"], color="tab:red")
    axes[1].axvline(shift_index, linestyle=":", color="black")
    axes[1].set_ylabel("TAMC gate")
    axes[1].set_xlabel("Time index")
    axes[1].set_ylim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _topology_params(system: str, args) -> tuple[int, int, int]:
    defaults = SYSTEM_TOPOLOGY_DEFAULTS[system]
    delay = args.delay if args.delay is not None else defaults["delay"]
    drift_dimension = (
        args.drift_dimension
        if args.drift_dimension is not None
        else defaults["drift_dimension"]
    )
    window = args.window if args.window is not None else defaults["window"]
    return delay, drift_dimension, window


def run_single_seed(args, figures_dir: Path) -> None:
    for system in args.systems:
        delay, drift_dimension, window = _topology_params(system, args)
        metrics_table, raw = run_experiment(
            system=system,
            seed=args.seed,
            context_length=args.context_length,
            horizon=args.horizon,
            stride=args.stride,
            window=window,
            delay=delay,
            drift_dimension=drift_dimension,
            rolling_train_context=args.rolling_train_context,
            gate_threshold=args.gate_threshold,
            min_history=args.min_history,
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
        figure_path = figures_dir / f"dynamical_forecast_adaptation_{system}.png"
        plot_results(raw, system, figure_path)

        metrics_path = (
            figures_dir / f"dynamical_forecast_adaptation_{system}_metrics.csv"
        )
        merged_metrics_table.to_csv(metrics_path, index=False)

        print(f"\n=== {system} (seed={args.seed}) ===")
        print(f"Saved figure to {figure_path}")
        print(f"Saved metrics to {metrics_path}")
        print(
            merged_metrics_table.to_string(
                index=False, float_format=lambda v: f"{v:.4f}"
            )
        )


def run_multi_seed(args, figures_dir: Path) -> None:
    tables = []
    for system in args.systems:
        delay, drift_dimension, window = _topology_params(system, args)
        for seed in range(args.multi_seed):
            metrics_table, _ = run_experiment(
                system=system,
                seed=seed,
                context_length=args.context_length,
                horizon=args.horizon,
                stride=args.stride,
                window=window,
                delay=delay,
                drift_dimension=drift_dimension,
                rolling_train_context=args.rolling_train_context,
                gate_threshold=args.gate_threshold,
                min_history=args.min_history,
            )
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
            merged.insert(0, "System", system)
            merged.insert(1, "Seed", seed)
            tables.append(merged)
            print(f"[{system}] seed {seed} done", flush=True)

    combined = pd.concat(tables, ignore_index=True)

    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = figures_dir / "dynamical_forecast_adaptation_metrics.csv"
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
    summary = combined.groupby(["System", "Variant", "Segment"])[summary_columns].agg(
        ["mean", "std"]
    )
    summary.columns = [f"{metric} {stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary_path = figures_dir / "dynamical_forecast_adaptation_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")

    tradeoff_columns_only = [
        "Pre Harm",
        "Post Gain",
        "Net Adaptation Score",
        "Post Improvement %",
    ]
    # Tradeoff metrics are identical across both segment rows for a given
    # (System, Seed, Variant), so de-duplicate before aggregating to avoid
    # biasing the across-seed std (see runtime/topology ablation scripts
    # for the same caveat).
    tradeoff_compact = combined.drop_duplicates(subset=["System", "Seed", "Variant"])
    tradeoff_summary = tradeoff_compact.groupby(["System", "Variant"])[
        tradeoff_columns_only
    ].agg(["mean", "std"])
    tradeoff_summary.columns = [
        f"{metric} {stat}" for metric, stat in tradeoff_summary.columns
    ]
    tradeoff_summary = tradeoff_summary.reset_index()
    tradeoff_path = figures_dir / "dynamical_forecast_adaptation_tradeoff_summary.csv"
    tradeoff_summary.to_csv(tradeoff_path, index=False)
    print(f"Saved tradeoff summary to {tradeoff_path}")

    print_interpretation(tradeoff_summary)


def print_interpretation(tradeoff_summary: pd.DataFrame) -> None:
    print("\n=== Honest interpretation ===")
    for system in tradeoff_summary["System"].unique():
        subset = tradeoff_summary[tradeoff_summary["System"] == system]
        best_row = subset.loc[subset["Net Adaptation Score mean"].idxmax()]

        def _score(variant: str) -> float | None:
            rows = subset[subset["Variant"] == variant]
            return (
                float(rows["Net Adaptation Score mean"].iloc[0])
                if not rows.empty
                else None
            )

        tamc_recent = _score("TAMC-gated blend (recent-pattern)")
        tamc_rolling = _score("TAMC-gated blend (rolling LinearAR)")
        always_recent = _score("Always-on blend (recent-pattern)")
        always_rolling = _score("Always-on blend (rolling LinearAR)")
        rolling_alone = _score("Adaptive rolling LinearAR forecaster")

        print(f"\n[{system}]")
        print(
            f"  Best variant by mean Net Adaptation Score: "
            f"{best_row['Variant']} ({best_row['Net Adaptation Score mean']:.4f})"
        )
        print(f"  TAMC-gated blend (recent-pattern): {tamc_recent:.4f}")
        print(f"  TAMC-gated blend (rolling LinearAR): {tamc_rolling:.4f}")
        print(f"  Always-on blend (recent-pattern): {always_recent:.4f}")
        print(f"  Always-on blend (rolling LinearAR): {always_rolling:.4f}")

        best_is_tamc = best_row["Variant"].startswith("TAMC-gated")
        if not best_is_tamc:
            print(
                f"  HONEST NOTE: TAMC-gated blending did not win on {system}; "
                f"the best variant was '{best_row['Variant']}'."
            )

        for label in ["recent-pattern", "rolling LinearAR"]:
            pre_harm_tamc_row = subset[
                subset["Variant"] == f"TAMC-gated blend ({label})"
            ]
            pre_harm_always_row = subset[
                subset["Variant"] == f"Always-on blend ({label})"
            ]
            if pre_harm_tamc_row.empty or pre_harm_always_row.empty:
                continue
            pre_harm_tamc = float(pre_harm_tamc_row["Pre Harm mean"].iloc[0])
            pre_harm_always = float(pre_harm_always_row["Pre Harm mean"].iloc[0])
            reduced = pre_harm_tamc < pre_harm_always
            print(
                f"  TAMC gating vs always-on ({label}): pre-shift harm "
                f"{pre_harm_tamc:.4f} vs {pre_harm_always:.4f} "
                f"({'reduced' if reduced else 'did NOT reduce'} pre-shift harm)"
            )

        if rolling_alone is not None and rolling_alone < 0:
            print(
                f"  HONEST NOTE: rolling LinearAR alone has a negative Net "
                f"Adaptation Score ({rolling_alone:.4f}) on {system}; it is "
                f"harmful as a standalone forecaster here."
            )
        if best_row["Net Adaptation Score mean"] <= 0:
            print(
                f"  HONEST NOTE: no adaptive variant improved on the frozen "
                f"forecaster on {system} (best Net Adaptation Score <= 0); "
                f"the frozen forecaster already appears strong/sufficient here."
            )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--systems",
        nargs="+",
        default=list(DEFAULT_SYSTEMS),
        choices=list(GENERATORS.keys()),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--multi-seed", type=int, default=None, metavar="N")
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument(
        "--window",
        type=int,
        default=None,
        help="Override topology window for all systems.",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=None,
        help="Override embedding delay for all systems.",
    )
    parser.add_argument(
        "--drift-dimension",
        type=int,
        default=None,
        choices=[0, 1],
        help="Override TAMC drift dimension for all systems.",
    )
    parser.add_argument(
        "--rolling-train-context",
        type=int,
        default=64,
        help="Sub-window length used inside RollingLinearARForecaster's local fit "
        "(must be smaller than --context-length to leave room for supervised "
        "windows plus horizon).",
    )
    parser.add_argument("--gate-threshold", type=float, default=2.0)
    parser.add_argument("--min-history", type=int, default=8)
    args = parser.parse_args()

    figures_dir = Path(__file__).resolve().parents[1] / "figures"

    if args.multi_seed is not None:
        run_multi_seed(args, figures_dir)
    else:
        run_single_seed(args, figures_dir)


if __name__ == "__main__":
    main()
