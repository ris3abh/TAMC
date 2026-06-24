"""Runtime/compute benchmark: TAMC topological drift vs. non-topological drift.

Goal:
    Quantify, honestly, how much slower TAMC's topological drift signal
    is to compute per window than simple non-topological baselines (mean/
    variance, autocorrelation, spectral), across a few topology/window
    sizes. This is a cost benchmark, not an accuracy benchmark -- see
    synthetic_regime_shift.py / logistic_map_shift.py / lorenz_shift.py
    for detection-accuracy comparisons, and topology_ablation.py for the
    H0-vs-H1 accuracy ablation.

Uses the same sine-to-quasi-periodic generator as the other experiments,
since runtime here depends on series/window length, not on which
dynamical system generated the series.

For TAMC, persistence is computed once per window and both the H0 and H1
diagrams are read off that single computation (mirroring
topology_ablation.py's efficiency note) -- but H0 and H1 are still timed
and reported as separate rows, each including its own Wasserstein-distance
call, so the per-method "Seconds Per Window" numbers reported are not
artificially inflated by counting persistence twice, nor are they
under-counting the real cost either method actually pays.

Run:
    uv run python experiments/runtime_benchmark.py \\
        --seed 0 --n-repeat 3 --windows 64 128 192 --stride 8 --max-windows 100
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from delay_embedding import sliding_windows, takens_embedding
from topology_metrics import (
    diagram_for_dimension,
    vietoris_rips_persistence,
    wasserstein_distance,
)

from synthetic_regime_shift import (
    autocorrelation_signature,
    generate_sine_to_quasiperiodic,
    l2_distance,
    spectral_signature,
)

METHODS = (
    "Mean/variance drift",
    "Autocorrelation drift",
    "Spectral drift",
    "TAMC H0 drift",
    "TAMC H1 drift",
)


def time_mean_variance_drift(windows: np.ndarray, source_window: np.ndarray) -> float:
    source_mean = source_window.mean()
    source_var = source_window.var()
    start = time.perf_counter()
    for w in windows:
        abs(w.mean() - source_mean) + abs(w.var() - source_var)
    return time.perf_counter() - start


def time_autocorrelation_drift(windows: np.ndarray, source_window: np.ndarray) -> float:
    source_signature = autocorrelation_signature(source_window)
    start = time.perf_counter()
    for w in windows:
        l2_distance(autocorrelation_signature(w), source_signature)
    return time.perf_counter() - start


def time_spectral_drift(windows: np.ndarray, source_window: np.ndarray) -> float:
    source_signature = spectral_signature(source_window)
    start = time.perf_counter()
    for w in windows:
        l2_distance(spectral_signature(w), source_signature)
    return time.perf_counter() - start


def time_tamc_drift(
    windows: np.ndarray,
    source_window: np.ndarray,
    embedding_dimension: int,
    delay: int,
) -> dict[str, float]:
    """Time TAMC H0 and TAMC H1, computing persistence once per window.

    `vietoris_rips_persistence` is called only once per window (not once
    per dimension) -- but the resulting cost is charged in full to *both*
    the H0 and H1 elapsed totals, not split between them. This reflects
    the real, honest standalone cost of either: `TamicSignal` as actually
    used in every other experiment script is configured with one fixed
    `drift_dimension` and pays the full persistence cost for it, since
    persistent homology has no cheaper "H0-only" mode -- the cost
    structure is the full Vietoris-Rips computation regardless of which
    diagram is read off afterward. The benchmark loop itself only avoids
    *redundant* computation (one persistence call serves both rows here,
    rather than two independent passes), it does not pretend either
    dimension is half-price in real use.
    """
    source_cloud = takens_embedding(
        source_window, dimension=embedding_dimension, delay=delay
    )
    source_persistence = vietoris_rips_persistence(source_cloud, max_dimension=1)
    source_diagrams = {
        dim: diagram_for_dimension(source_persistence, dim) for dim in (0, 1)
    }

    h0_elapsed = 0.0
    h1_elapsed = 0.0
    for w in windows:
        start = time.perf_counter()
        cloud = takens_embedding(w, dimension=embedding_dimension, delay=delay)
        persistence = vietoris_rips_persistence(cloud, max_dimension=1)
        persistence_done = time.perf_counter()
        persistence_cost = persistence_done - start

        diagram_h0 = diagram_for_dimension(persistence, 0)
        wasserstein_distance(diagram_h0, source_diagrams[0])
        h0_done = time.perf_counter()

        diagram_h1 = diagram_for_dimension(persistence, 1)
        wasserstein_distance(diagram_h1, source_diagrams[1])
        h1_done = time.perf_counter()

        h0_elapsed += persistence_cost + (h0_done - persistence_done)
        h1_elapsed += persistence_cost + (h1_done - h0_done)

    return {"TAMC H0 drift": h0_elapsed, "TAMC H1 drift": h1_elapsed}


def run_benchmark(
    series: np.ndarray,
    windows_list: list[int],
    stride: int,
    embedding_dimension: int,
    delay: int,
    max_windows: int,
    n_repeat: int,
) -> pd.DataFrame:
    rows: list[dict] = []

    for window in windows_list:
        all_windows = sliding_windows(series, window=window, stride=stride)
        if len(all_windows) > max_windows:
            all_windows = all_windows[:max_windows]
        num_windows = len(all_windows)
        source_window = series[:window]

        print(
            f"window={window}: {num_windows} windows, "
            f"{n_repeat} repeats per method",
            flush=True,
        )

        for repeat in range(n_repeat):
            elapsed = {
                "Mean/variance drift": time_mean_variance_drift(
                    all_windows, source_window
                ),
                "Autocorrelation drift": time_autocorrelation_drift(
                    all_windows, source_window
                ),
                "Spectral drift": time_spectral_drift(all_windows, source_window),
            }
            elapsed.update(
                time_tamc_drift(all_windows, source_window, embedding_dimension, delay)
            )

            for method in METHODS:
                total_seconds = elapsed[method]
                rows.append(
                    {
                        "Method": method,
                        "Window": window,
                        "Repeat": repeat,
                        "Num Windows": num_windows,
                        "Total Seconds": total_seconds,
                        "Seconds Per Window": total_seconds / num_windows,
                        "Windows Per Second": (
                            num_windows / total_seconds
                            if total_seconds > 0
                            else float("inf")
                        ),
                    }
                )

    return pd.DataFrame(rows)


def build_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    summary_columns = ["Total Seconds", "Seconds Per Window", "Windows Per Second"]
    grouped = metrics_df.groupby(["Method", "Window"])[summary_columns].agg(
        ["mean", "std"]
    )
    grouped.columns = [f"{metric} {stat}" for metric, stat in grouped.columns]
    return grouped.reset_index()


def plot_runtime(summary_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    for method in METHODS:
        subset = summary_df[summary_df["Method"] == method].sort_values("Window")
        ax.plot(
            subset["Window"],
            subset["Seconds Per Window mean"],
            marker="o",
            label=method,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Window size")
    ax.set_ylabel("Seconds per window (log scale)")
    ax.set_title("TAMC topological drift vs. non-topological drift: runtime cost")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def print_slowdowns(summary_df: pd.DataFrame, windows_list: list[int]) -> None:
    print("\nApproximate TAMC H0 slowdown vs. non-topological baselines:")
    for window in windows_list:
        subset = summary_df[summary_df["Window"] == window]

        def _seconds_per_window(method: str) -> float | None:
            rows = subset[subset["Method"] == method]
            if rows.empty:
                return None
            return float(rows["Seconds Per Window mean"].iloc[0])

        tamc_h0 = _seconds_per_window("TAMC H0 drift")
        spectral = _seconds_per_window("Spectral drift")
        autocorrelation = _seconds_per_window("Autocorrelation drift")

        if tamc_h0 is None or spectral is None or autocorrelation is None:
            continue

        spectral_slowdown = tamc_h0 / spectral if spectral > 0 else float("inf")
        autocorr_slowdown = (
            tamc_h0 / autocorrelation if autocorrelation > 0 else float("inf")
        )
        print(
            f"Window {window}: TAMC H0 is {spectral_slowdown:.1f}x slower than "
            f"spectral drift and {autocorr_slowdown:.1f}x slower than "
            f"autocorrelation drift."
        )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-repeat", type=int, default=3)
    parser.add_argument("--windows", nargs="+", type=int, default=[64, 128, 192])
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--embedding-dimension", type=int, default=3)
    parser.add_argument("--delay", type=int, default=8)
    parser.add_argument("--max-windows", type=int, default=100)
    args = parser.parse_args()

    series, _ = generate_sine_to_quasiperiodic(seed=args.seed)

    metrics_df = run_benchmark(
        series=series,
        windows_list=args.windows,
        stride=args.stride,
        embedding_dimension=args.embedding_dimension,
        delay=args.delay,
        max_windows=args.max_windows,
        n_repeat=args.n_repeat,
    )

    figures_dir = Path(__file__).resolve().parents[1] / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = figures_dir / "runtime_benchmark_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved per-repeat metrics to {metrics_path}")

    summary_df = build_summary(metrics_df)
    summary_path = figures_dir / "runtime_benchmark_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")

    figure_path = figures_dir / "runtime_benchmark.png"
    plot_runtime(summary_df, figure_path)
    print(f"Saved figure to {figure_path}")

    print("\nFull summary (mean +/- std over repeats):")
    print(summary_df.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    print_slowdowns(summary_df, args.windows)


if __name__ == "__main__":
    main()
