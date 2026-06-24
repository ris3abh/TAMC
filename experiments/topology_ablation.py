"""Topology parameter ablation: homology dimension, delay, and window size.

Goal:
    Show how TAMC's topological drift detection performance depends on
    three modeling choices -- homology dimension (H0 vs H1), embedding
    delay, and topology window size -- across the three controlled
    detection systems already used elsewhere in this repo (sine to
    quasi-periodic, logistic map, Lorenz). This is an ablation, not a
    leaderboard: every combination in the grid is evaluated and reported
    as-is, including weak ones; no parameter is hand-picked after seeing
    results.

Reuses the existing causal generators and detection-metric definitions
from synthetic_regime_shift.py, logistic_map_shift.py, and
lorenz_shift.py (the latter two are byte-identical to the first's
detection_metrics; see those files for the underlying system
definitions). For a fixed (delay, window) pair, H0 and H1 diagrams come
from the *same* Vietoris-Rips persistence computation, so this script
computes persistence once per window and reads off both diagrams, rather
than recomputing persistence twice (once per drift dimension) the way
two independent TamicSignal instances would.

Run:
    uv run python experiments/topology_ablation.py --multi-seed 10
    uv run python experiments/topology_ablation.py --systems lorenz --multi-seed 3
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from delay_embedding import EmbeddingConfig, sliding_windows, takens_embedding
from topology_metrics import (
    diagram_for_dimension,
    vietoris_rips_persistence,
    wasserstein_distance,
)

from logistic_map_shift import generate_logistic_map_regime_shift
from lorenz_shift import generate_lorenz_regime_shift
from synthetic_regime_shift import detection_metrics, generate_sine_to_quasiperiodic

GENERATORS = {
    "sine_quasiperiodic": generate_sine_to_quasiperiodic,
    "logistic_map": generate_logistic_map_regime_shift,
    "lorenz": generate_lorenz_regime_shift,
}

DEFAULT_SYSTEMS = tuple(GENERATORS.keys())
DEFAULT_DELAYS = (2, 4, 6, 8, 12)
# 192 is supported via --windows but left out of the default grid to keep
# the default multi-seed run's persistent-homology call count manageable.
DEFAULT_WINDOWS = (64, 128)
DEFAULT_DIMENSIONS = (0, 1)
DEFAULT_EMBEDDING_DIMENSION = 3
DEFAULT_STRIDE = 8
DEFAULT_N_STD = 3.0
DEFAULT_N_SEEDS = 10

METRIC_COLUMNS = [
    "AUROC",
    "Detection Delay",
    "False Alarms",
    "Separation",
    "Pre Mean",
    "Pre Std",
    "Post Mean",
    "Max Post",
]


def compute_drift_for_both_dimensions(
    series: np.ndarray,
    source_window: np.ndarray,
    embedding_dimension: int,
    delay: int,
    window: int,
    stride: int,
) -> dict[int, np.ndarray]:
    """Compute H0 and H1 topological drift in one pass of persistence calls.

    Returns {0: drift_array, 1: drift_array}, each aligned with the window
    times from `sliding_windows(series, window, stride)`.
    """
    source_cloud = takens_embedding(
        source_window, dimension=embedding_dimension, delay=delay
    )
    source_persistence = vietoris_rips_persistence(source_cloud, max_dimension=1)
    source_diagrams = {
        dim: diagram_for_dimension(source_persistence, dim) for dim in (0, 1)
    }

    windows = sliding_windows(series, window=window, stride=stride)
    drifts: dict[int, list[float]] = {0: [], 1: []}
    for w in windows:
        cloud = takens_embedding(w, dimension=embedding_dimension, delay=delay)
        persistence = vietoris_rips_persistence(cloud, max_dimension=1)
        for dim in (0, 1):
            diagram = diagram_for_dimension(persistence, dim)
            drifts[dim].append(wasserstein_distance(diagram, source_diagrams[dim]))

    return {dim: np.array(values) for dim, values in drifts.items()}


def run_combination(
    series: np.ndarray,
    shift_index: int,
    embedding_dimension: int,
    delay: int,
    window: int,
    stride: int,
    n_std: float,
) -> dict[int, dict] | None:
    """Run one (delay, window) combination across both homology dimensions.

    Returns {0: metrics_dict, 1: metrics_dict}, or None if the combination
    is structurally invalid (e.g. window too small for dimension/delay) --
    invalid combinations are skipped cleanly rather than attempted.
    """
    try:
        EmbeddingConfig(dimension=embedding_dimension, delay=delay, window=window)
    except ValueError:
        return None
    if window > len(series) or window > shift_index:
        return None

    source_window = series[:window]
    drifts = compute_drift_for_both_dimensions(
        series, source_window, embedding_dimension, delay, window, stride
    )
    windows_arr = sliding_windows(series, window=window, stride=stride)
    times = np.arange(len(windows_arr)) * stride + window - 1

    results = {}
    for dim in (0, 1):
        metrics = detection_metrics(drifts[dim], times, shift_index, n_std=n_std)
        results[dim] = metrics
    return results


def _metrics_row(
    system: str, seed: int, dim: int, delay: int, window: int, metrics: dict | None
) -> dict:
    row = {
        "System": system,
        "Seed": seed,
        "Drift Dimension": dim,
        "Delay": delay,
        "Window": window,
    }
    if metrics is None:
        for column in METRIC_COLUMNS:
            row[column] = float("nan")
        return row

    row["AUROC"] = metrics["AUROC"]
    row["Detection Delay"] = metrics["Delay"]
    row["False Alarms"] = metrics["False Alarms"]
    row["Separation"] = metrics["Separation"]
    row["Pre Mean"] = metrics["Pre Mean"]
    row["Pre Std"] = metrics["Pre Std"]
    row["Post Mean"] = metrics["Post Mean"]
    row["Max Post"] = metrics["Max Post"]
    return row


def run_ablation(
    systems: list[str],
    delays: list[int],
    windows: list[int],
    dimensions: list[int],
    embedding_dimension: int,
    stride: int,
    n_std: float,
    n_seeds: int,
) -> pd.DataFrame:
    """Sweep (system, seed, delay, window) and record both H0/H1 metrics per cell."""
    rows: list[dict] = []
    total = len(systems) * n_seeds * len(delays) * len(windows)
    done = 0

    for system in systems:
        generator = GENERATORS[system]
        for seed in range(n_seeds):
            series, shift_index = generator(seed=seed)
            for delay in delays:
                for window in windows:
                    done += 1
                    print(
                        f"[{done}/{total}] system={system} seed={seed} "
                        f"delay={delay} window={window}",
                        flush=True,
                    )
                    try:
                        result = run_combination(
                            series,
                            shift_index,
                            embedding_dimension,
                            delay,
                            window,
                            stride,
                            n_std,
                        )
                    except Exception as exc:  # noqa: BLE001 - keep the grid running
                        print(
                            f"  WARNING: combination failed "
                            f"(system={system}, seed={seed}, delay={delay}, "
                            f"window={window}): {exc}"
                        )
                        result = {dim: None for dim in (0, 1)}

                    if result is None:
                        # Structurally invalid embedding for this combination;
                        # skip cleanly rather than recording a row.
                        continue

                    for dim in dimensions:
                        rows.append(
                            _metrics_row(
                                system, seed, dim, delay, window, result.get(dim)
                            )
                        )

    return pd.DataFrame(rows)


def build_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean/std of AUROC, Detection Delay, False Alarms, Separation."""
    summary_columns = ["AUROC", "Detection Delay", "False Alarms", "Separation"]
    grouped = metrics_df.groupby(["System", "Drift Dimension", "Delay", "Window"])[
        summary_columns
    ].agg(["mean", "std"])
    grouped.columns = [f"{metric} {stat}" for metric, stat in grouped.columns]
    return grouped.reset_index()


def plot_heatmaps(
    summary_df: pd.DataFrame,
    systems: list[str],
    dimensions: list[int],
    output_path: Path,
) -> None:
    """One AUROC-mean heatmap per (system, drift dimension), delay x window."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = len(dimensions)
    n_cols = len(systems)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.5 * n_cols, 4.0 * n_rows), squeeze=False
    )

    for row, dim in enumerate(dimensions):
        for col, system in enumerate(systems):
            ax = axes[row][col]
            subset = summary_df[
                (summary_df["System"] == system)
                & (summary_df["Drift Dimension"] == dim)
            ]
            if subset.empty or "AUROC mean" not in subset.columns:
                ax.set_title(f"{system} (H{dim}) -- no data")
                ax.axis("off")
                continue

            pivot = subset.pivot(index="Window", columns="Delay", values="AUROC mean")
            pivot = pivot.sort_index(axis=0).sort_index(axis=1)

            image = ax.imshow(pivot.values, aspect="auto", vmin=0.5, vmax=1.0)
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            ax.set_xlabel("Delay")
            ax.set_ylabel("Window")
            ax.set_title(f"{system} (H{dim})")

            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    value = pivot.values[i, j]
                    if np.isfinite(value):
                        ax.text(
                            j, i, f"{value:.2f}", ha="center", va="center", fontsize=8
                        )

            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="AUROC mean")

    fig.suptitle("TAMC topological drift detection: AUROC mean over delay x window")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--systems",
        nargs="+",
        default=list(DEFAULT_SYSTEMS),
        choices=list(GENERATORS.keys()),
    )
    parser.add_argument("--delays", nargs="+", type=int, default=list(DEFAULT_DELAYS))
    parser.add_argument("--windows", nargs="+", type=int, default=list(DEFAULT_WINDOWS))
    parser.add_argument(
        "--dimensions",
        nargs="+",
        type=int,
        default=list(DEFAULT_DIMENSIONS),
        choices=[0, 1],
    )
    parser.add_argument(
        "--embedding-dimension", type=int, default=DEFAULT_EMBEDDING_DIMENSION
    )
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--n-std", type=float, default=DEFAULT_N_STD)
    parser.add_argument(
        "--multi-seed",
        type=int,
        default=DEFAULT_N_SEEDS,
        metavar="N",
        help="Number of seeds (0..N-1) to run per system (default: 10).",
    )
    args = parser.parse_args()

    start_time = time.time()
    metrics_df = run_ablation(
        systems=args.systems,
        delays=args.delays,
        windows=args.windows,
        dimensions=args.dimensions,
        embedding_dimension=args.embedding_dimension,
        stride=args.stride,
        n_std=args.n_std,
        n_seeds=args.multi_seed,
    )
    elapsed = time.time() - start_time
    print(f"\nAblation grid finished in {elapsed:.1f}s ({len(metrics_df)} rows)")

    figures_dir = Path(__file__).resolve().parents[1] / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = figures_dir / "topology_ablation_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved long-format metrics to {metrics_path}")

    if metrics_df.empty:
        print("No successful combinations; skipping summary and heatmap outputs.")
        return

    summary_df = build_summary(metrics_df)
    summary_path = figures_dir / "topology_ablation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")

    heatmap_path = figures_dir / "topology_ablation_heatmap.png"
    plot_heatmaps(summary_df, args.systems, args.dimensions, heatmap_path)
    print(f"Saved heatmap to {heatmap_path}")

    print("\nSummary (AUROC mean +/- std), head:")
    print(
        summary_df[
            ["System", "Drift Dimension", "Delay", "Window", "AUROC mean", "AUROC std"]
        ]
        .sort_values(["System", "Drift Dimension", "Window", "Delay"])
        .to_string(index=False, float_format=lambda v: f"{v:.4f}")
    )


if __name__ == "__main__":
    main()
