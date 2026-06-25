"""Diagnostic shift-type sweep: TAMC vs. RG-style regime-similarity gate.

Goal:
    `experiments/benchmark_regime_control.py` found that TAMC loses to an
    RG-style statistical/distributional regime-similarity gate on all 5
    real datasets tested, under one shift type (`seasonality_break`).
    This is a diagnostic, not a paper-polish task: does TAMC underperform
    RG-style specifically on `seasonality_break`, or across every
    controlled real-data perturbation type? This script sweeps all 5
    shift types supported by `real_data_controlled_shift.py` across all 5
    datasets and reports an honest, unaveraged per-cell comparison.

Reuses `run_experiment` from `benchmark_regime_control.py` directly
(same causal protocol: normalization fit on the source half only, shift
injected into the post-shift half only, frozen forecaster and every
gate's source reference built only from the source segment, no future
labels) rather than duplicating the benchmark logic. The only addition
here is the outer sweep over shift types and the TAMC-vs-RG diagnostic
summary.

This is expensive: 5 datasets x 5 shift types x N seeds. Start with
`--multi-seed 1`, then `--multi-seed 3`; only run `--multi-seed 10` if
runtime is acceptable (the single-shift-type benchmark took ~26 minutes
for 10 seeds x 5 datasets, so this full sweep is expected to take roughly
5x that).

Run:
    uv run python experiments/benchmark_shift_types.py --multi-seed 1
    uv run python experiments/benchmark_shift_types.py --multi-seed 3
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from benchmark_regime_control import DATASET_FILENAMES, run_experiment
from real_data_controlled_shift import SHIFT_TYPES
from tamc_lite_synthetic_forecast import compute_tradeoff_metrics

DEFAULT_DATASETS = tuple(DATASET_FILENAMES.keys())
DEFAULT_SHIFT_TYPES = tuple(SHIFT_TYPES)

TAMC_VARIANT = "TAMC-gated blend"
RG_VARIANT = "RG-style regime-similarity-gated blend"

WINNER_CATEGORY_MAP = {
    "Frozen forecaster": "Frozen",
    "Adaptive recent-pattern forecaster": "Adaptive alone",
    "Always-on 50/50 blend": "Always-on",
    "TAMC-gated blend": "TAMC",
    "RG-style regime-similarity-gated blend": "RG-style",
    "Mean/variance-gated blend": "Other non-topological gate",
    "Autocorrelation-gated blend": "Other non-topological gate",
    "Spectral-gated blend": "Other non-topological gate",
}


def run_sweep(args) -> pd.DataFrame:
    """Sweep (dataset, shift_type, seed); return the combined long-format table."""
    tables = []
    total = len(args.datasets) * len(args.shift_types) * args.multi_seed
    done = 0

    for dataset in args.datasets:
        csv_path = Path(args.data_dir) / DATASET_FILENAMES[dataset]
        for shift_type in args.shift_types:
            cell_failed = False
            for seed in range(args.multi_seed):
                done += 1
                if cell_failed:
                    print(
                        f"[{done}/{total}] Dataset={dataset} Shift={shift_type} "
                        f"Seed={seed}: SKIPPED (dataset/shift unavailable, "
                        f"already confirmed for this cell)",
                        flush=True,
                    )
                    continue
                try:
                    result = run_experiment(
                        dataset=dataset,
                        csv_path=csv_path,
                        value_column_override=args.value_column,
                        shift_type=shift_type,
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
                except Exception as exc:  # noqa: BLE001 - keep the sweep running
                    print(
                        f"[{done}/{total}] Dataset={dataset} Shift={shift_type} "
                        f"Seed={seed}: FAILED with {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    cell_failed = True
                    continue

                if result is None:
                    # run_experiment already printed the SKIP reason
                    # (missing file or column); no point retrying other
                    # seeds for this (dataset, shift_type) cell.
                    cell_failed = True
                    continue

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
                merged.insert(1, "Shift Type", shift_type)
                merged.insert(2, "Seed", seed)
                tables.append(merged)
                print(
                    f"[{done}/{total}] Dataset={dataset} Shift={shift_type} Seed={seed} done",
                    flush=True,
                )

    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True)


def build_summary(combined: pd.DataFrame) -> pd.DataFrame:
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
    return summary.reset_index()


def build_tradeoff_summary(combined: pd.DataFrame) -> pd.DataFrame:
    tradeoff_columns = [
        "Pre Harm",
        "Post Gain",
        "Net Adaptation Score",
        "Post Improvement %",
    ]
    # Tradeoff metrics are identical across both segment rows for a given
    # (Dataset, Shift Type, Seed, Variant); de-duplicate before aggregating
    # so the across-seed std is not biased (same caveat as other benchmark
    # scripts in this repo).
    compact = combined.drop_duplicates(
        subset=["Dataset", "Shift Type", "Seed", "Variant"]
    )
    tradeoff_summary = compact.groupby(["Dataset", "Shift Type", "Variant"])[
        tradeoff_columns
    ].agg(["mean", "std"])
    tradeoff_summary.columns = [
        f"{metric} {stat}" for metric, stat in tradeoff_summary.columns
    ]
    return tradeoff_summary.reset_index()


def classify_winner(
    best_variant: str, best_nas_mean: float, near_zero_threshold: float
) -> str:
    if abs(best_nas_mean) < near_zero_threshold:
        return "Tie/near-zero"
    return WINNER_CATEGORY_MAP.get(best_variant, "Other non-topological gate")


def build_tamc_vs_rg(
    tradeoff_summary: pd.DataFrame, near_zero_threshold: float
) -> pd.DataFrame:
    """One row per (Dataset, Shift Type): TAMC vs. RG-style head-to-head."""
    rows = []
    for (dataset, shift_type), group in tradeoff_summary.groupby(
        ["Dataset", "Shift Type"]
    ):
        group = group.set_index("Variant")
        if TAMC_VARIANT not in group.index or RG_VARIANT not in group.index:
            continue

        ranked = group["Net Adaptation Score mean"].rank(method="min", ascending=False)
        tamc_nas = float(group.loc[TAMC_VARIANT, "Net Adaptation Score mean"])
        rg_nas = float(group.loc[RG_VARIANT, "Net Adaptation Score mean"])
        tamc_rank = int(ranked.loc[TAMC_VARIANT])
        rg_rank = int(ranked.loc[RG_VARIANT])

        best_variant = group["Net Adaptation Score mean"].idxmax()
        best_nas = float(group.loc[best_variant, "Net Adaptation Score mean"])

        rows.append(
            {
                "Dataset": dataset,
                "Shift Type": shift_type,
                "TAMC NAS mean": tamc_nas,
                "RG-style NAS mean": rg_nas,
                "TAMC minus RG": tamc_nas - rg_nas,
                "TAMC rank": tamc_rank,
                "RG-style rank": rg_rank,
                "Best variant": best_variant,
                "Best NAS mean": best_nas,
                "Winner Category": classify_winner(
                    best_variant, best_nas, near_zero_threshold
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_tamc_vs_rg(tamc_vs_rg: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pivot = tamc_vs_rg.pivot(
        index="Dataset", columns="Shift Type", values="TAMC minus RG"
    )
    pivot = pivot.reindex(
        columns=[s for s in DEFAULT_SHIFT_TYPES if s in pivot.columns]
    )

    limit = float(np.nanmax(np.abs(pivot.values))) if pivot.size else 1.0
    limit = max(limit, 1e-6)

    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(pivot.values, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("TAMC vs RG-style gate across real-data shift types")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.values[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:+.4f}", ha="center", va="center", fontsize=8)

    fig.colorbar(image, ax=ax, label="TAMC NAS - RG-style NAS")
    fig.text(
        0.5,
        0.01,
        "Positive means TAMC has higher Net Adaptation Score than RG-style gate; "
        "values near zero should not be overinterpreted.",
        ha="center",
        fontsize=8,
        style="italic",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def print_interpretation(
    tamc_vs_rg: pd.DataFrame, near_zero_threshold: float, meaningful_threshold: float
) -> None:
    print("\n=== Diagnostic interpretation: TAMC vs RG-style gate ===")

    if tamc_vs_rg.empty:
        print("No cells completed; nothing to interpret.")
        return

    diff = tamc_vs_rg["TAMC minus RG"]
    tamc_wins_mask = diff > near_zero_threshold
    rg_wins_mask = diff < -near_zero_threshold
    tie_mask = diff.abs() <= near_zero_threshold

    n_cells = len(tamc_vs_rg)
    n_tamc_wins = int(tamc_wins_mask.sum())
    n_rg_wins = int(rg_wins_mask.sum())
    n_ties = int(tie_mask.sum())

    print(f"\nTotal dataset/shift cells completed: {n_cells}")
    print(
        f"1. TAMC beats RG-style (margin > {near_zero_threshold}): {n_tamc_wins} cells"
    )
    print(f"2. RG-style beats TAMC (margin > {near_zero_threshold}): {n_rg_wins} cells")
    print(f"3. Near-tied (|margin| <= {near_zero_threshold}): {n_ties} cells")

    by_shift = (
        tamc_vs_rg.groupby("Shift Type")["TAMC minus RG"]
        .mean()
        .sort_values(ascending=False)
    )
    print("\n4-5. Mean (TAMC - RG) by shift type, descending (favors TAMC at top):")
    for shift_type, mean_diff in by_shift.items():
        label = (
            "favors TAMC"
            if mean_diff > near_zero_threshold
            else "favors RG-style" if mean_diff < -near_zero_threshold else "near-tied"
        )
        print(f"   {shift_type:<20s} {mean_diff:+.4f}  ({label})")

    if "seasonality_break" in by_shift.index:
        seasonality_diff = by_shift["seasonality_break"]
        other_mean = (
            by_shift.drop("seasonality_break").mean()
            if len(by_shift) > 1
            else float("nan")
        )
        print(
            f"\n6. seasonality_break mean (TAMC-RG)={seasonality_diff:+.4f} vs. "
            f"mean of other shift types={other_mean:+.4f} -- "
            + (
                "seasonality_break looks broadly representative of the other shift types."
                if np.isfinite(other_mean)
                and abs(seasonality_diff - other_mean) < meaningful_threshold
                else "seasonality_break differs from the other shift types; do not generalize from it alone."
            )
        )

    max_abs_margin = diff.abs().max()
    print(
        f"\n7. Largest |TAMC - RG| margin across all cells: {max_abs_margin:.4f} -- "
        + (
            "most differences are tiny; do not treat them as practically meaningful."
            if max_abs_margin < meaningful_threshold
            else "at least one cell shows a practically meaningful gap."
        )
    )

    n_tamc_rank1 = int((tamc_vs_rg["TAMC rank"] == 1).sum())
    print(
        f"\n8. TAMC ranks 1st overall (of 8 variants) in {n_tamc_rank1}/{n_cells} cells."
    )

    n_tamc_beats_all_non_topological = int(
        (tamc_vs_rg["TAMC minus RG"] > near_zero_threshold).sum()
    )
    # "beats all non-topological gates" is stricter than "beats RG-style
    # alone"; report both so the claim is not overstated.
    print(
        f"9. TAMC beats the RG-style gate specifically in "
        f"{n_tamc_beats_all_non_topological}/{n_cells} cells (see note 1 above for "
        f"the full non-topological-gate comparison, available per-cell in the "
        f"tradeoff summary CSV)."
    )

    n_frozen_best = int((tamc_vs_rg["Winner Category"] == "Frozen").sum())
    print(
        f"10. Frozen forecaster is the outright best variant in {n_frozen_best}/{n_cells} "
        f"cells -- {'adaptation is frequently not useful under this protocol.' if n_frozen_best > n_cells / 3 else 'adaptation is sometimes useful, but inconsistently.'}"
    )

    print("\n--- Per-shift-type summary ---")
    summary_rows = []
    for shift_type in DEFAULT_SHIFT_TYPES:
        subset = tamc_vs_rg[tamc_vs_rg["Shift Type"] == shift_type]
        if subset.empty:
            continue
        tamc_wins = int((subset["TAMC minus RG"] > near_zero_threshold).sum())
        rg_wins = int((subset["TAMC minus RG"] < -near_zero_threshold).sum())
        ties = int((subset["TAMC minus RG"].abs() <= near_zero_threshold).sum())
        best_dataset_row = subset.loc[subset["TAMC minus RG"].idxmax()]
        worst_dataset_row = subset.loc[subset["TAMC minus RG"].idxmin()]
        interpretation = (
            "TAMC favored"
            if tamc_wins > rg_wins
            else "RG-style favored" if rg_wins > tamc_wins else "Mixed/near-tied"
        )
        summary_rows.append(
            {
                "Shift Type": shift_type,
                "TAMC wins": tamc_wins,
                "RG wins": rg_wins,
                "Near ties": ties,
                "Best TAMC dataset": f"{best_dataset_row['Dataset']} ({best_dataset_row['TAMC minus RG']:+.4f})",
                "Worst TAMC dataset": f"{worst_dataset_row['Dataset']} ({worst_dataset_row['TAMC minus RG']:+.4f})",
                "Interpretation": interpretation,
            }
        )
    print(pd.DataFrame(summary_rows).to_string(index=False))

    print("\n--- Per-cell table ---")
    cell_table = tamc_vs_rg[
        [
            "Dataset",
            "Shift Type",
            "Best variant",
            "TAMC NAS mean",
            "RG-style NAS mean",
            "TAMC minus RG",
            "TAMC rank",
            "RG-style rank",
        ]
    ].rename(
        columns={
            "TAMC NAS mean": "TAMC NAS",
            "RG-style NAS mean": "RG NAS",
            "TAMC minus RG": "TAMC-RG",
        }
    )
    print(
        cell_table.sort_values(["Dataset", "Shift Type"]).to_string(
            index=False, float_format=lambda v: f"{v:.4f}"
        )
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        choices=list(DATASET_FILENAMES.keys()),
    )
    parser.add_argument(
        "--shift-types",
        nargs="+",
        default=list(DEFAULT_SHIFT_TYPES),
        choices=list(SHIFT_TYPES),
    )
    parser.add_argument("--multi-seed", type=int, default=3, metavar="N")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--n-points", type=int, default=3000)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--delay", type=int, default=8)
    parser.add_argument("--drift-dimension", type=int, default=0, choices=[0, 1])
    parser.add_argument("--gate-threshold", type=float, default=2.0)
    parser.add_argument("--min-history", type=int, default=8)
    parser.add_argument("--value-column", default=None)
    parser.add_argument(
        "--near-zero-threshold",
        type=float,
        default=0.001,
        help="Net Adaptation Score margin below which a TAMC-vs-RG comparison "
        "is treated as a tie rather than a meaningful win/loss.",
    )
    parser.add_argument(
        "--meaningful-threshold",
        type=float,
        default=0.005,
        help="Margin above which a TAMC-vs-RG difference is treated as practically "
        "meaningful (separate from, and larger than, --near-zero-threshold).",
    )
    args = parser.parse_args()

    figures_dir = Path(__file__).resolve().parents[1] / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    combined = run_sweep(args)
    if combined.empty:
        print("\nNo cells completed; nothing to save or interpret.")
        return

    metrics_path = figures_dir / "benchmark_shift_types_metrics.csv"
    combined.to_csv(metrics_path, index=False)
    print(f"\nSaved long-format metrics to {metrics_path}")

    summary = build_summary(combined)
    summary_path = figures_dir / "benchmark_shift_types_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")

    tradeoff_summary = build_tradeoff_summary(combined)
    tradeoff_path = figures_dir / "benchmark_shift_types_tradeoff_summary.csv"
    tradeoff_summary.to_csv(tradeoff_path, index=False)
    print(f"Saved tradeoff summary to {tradeoff_path}")

    tamc_vs_rg = build_tamc_vs_rg(tradeoff_summary, args.near_zero_threshold)
    tamc_vs_rg_path = figures_dir / "benchmark_shift_types_tamc_vs_rg.csv"
    tamc_vs_rg.to_csv(tamc_vs_rg_path, index=False)
    print(f"Saved TAMC-vs-RG summary to {tamc_vs_rg_path}")

    plot_path = figures_dir / "benchmark_shift_types_tamc_vs_rg.png"
    plot_tamc_vs_rg(tamc_vs_rg, plot_path)
    print(f"Saved TAMC-vs-RG plot to {plot_path}")

    print_interpretation(
        tamc_vs_rg, args.near_zero_threshold, args.meaningful_threshold
    )


if __name__ == "__main__":
    main()
