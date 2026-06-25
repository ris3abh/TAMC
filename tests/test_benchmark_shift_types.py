import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "experiments"))

from benchmark_shift_types import build_tamc_vs_rg, classify_winner


def _tradeoff_row(dataset, shift_type, variant, nas_mean):
    return {
        "Dataset": dataset,
        "Shift Type": shift_type,
        "Variant": variant,
        "Pre Harm mean": 0.0,
        "Pre Harm std": 0.0,
        "Post Gain mean": nas_mean,
        "Post Gain std": 0.0,
        "Net Adaptation Score mean": nas_mean,
        "Net Adaptation Score std": 0.0,
        "Post Improvement % mean": 0.0,
        "Post Improvement % std": 0.0,
    }


def _tiny_tradeoff_summary() -> pd.DataFrame:
    rows = [
        # Cell where TAMC clearly beats RG-style.
        _tradeoff_row("DatasetA", "shiftX", "Frozen forecaster", 0.0),
        _tradeoff_row(
            "DatasetA", "shiftX", "Adaptive recent-pattern forecaster", -0.05
        ),
        _tradeoff_row("DatasetA", "shiftX", "Always-on 50/50 blend", -0.01),
        _tradeoff_row("DatasetA", "shiftX", "TAMC-gated blend", 0.05),
        _tradeoff_row(
            "DatasetA", "shiftX", "RG-style regime-similarity-gated blend", 0.01
        ),
        _tradeoff_row("DatasetA", "shiftX", "Mean/variance-gated blend", 0.0),
        _tradeoff_row("DatasetA", "shiftX", "Autocorrelation-gated blend", 0.0),
        _tradeoff_row("DatasetA", "shiftX", "Spectral-gated blend", 0.0),
        # Cell where RG-style clearly beats TAMC.
        _tradeoff_row("DatasetB", "shiftY", "Frozen forecaster", 0.0),
        _tradeoff_row(
            "DatasetB", "shiftY", "Adaptive recent-pattern forecaster", -0.05
        ),
        _tradeoff_row("DatasetB", "shiftY", "Always-on 50/50 blend", -0.01),
        _tradeoff_row("DatasetB", "shiftY", "TAMC-gated blend", -0.02),
        _tradeoff_row(
            "DatasetB", "shiftY", "RG-style regime-similarity-gated blend", 0.03
        ),
        _tradeoff_row("DatasetB", "shiftY", "Mean/variance-gated blend", 0.0),
        _tradeoff_row("DatasetB", "shiftY", "Autocorrelation-gated blend", 0.0),
        _tradeoff_row("DatasetB", "shiftY", "Spectral-gated blend", 0.0),
        # Near-tied cell: both gates within the near-zero threshold of 0.
        _tradeoff_row("DatasetC", "shiftZ", "Frozen forecaster", 0.0),
        _tradeoff_row(
            "DatasetC", "shiftZ", "Adaptive recent-pattern forecaster", -0.05
        ),
        _tradeoff_row("DatasetC", "shiftZ", "Always-on 50/50 blend", -0.01),
        _tradeoff_row("DatasetC", "shiftZ", "TAMC-gated blend", 0.0001),
        _tradeoff_row(
            "DatasetC", "shiftZ", "RG-style regime-similarity-gated blend", 0.0002
        ),
        _tradeoff_row("DatasetC", "shiftZ", "Mean/variance-gated blend", 0.0),
        _tradeoff_row("DatasetC", "shiftZ", "Autocorrelation-gated blend", 0.0),
        _tradeoff_row("DatasetC", "shiftZ", "Spectral-gated blend", 0.0),
    ]
    return pd.DataFrame(rows)


def test_build_tamc_vs_rg_produces_one_row_per_cell():
    tradeoff_summary = _tiny_tradeoff_summary()
    result = build_tamc_vs_rg(tradeoff_summary, near_zero_threshold=0.001)

    assert len(result) == 3
    assert set(result["Dataset"]) == {"DatasetA", "DatasetB", "DatasetC"}


def test_build_tamc_vs_rg_tamc_wins_cell():
    tradeoff_summary = _tiny_tradeoff_summary()
    result = build_tamc_vs_rg(tradeoff_summary, near_zero_threshold=0.001)
    row = result[result["Dataset"] == "DatasetA"].iloc[0]

    assert row["TAMC NAS mean"] == 0.05
    assert row["RG-style NAS mean"] == 0.01
    assert abs(row["TAMC minus RG"] - 0.04) < 1e-9
    assert row["TAMC rank"] == 1
    assert row["Best variant"] == "TAMC-gated blend"
    assert row["Winner Category"] == "TAMC"


def test_build_tamc_vs_rg_rg_wins_cell():
    tradeoff_summary = _tiny_tradeoff_summary()
    result = build_tamc_vs_rg(tradeoff_summary, near_zero_threshold=0.001)
    row = result[result["Dataset"] == "DatasetB"].iloc[0]

    assert row["RG-style NAS mean"] == 0.03
    assert row["TAMC NAS mean"] == -0.02
    assert row["RG-style rank"] == 1
    assert row["Best variant"] == "RG-style regime-similarity-gated blend"
    assert row["Winner Category"] == "RG-style"


def test_build_tamc_vs_rg_near_tied_cell_is_classified_as_tie():
    tradeoff_summary = _tiny_tradeoff_summary()
    result = build_tamc_vs_rg(tradeoff_summary, near_zero_threshold=0.001)
    row = result[result["Dataset"] == "DatasetC"].iloc[0]

    assert row["Winner Category"] == "Tie/near-zero"


def test_classify_winner_respects_near_zero_threshold():
    assert (
        classify_winner("TAMC-gated blend", 0.0005, near_zero_threshold=0.001)
        == "Tie/near-zero"
    )
    assert (
        classify_winner("TAMC-gated blend", 0.002, near_zero_threshold=0.001) == "TAMC"
    )
    assert (
        classify_winner(
            "RG-style regime-similarity-gated blend", 0.002, near_zero_threshold=0.001
        )
        == "RG-style"
    )
    assert (
        classify_winner("Autocorrelation-gated blend", 0.002, near_zero_threshold=0.001)
        == "Other non-topological gate"
    )
    assert (
        classify_winner("Always-on 50/50 blend", 0.002, near_zero_threshold=0.001)
        == "Always-on"
    )
    assert (
        classify_winner(
            "Adaptive recent-pattern forecaster", 0.002, near_zero_threshold=0.001
        )
        == "Adaptive alone"
    )
    assert (
        classify_winner("Frozen forecaster", 0.002, near_zero_threshold=0.001)
        == "Frozen"
    )
