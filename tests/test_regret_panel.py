"""Tests for phenogame.regret_panel."""

from __future__ import annotations

import numpy as np
import pytest

from phenogame.fixtures import load_phenology_table
from phenogame.game_compiler import compile_game_from_data
from phenogame.regret_panel import (
    compare_regret_transforms,
    RegretComparisonResult, TransformAggregate, TransformRun,
)


# Canonical 2×2 zero-sum game: matching pennies.
MATCHING_PENNIES = np.array([
    [1.0, -1.0],
    [-1.0, 1.0],
])


def test_compare_returns_typed_result():
    r = compare_regret_transforms(MATCHING_PENNIES,
                                  n_seeds=2, iterations=200, base_seed=0)
    assert isinstance(r, RegretComparisonResult)
    assert r.payoff_shape == (2, 2)
    assert r.n_seeds == 2
    assert r.iterations == 200
    assert len(r.transforms) == 4
    assert len(r.runs) == 8  # 4 transforms × 2 seeds
    assert len(r.aggregates) == 4


def test_run_records_have_finite_metrics():
    r = compare_regret_transforms(MATCHING_PENNIES,
                                  n_seeds=2, iterations=100, base_seed=0)
    for run in r.runs:
        assert np.isfinite(run.final_cce_gap)
        assert run.final_cce_gap >= 0.0
        assert np.isfinite(run.joint_corr_normalized)
        assert 0.0 <= run.joint_corr_normalized <= 1.0
        assert run.wall_clock_seconds >= 0.0
        assert isinstance(run, TransformRun)


def test_aggregates_have_correct_n_seeds():
    r = compare_regret_transforms(MATCHING_PENNIES,
                                  n_seeds=4, iterations=100, base_seed=0)
    for agg in r.aggregates:
        assert isinstance(agg, TransformAggregate)
        assert agg.n_seeds == 4
        assert np.isfinite(agg.final_cce_gap_mean)
        assert agg.final_cce_gap_std >= 0.0


def test_to_dataframe_columns():
    r = compare_regret_transforms(MATCHING_PENNIES,
                                  n_seeds=2, iterations=100, base_seed=0)
    df = r.to_dataframe()
    assert len(df) == 4
    for col in ["transform", "n_seeds", "final_cce_gap_mean",
                "final_cce_gap_std", "joint_corr_norm_mean",
                "wall_clock_s_mean", "no_regret_proof",
                "is_strict_rm", "always_positive"]:
        assert col in df.columns


def test_summary_contains_audit_findings():
    r = compare_regret_transforms(MATCHING_PENNIES,
                                  n_seeds=2, iterations=100, base_seed=0)
    s = r.summary()
    # Audit findings F2/F4/F6/F7/F8/F9/F10 are documented in the panel notes.
    for f in ["F2", "F4", "F6", "F7", "F8", "F9", "F10"]:
        assert f in s


def test_subset_of_transforms_works():
    r = compare_regret_transforms(MATCHING_PENNIES,
                                  transforms=("standard", "eml"),
                                  n_seeds=2, iterations=100, base_seed=0)
    assert r.transforms == ["standard", "eml"]
    assert len(r.aggregates) == 2


def test_unknown_transform_raises():
    with pytest.raises(KeyError):
        compare_regret_transforms(MATCHING_PENNIES,
                                  transforms=("bogus",),
                                  n_seeds=1, iterations=10)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        compare_regret_transforms(MATCHING_PENNIES, iterations=0, n_seeds=1)
    with pytest.raises(ValueError):
        compare_regret_transforms(MATCHING_PENNIES, n_seeds=0, iterations=100)
    bad = np.array([[1.0, np.nan], [0.0, 1.0]])
    with pytest.raises(ValueError):
        compare_regret_transforms(bad, n_seeds=1, iterations=100)


def test_reproducibility_byte_identical():
    """Same base_seed must produce identical run records."""
    r1 = compare_regret_transforms(MATCHING_PENNIES,
                                   n_seeds=3, iterations=200, base_seed=42)
    r2 = compare_regret_transforms(MATCHING_PENNIES,
                                   n_seeds=3, iterations=200, base_seed=42)
    for a, b in zip(r1.runs, r2.runs):
        assert a.transform == b.transform
        assert a.seed == b.seed
        assert a.final_cce_gap == b.final_cce_gap
        assert a.iters_to_epsilon == b.iters_to_epsilon
        assert a.joint_corr_normalized == b.joint_corr_normalized


def test_paired_seeding_F7():
    """Audit F7: paired comparison ⇒ for the same run index, all
    transforms see the same RNG-derived seed offset structure."""
    r = compare_regret_transforms(MATCHING_PENNIES,
                                  n_seeds=2, iterations=100, base_seed=100)
    # Per-transform seeds: standard {100, 101}, exp {1100, 1101}, etc.
    by_transform = {}
    for run in r.runs:
        by_transform.setdefault(run.transform, []).append(run.seed)
    # Each transform has 2 distinct seeds, ordered.
    for name, seeds in by_transform.items():
        assert len(seeds) == 2
        assert seeds[0] < seeds[1]


def test_panel_on_real_data_grape():
    """Run the comparator on a real USA-NPN data-induced game."""
    df = load_phenology_table("grape")
    cg = compile_game_from_data(
        df, target_col="Mean_First_Yes_DOY",
        feature_cols=["Latitude", "Longitude",
                      "Elevation_in_Meters", "Mean_First_Yes_Year"],
        scenario_cols=["State"], base_seed=42,
    )
    r = compare_regret_transforms(cg.payoff_matrix,
                                  n_seeds=2, iterations=200, base_seed=0)
    assert r.payoff_shape == (13, 4)
    # Just verify it runs end-to-end and produces finite numbers.
    for agg in r.aggregates:
        assert np.isfinite(agg.final_cce_gap_mean)
        assert np.isfinite(agg.joint_corr_normalized_mean)


def test_softplus_F6_recorded_in_aggregate():
    """Audit F6 metadata must surface in the aggregate."""
    r = compare_regret_transforms(MATCHING_PENNIES,
                                  n_seeds=2, iterations=100, base_seed=0)
    softplus_agg = next(a for a in r.aggregates if a.transform == "softplus")
    assert softplus_agg.always_positive is True
    assert softplus_agg.is_strict_rm is False


def test_eml_F9_recorded_in_aggregate():
    """Audit F9 metadata must surface in the aggregate."""
    r = compare_regret_transforms(MATCHING_PENNIES,
                                  n_seeds=2, iterations=100, base_seed=0)
    eml_agg = next(a for a in r.aggregates if a.transform == "eml")
    assert "empirical" in eml_agg.no_regret_proof.lower()
