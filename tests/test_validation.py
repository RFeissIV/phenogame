"""Tests for the train/test validation and baseline-comparison module.

The previous ``_toy_dataset`` synthetic helper has been removed in favour
of real USA-NPN observations from
:func:`phenogame.fixtures.load_phenology_table`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import pytest

from phenogame import (
    BaselineComparison,
    ModelScore,
    action_stability,
    compare_baselines,
    train_test_split_indices,
)
from phenogame.fixtures import load_phenology_table


_FEATURES = ["Latitude", "Longitude", "Elevation_in_Meters", "Mean_First_Yes_Year"]
_TARGET = "Mean_First_Yes_DOY"


def _real_action_scenario_table(species: str = "peach",
                                n_max: Optional[int] = None) -> pd.DataFrame:
    """Real USA-NPN data reshaped for the validation API: 'action' = state,
    'scenario' = year, x1..x4 = lat/lon/elev/year, payoff = first-yes DOY.

    Nothing here is fabricated — every row is a real observed phenology
    measurement for the requested species.
    """
    df = load_phenology_table(species, feature_cols=_FEATURES,
                              target_col=_TARGET, scenario_col="State",
                              min_rows=20)
    out = pd.DataFrame({
        "action": df["State"].astype(str),
        "scenario": df["Mean_First_Yes_Year"].astype(int),
        "x1": df["Latitude"].astype(float),
        "x2": df["Longitude"].astype(float),
        "x3": df["Elevation_in_Meters"].astype(float),
        "x4": df["Mean_First_Yes_Year"].astype(float),
        "payoff": df["Mean_First_Yes_DOY"].astype(float),
    })
    if n_max is not None:
        out = out.head(n_max)
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# train/test split (data-independent)
# ---------------------------------------------------------------------------

def test_train_test_split_disjoint_and_covers_all_indices():
    train, test = train_test_split_indices(40, test_fraction=0.25, seed=7)
    assert len(train) + len(test) == 40
    assert set(train).isdisjoint(set(test))
    assert set(train) | set(test) == set(range(40))


def test_train_test_split_is_deterministic():
    a = train_test_split_indices(50, test_fraction=0.3, seed=42)
    b = train_test_split_indices(50, test_fraction=0.3, seed=42)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_train_test_split_validates_inputs():
    with pytest.raises(ValueError):
        train_test_split_indices(2)
    with pytest.raises(ValueError):
        train_test_split_indices(20, test_fraction=0.0)
    with pytest.raises(ValueError):
        train_test_split_indices(20, test_fraction=1.0)


# ---------------------------------------------------------------------------
# baseline comparison — REAL DATA
# ---------------------------------------------------------------------------

def test_compare_baselines_returns_expected_models_real():
    df = _real_action_scenario_table()
    comp = compare_baselines(
        df, feature_columns=["x1", "x2"], target_column="payoff",
        test_fraction=0.25, seed=3, n_eml_features=24,
        include_random_forest=False,
    )
    assert isinstance(comp, BaselineComparison)
    names = [s.name for s in comp.scores]
    assert "naive_mean" in names
    assert "ridge_linear" in names
    assert "eml_tree_ridge" in names
    for s in comp.scores:
        assert isinstance(s, ModelScore)
        assert np.isfinite(s.train_rmse)
        assert np.isfinite(s.test_rmse)
        assert np.isfinite(s.test_mae)
        assert np.isfinite(s.test_r2)


def test_compare_baselines_dataframe_and_summary_real():
    df = _real_action_scenario_table()
    comp = compare_baselines(
        df, feature_columns=["x1", "x2"], target_column="payoff",
        seed=1, n_eml_features=16, include_random_forest=False,
    )
    table = comp.to_dataframe()
    assert {"model", "n_train", "n_test", "test_rmse", "test_r2"} <= set(table.columns)
    assert "Baseline comparison" in comp.summary()


def test_compare_baselines_eml_tree_beats_naive_on_real_data():
    """EML-tree must beat the naive-mean floor on real peach phenology
    using the full lat/lon/elev/year feature set."""
    df = _real_action_scenario_table("peach")
    comp = compare_baselines(
        df, feature_columns=["x1", "x2", "x3", "x4"], target_column="payoff",
        test_fraction=0.25, seed=11, n_eml_features=64,
        include_random_forest=False,
    )
    by_name = {s.name: s for s in comp.scores}
    assert by_name["eml_tree_ridge"].test_rmse < by_name["naive_mean"].test_rmse


def test_compare_baselines_rejects_missing_columns():
    df = _real_action_scenario_table(n_max=60)
    with pytest.raises(ValueError):
        compare_baselines(
            df, feature_columns=["x1", "missing"], target_column="payoff",
            include_random_forest=False,
        )


# ---------------------------------------------------------------------------
# action stability — REAL DATA
# ---------------------------------------------------------------------------

def test_action_stability_runs_and_returns_valid_frequencies_real():
    df = _real_action_scenario_table("peach")
    res = action_stability(
        df,
        strategy_column="action",
        scenario_column="scenario",
        feature_columns=["x1", "x2"],
        target_column="payoff",
        n_resamples=4,
        n_eml_features=16,
        iterations=20,
        seed=4,
    )
    assert res.n_resamples == 4
    assert 0.0 <= res.stability <= 1.0
    assert 0.0 <= res.most_common_frequency <= 1.0
    assert sum(res.action_counts.values()) == res.n_resamples
    assert "Action stability" in res.summary()


def test_action_stability_requires_columns_and_enough_data():
    df = _real_action_scenario_table("peach", n_max=8)
    with pytest.raises(ValueError):
        action_stability(
            df,
            strategy_column="action",
            scenario_column="scenario",
            feature_columns=["x1", "missing_feat"],
            target_column="payoff",
            n_resamples=3,
        )
    with pytest.raises(ValueError):
        action_stability(
            df.head(2),
            strategy_column="action",
            scenario_column="scenario",
            feature_columns=["x1", "x2"],
            target_column="payoff",
            n_resamples=3,
        )
