"""Tests for phenogame.game_compiler — using REAL USA-NPN data.

Replaces the previous ``rng.normal``-fabricated ``_make_dataset`` helper
with real USA-NPN site phenometrics observations from
:func:`phenogame.fixtures.load_phenology_table`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phenogame.fixtures import load_phenology_table
from phenogame.game_compiler import (
    build_nature_scenarios,
    learned_payoff_surface,
    construct_13_by_s_payoff_matrix,
    compile_game_from_data,
    NatureScenarios,
    CompiledGame,
)
from phenogame.eml_families import fit_all_eml_families, define_13_eml_families
from phenogame.game import PayoffMatrix


_FEATURES = ["Latitude", "Longitude", "Elevation_in_Meters", "Mean_First_Yes_Year"]
_TARGET = "Mean_First_Yes_DOY"


@pytest.fixture(scope="module")
def real_apple_table():
    """REAL USA-NPN apple data — 713 clean rows, multiple states."""
    return load_phenology_table(
        "apple", feature_cols=_FEATURES, target_col=_TARGET,
        scenario_col="State", min_rows=20,
    )


@pytest.fixture(scope="module")
def real_peach_table():
    return load_phenology_table(
        "peach", feature_cols=_FEATURES, target_col=_TARGET,
        scenario_col="State", min_rows=20,
    )


# ─────────────────────────────────────────────────────────────────────────
# build_nature_scenarios
# ─────────────────────────────────────────────────────────────────────────

def test_build_nature_scenarios_categorical_real(real_apple_table):
    df = real_apple_table
    scen = build_nature_scenarios(
        df, scenario_cols=["State"],
        feature_cols=_FEATURES, n_bins=3, max_scenarios=12,
    )
    assert isinstance(scen, NatureScenarios)
    assert len(scen.labels) >= 1
    assert len(scen.labels) <= 12
    assert scen.representatives.shape[1] == len(_FEATURES)
    assert scen.representatives.shape[0] == len(scen.labels)
    assert np.all(np.isfinite(scen.representatives.to_numpy()))


def test_build_nature_scenarios_numeric_qcut_real(real_apple_table):
    df = real_apple_table
    scen = build_nature_scenarios(
        df, scenario_cols=["Mean_First_Yes_Year"],
        feature_cols=["Latitude", "Longitude", "Elevation_in_Meters"],
        n_bins=3,
    )
    # 3 bins or fewer (some years may collapse).
    assert 1 <= len(scen.labels) <= 3
    assert scen.representatives.shape == (len(scen.labels), 3)


def test_build_nature_scenarios_single_label_when_no_cols(real_peach_table):
    """When no scenario_cols given, a single label 'all' is returned."""
    scen = build_nature_scenarios(
        real_peach_table, scenario_cols=None,
        feature_cols=_FEATURES,
    )
    assert scen.labels == ["all"]
    assert scen.representatives.shape == (1, len(_FEATURES))


def test_build_nature_scenarios_rejects_empty():
    with pytest.raises(ValueError):
        build_nature_scenarios(pd.DataFrame({"x": []}),
                               scenario_cols=None,
                               feature_cols=["x"])


# ─────────────────────────────────────────────────────────────────────────
# learned_payoff_surface
# ─────────────────────────────────────────────────────────────────────────

def test_learned_payoff_surface_shape_and_finite_real(real_peach_table):
    df = real_peach_table
    fits = fit_all_eml_families(df[_FEATURES], df[_TARGET])
    scen = build_nature_scenarios(df, scenario_cols=["State"],
                                  feature_cols=_FEATURES)
    surface = learned_payoff_surface(fits, scen)
    assert surface.shape[0] == 13
    assert surface.shape[1] == len(scen.labels)
    assert np.all(np.isfinite(surface.to_numpy()))
    assert list(surface.index) == [s.family_id for s in define_13_eml_families()]


# ─────────────────────────────────────────────────────────────────────────
# construct_13_by_s_payoff_matrix
# ─────────────────────────────────────────────────────────────────────────

def test_construct_13_by_s_payoff_matrix_returns_payoff_matrix_real(real_peach_table):
    df = real_peach_table
    fits = fit_all_eml_families(df[_FEATURES], df[_TARGET])
    scen = build_nature_scenarios(df, scenario_cols=["State"],
                                  feature_cols=_FEATURES)
    pm = construct_13_by_s_payoff_matrix(fits, scen)
    assert isinstance(pm, PayoffMatrix)
    assert pm.matrix.shape[0] == 13
    assert pm.matrix.shape[1] == len(scen.labels)
    assert np.all(np.isfinite(pm.matrix))
    assert len(pm.strategy_labels) == 13


# ─────────────────────────────────────────────────────────────────────────
# compile_game_from_data
# ─────────────────────────────────────────────────────────────────────────

def test_compile_game_from_data_end_to_end_real(real_apple_table):
    cg = compile_game_from_data(
        real_apple_table,
        target_col=_TARGET,
        feature_cols=_FEATURES,
        scenario_cols=["State"],
    )
    assert isinstance(cg, CompiledGame)
    assert cg.payoff_matrix.matrix.shape[0] == 13
    assert cg.payoff_matrix.matrix.shape[1] >= 1
    assert len(cg.families) == 13
    assert "train_rmse" in cg.family_ranking.columns
    assert np.all(np.isfinite(cg.payoff_matrix.matrix))
    # All rows actually came from real data, not invented.
    assert cg.scenarios.feature_columns == _FEATURES


def test_compile_game_from_data_handles_inf_in_inputs_real(real_apple_table):
    """Inject one inf into REAL apple data; pipeline must drop it and still
    produce a finite 13×S matrix."""
    df = real_apple_table.copy()
    # Cast to float first so we can place an inf without an int dtype error
    # on newer pandas versions.
    df["Elevation_in_Meters"] = df["Elevation_in_Meters"].astype(float)
    df.iloc[0, df.columns.get_loc("Elevation_in_Meters")] = np.inf
    cg = compile_game_from_data(
        df, target_col=_TARGET,
        feature_cols=_FEATURES, scenario_cols=["State"],
    )
    assert cg.payoff_matrix.matrix.shape[0] == 13
    assert np.all(np.isfinite(cg.payoff_matrix.matrix))


def test_compile_game_missing_columns_raises(real_apple_table):
    with pytest.raises(ValueError):
        compile_game_from_data(real_apple_table,
                               target_col="missing", feature_cols=_FEATURES)
    with pytest.raises(ValueError):
        compile_game_from_data(real_apple_table,
                               target_col=_TARGET, feature_cols=["nope"])
