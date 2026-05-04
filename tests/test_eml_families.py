"""Tests for phenogame.eml_families — using REAL USA-NPN data.

Replaces the previous ``rng.normal``-fabricated ``_make_data`` helper with
real USA-NPN site phenometrics observations loaded via
:func:`phenogame.fixtures.load_phenology_table`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phenogame.eml_families import (
    define_13_eml_families,
    fit_eml_family,
    fit_all_eml_families,
    predict_eml_family,
    rank_eml_families,
    FittedEMLFamily,
    EMLFamilySpec,
)
from phenogame.fixtures import load_phenology_table


_FEATURES = ["Latitude", "Longitude", "Elevation_in_Meters", "Mean_First_Yes_Year"]
_TARGET = "Mean_First_Yes_DOY"


@pytest.fixture(scope="module")
def real_xy_peach():
    """Real USA-NPN peach data: 301 clean rows after sentinel filtering."""
    df = load_phenology_table(
        "peach", feature_cols=_FEATURES, target_col=_TARGET,
        scenario_col="State", min_rows=20,
    )
    return df[_FEATURES], df[_TARGET]


@pytest.fixture(scope="module")
def real_xy_apple():
    """Real USA-NPN apple data: 713 clean rows after sentinel filtering."""
    df = load_phenology_table(
        "apple", feature_cols=_FEATURES, target_col=_TARGET,
        scenario_col="State", min_rows=20,
    )
    return df[_FEATURES], df[_TARGET]


def test_define_13_eml_families_returns_13_distinct():
    fams = define_13_eml_families()
    assert len(fams) == 13
    ids = [f.family_id for f in fams]
    assert len(set(ids)) == 13
    for f in fams:
        assert isinstance(f, EMLFamilySpec)
        assert f.family_id.startswith("F")
        assert f.backbone in {"linear", "eml_tree", "ridge_eml"}


def test_fit_eml_family_basic_real(real_xy_peach):
    X, y = real_xy_peach
    fit = fit_eml_family(X, y, "F03")
    assert isinstance(fit, FittedEMLFamily)
    assert fit.spec.family_id == "F03"
    assert fit.n_rows == len(X)
    assert np.isfinite(fit.train_rmse)
    assert fit.train_rmse >= 0.0


def test_fit_eml_family_unknown_id_raises(real_xy_peach):
    X, y = real_xy_peach
    with pytest.raises(KeyError):
        fit_eml_family(X, y, "F99")


def test_fit_eml_family_rejects_nonfinite_inputs(real_xy_peach):
    X, y = real_xy_peach
    Xbad = X.copy()
    Xbad.iloc[0, 0] = np.inf
    with pytest.raises(ValueError):
        fit_eml_family(Xbad, y, "F03")


def test_fit_all_eml_families_completes_for_all_13_real(real_xy_peach):
    X, y = real_xy_peach
    fits = fit_all_eml_families(X, y)
    assert len(fits) == 13
    assert set(fits.keys()) == {f.family_id for f in define_13_eml_families()}
    for fit in fits.values():
        assert isinstance(fit, FittedEMLFamily)
        assert np.isfinite(fit.train_rmse)


def test_predict_eml_family_finite_outputs_real(real_xy_apple):
    X, y = real_xy_apple
    fit = fit_eml_family(X, y, "F04")
    preds = predict_eml_family(fit, X)
    assert preds.shape == (len(X),)
    assert np.all(np.isfinite(preds))


def test_predict_handles_held_out_real_rows(real_xy_apple):
    """Train on first 80% of REAL apple observations, predict on the held-out
    20%; predictions must be finite."""
    X, y = real_xy_apple
    n = len(X)
    cut = int(0.8 * n)
    fit = fit_eml_family(X.iloc[:cut], y.iloc[:cut], "F02")
    preds = predict_eml_family(fit, X.iloc[cut:])
    assert preds.shape == (n - cut,)
    assert np.all(np.isfinite(preds))


def test_rank_eml_families_orders_by_rmse_real(real_xy_peach):
    X, y = real_xy_peach
    fits = fit_all_eml_families(X, y)
    ranking = rank_eml_families(fits)
    assert len(ranking) == 13
    rmse = ranking["train_rmse"].to_numpy()
    assert np.all(np.diff(rmse) >= -1e-12)
    for col in ["family_id", "name", "backbone", "n_features",
                "alpha", "train_rmse", "n_rows"]:
        assert col in ranking.columns


def test_seed_diversity_F09_F10_differ_from_F03_real(real_xy_peach):
    """F03, F09, F10 share architecture but use different seeds — predictions
    on REAL peach data must not be byte-identical."""
    X, y = real_xy_peach
    f03 = fit_eml_family(X, y, "F03")
    f09 = fit_eml_family(X, y, "F09")
    f10 = fit_eml_family(X, y, "F10")
    p03 = predict_eml_family(f03, X)
    p09 = predict_eml_family(f09, X)
    p10 = predict_eml_family(f10, X)
    assert not np.allclose(p03, p09)
    assert not np.allclose(p03, p10)
    assert not np.allclose(p09, p10)
