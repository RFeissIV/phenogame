"""Tests for phenogame.fixtures — verifies real USA-NPN data loads.

These tests verify that the bundled USA-NPN site phenometrics CSVs are
present, parseable, and produce non-empty cleaned tables. They also
verify the row counts for each species so that any silent corruption of
the bundled data would fail loudly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phenogame.fixtures import (
    available_species,
    load_real_npn,
    load_phenology_table,
    real_score_matrix_from_models,
)


# Expected RAW row counts (before -9999 sentinel filtering) for the bundled
# USA-NPN site phenometrics CSVs. These match the "all four species"
# datasheet superset (12,215 rows total).
EXPECTED_RAW_ROWS = {
    "apple": 843,
    "grape": 130,
    "peach": 329,
    "red_maple": 10913,
}


# Expected CLEAN row counts after sentinel + numeric filtering (-9999 removed,
# Mean_First_Yes_DOY > 0, Mean_AGDD valid). These are the rows the EML game
# pipeline actually consumes.
EXPECTED_CLEAN_ROWS = {
    "apple": 713,
    "grape": 121,
    "peach": 301,
    "red_maple": 9273,
}


def test_available_species_returns_all_four():
    species = available_species()
    assert set(species) == {"apple", "grape", "peach", "red_maple"}


@pytest.mark.parametrize("species", ["apple", "grape", "peach", "red_maple"])
def test_load_real_npn_clean_row_counts(species):
    """Bundled CSVs must produce expected row counts after -9999 filtering."""
    df = load_real_npn(species)
    assert len(df) == EXPECTED_CLEAN_ROWS[species], (
        f"{species}: got {len(df)} clean rows, expected "
        f"{EXPECTED_CLEAN_ROWS[species]}. The bundled CSV may have been altered."
    )
    # Required NPN columns must be present.
    for col in ["Site_ID", "Latitude", "Longitude", "State",
                "Common_Name", "Phenophase_Description",
                "Mean_First_Yes_DOY", "Mean_AGDD", "Mean_First_Yes_Year"]:
        assert col in df.columns


@pytest.mark.parametrize("species", ["apple", "grape", "peach", "red_maple"])
def test_load_real_npn_no_sentinel_values(species):
    df = load_real_npn(species)
    assert (df["Mean_First_Yes_DOY"] != -9999).all()
    assert (df["Mean_AGDD"] != -9999).all()


def test_load_real_npn_unknown_species_raises():
    with pytest.raises(KeyError):
        load_real_npn("unobtainium")


def test_load_phenology_table_default_works():
    df = load_phenology_table()  # red_maple default
    assert len(df) >= 1000
    for col in ["Latitude", "Longitude", "Elevation_in_Meters",
                "Mean_First_Yes_Year", "Mean_First_Yes_DOY", "State"]:
        assert col in df.columns
    assert np.all(np.isfinite(df.drop(columns=["State"]).to_numpy(dtype=float)))


def test_load_phenology_table_grape_has_states():
    df = load_phenology_table("grape")
    states = set(df["State"].unique())
    # Wine grape data is from CA, NY, OR, MA per USA-NPN export.
    assert states.issubset({"CA", "NY", "OR", "MA"})
    assert len(states) >= 2


def test_load_phenology_table_min_rows_enforced():
    with pytest.raises(ValueError):
        load_phenology_table("grape", min_rows=99999)


def test_real_score_matrix_shape_and_finite():
    sm = real_score_matrix_from_models("peach", n_repetitions=4, seed=0)
    assert sm.shape == (4, 13)
    assert list(sm.columns) == [f"F{i:02d}" for i in range(1, 14)]
    # Negative training RMSE; bootstrap repetitions must be finite.
    assert np.all(np.isfinite(sm.to_numpy()))
    assert (sm <= 0.0).all().all()  # since values are -RMSE


def test_real_score_matrix_reproducibility():
    sm1 = real_score_matrix_from_models("peach", n_repetitions=3, seed=42)
    sm2 = real_score_matrix_from_models("peach", n_repetitions=3, seed=42)
    pd.testing.assert_frame_equal(sm1, sm2)
