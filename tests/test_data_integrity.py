"""Integrity audit: bundled USA-NPN datasheet ZIPs == bundled CSVs.

These tests verify that the bundled USA-NPN datasheet ZIPs in
``phenogame/data/usa_npn_datasheets/`` contain exactly the same rows as
the bundled species-split CSVs in ``phenogame/data/npn_*.csv``. If
anything in either form gets corrupted, these tests fail.

This is the data-integrity audit the package promises in
``phenogame/data/usa_npn_datasheets/README.md``.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest


_DATA_DIR = Path(__file__).resolve().parent.parent / "phenogame" / "data"
_DS_DIR = _DATA_DIR / "usa_npn_datasheets"

# Canonical "all four species" superset zip.
_SUPERSET_ZIP = _DS_DIR / "datasheet_1777767104274.zip"

# Mapping bundled CSV → expected USA-NPN Common_Name in superset.
_CSV_TO_NAME = {
    "npn_apple.csv":     "apple",
    "npn_grape.csv":     "wine grape",
    "npn_peach.csv":     "peach",
    "npn_red_maple.csv": "red maple",
}

# Stable join key for row-level equality.
_KEY_COLS = ["Site_ID", "Phenophase_ID", "Mean_First_Yes_Year",
             "Mean_First_Yes_DOY", "Mean_AGDD"]


def _read_superset() -> pd.DataFrame:
    """Read the all-four-species superset CSV out of the bundled zip."""
    with zipfile.ZipFile(_SUPERSET_ZIP) as z:
        # Find the single site_phenometrics_data.csv inside the archive.
        candidates = [n for n in z.namelist()
                      if n.endswith("site_phenometrics_data.csv")]
        assert len(candidates) == 1, f"Unexpected zip layout: {z.namelist()}"
        with z.open(candidates[0]) as f:
            return pd.read_csv(io.BytesIO(f.read()))


@pytest.fixture(scope="module")
def superset() -> pd.DataFrame:
    return _read_superset()


def test_all_five_datasheets_present():
    expected = {
        "datasheet_1777766690685.zip",
        "datasheet_1777766770140.zip",
        "datasheet_1777766839978.zip",
        "datasheet_1777766957944.zip",
        "datasheet_1777767104274.zip",
    }
    actual = {p.name for p in _DS_DIR.glob("*.zip")}
    assert expected <= actual, f"Missing bundled datasheets: {expected - actual}"


def test_superset_loads_with_expected_total_rows(superset):
    assert len(superset) == 12215, (
        f"Superset row count drifted: got {len(superset)}, expected 12215. "
        "The bundled USA-NPN datasheet may have been altered."
    )


def test_superset_columns_match_bundled_schema(superset):
    """The superset's column schema must match the bundled CSVs."""
    bundled = pd.read_csv(_DATA_DIR / "npn_grape.csv")
    assert list(superset.columns) == list(bundled.columns)


@pytest.mark.parametrize("csv_name,common_name", list(_CSV_TO_NAME.items()))
def test_bundled_csv_equals_superset_subset(csv_name, common_name, superset):
    """Each bundled CSV must equal the corresponding species subset of the
    superset on the stable join keys."""
    bundled = pd.read_csv(_DATA_DIR / csv_name)
    subset = superset[superset["Common_Name"] == common_name]
    assert len(bundled) == len(subset), (
        f"{csv_name}: bundled rows={len(bundled)}, superset subset={len(subset)}"
    )
    b = bundled[_KEY_COLS].sort_values(_KEY_COLS).reset_index(drop=True)
    s = subset[_KEY_COLS].sort_values(_KEY_COLS).reset_index(drop=True)
    pd.testing.assert_frame_equal(b, s)


def test_no_extraneous_synthetic_rows_in_bundled_data():
    """No bundled CSV may contain a row whose Site_ID is not present in the
    USA-NPN superset. Catches accidental fabricated additions."""
    superset = _read_superset()
    superset_sites = set(superset["Site_ID"].unique())
    for csv_name in _CSV_TO_NAME:
        bundled = pd.read_csv(_DATA_DIR / csv_name)
        bundled_sites = set(bundled["Site_ID"].unique())
        extras = bundled_sites - superset_sites
        assert not extras, (
            f"{csv_name} contains {len(extras)} sites not in USA-NPN superset: "
            f"{sorted(extras)[:10]}"
        )
