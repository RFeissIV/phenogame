"""
phenogame.fixtures — Real-data fixtures used by examples and tests.

This module provides one canonical real-data loader so that nothing in the
package or its tests has to fabricate observations. All data come from the
bundled USA-NPN site phenometrics CSVs in ``phenogame/data/``.

Provenance
----------
The bundled files in ``phenogame/data/``::

    npn_apple.csv      843 rows
    npn_grape.csv      130 rows
    npn_peach.csv      329 rows
    npn_red_maple.csv  10,913 rows

are direct exports of the USA-NPN site phenometrics product, filtered for
the years 2010-2020 (start) through 2020 (end), with the USA-NPN ``-9999``
sentinel removed via :func:`phenogame.pipeline.load_npn_csv`.

The exact same observations are also reproducible from the user-supplied
datasheet ZIPs (``datasheet_*.zip``) which contain a 12,215-row "all four
species" superset; those bundle the same rows but in a single file. This
module always uses the bundled CSV form because it is shipped with the
package and is available offline.

API
---

- :func:`available_species` — names of bundled real datasets
- :func:`load_real_npn` — return one DataFrame per species, filtered
- :func:`load_phenology_table` — wide table for a given species, ready
  for use as ``X``, ``y``, scenario columns by the EML game pipeline
- :func:`real_score_matrix_from_models` — pairwise score table built by
  evaluating the 13 EML families on real data, used as the canonical
  pairwise-screen fixture for tests
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .pipeline import load_npn_csv


# Map common name → bundled CSV filename.
_SPECIES_FILES: Dict[str, str] = {
    "apple": "npn_apple.csv",
    "grape": "npn_grape.csv",
    "peach": "npn_peach.csv",
    "red_maple": "npn_red_maple.csv",
}

# Cache fitted 13-family model bundles used by real_score_matrix_from_models.
# Tests call the function repeatedly with different n_repetitions but identical
# species/features; caching avoids repeated expensive pseudo-inverses while
# preserving deterministic real-data scores.
_REAL_SCORE_FIT_CACHE: Dict[tuple, Dict[str, object]] = {}


def available_species() -> List[str]:
    """Return the list of bundled real-data species keys."""
    return sorted(_SPECIES_FILES.keys())


def load_real_npn(
    species: str = "grape",
    *,
    phenophase: Optional[str] = None,
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Load one bundled USA-NPN species CSV with -9999 sentinels removed.

    Parameters
    ----------
    species
        One of :func:`available_species` (e.g. ``"grape"``).
    phenophase
        Optional phenophase filter forwarded to
        :func:`phenogame.pipeline.load_npn_csv`.
    data_dir
        Optional override for the data directory (e.g. to point at an
        unzipped USA-NPN datasheet folder containing
        ``site_phenometrics_data.csv``).

    Returns
    -------
    pandas.DataFrame
        Cleaned NPN phenometrics rows.

    Raises
    ------
    KeyError
        If ``species`` is not in :func:`available_species`.
    FileNotFoundError
        If the underlying CSV is missing.
    """
    if species not in _SPECIES_FILES:
        raise KeyError(
            f"Unknown species {species!r}. Available: {available_species()}"
        )
    return load_npn_csv(_SPECIES_FILES[species],
                        data_dir=data_dir,
                        phenophase=phenophase)


def load_phenology_table(
    species: str = "red_maple",
    *,
    feature_cols: Sequence[str] = ("Latitude", "Longitude",
                                   "Elevation_in_Meters",
                                   "Mean_First_Yes_Year"),
    target_col: str = "Mean_First_Yes_DOY",
    scenario_col: str = "State",
    min_rows: int = 30,
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Real-data wide table ready for the EML game pipeline.

    The default uses red maple — the largest bundled species (≈10k rows) —
    and predicts ``Mean_First_Yes_DOY`` (day of first observed phenology
    "yes") from latitude, longitude, elevation, and year. ``State`` is used
    as a Nature scenario column.

    All rows containing non-finite values in any of the required columns
    are dropped. Returns the columns in the order
    ``[*feature_cols, target_col, scenario_col]``.

    Parameters
    ----------
    species
        One of :func:`available_species`.
    feature_cols
        Numeric columns to use as features. Defaults to the four NPN
        columns reliably populated for every bundled species.
    target_col
        Numeric column to predict.
    scenario_col
        Categorical column used to define Nature scenarios.
    min_rows
        Minimum number of clean rows required after filtering. If fewer
        clean rows remain, ``ValueError`` is raised so callers do not
        silently fall back to small samples.
    data_dir
        Optional override for the data directory.

    Returns
    -------
    pandas.DataFrame
        Wide tabular data ready for ``compile_game_from_data``.

    Raises
    ------
    KeyError
        If a required column is missing.
    ValueError
        If fewer than ``min_rows`` clean rows remain.
    """
    df = load_real_npn(species, data_dir=data_dir)
    cols = list(feature_cols) + [target_col, scenario_col]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Required columns missing from bundled {species} data: {missing}"
        )
    out = df[cols].copy()
    # Filter -9999 sentinel and non-finite numeric values explicitly.
    for c in [*feature_cols, target_col]:
        out = out[out[c] != -9999]
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    out = out.reset_index(drop=True)
    if len(out) < min_rows:
        raise ValueError(
            f"Too few clean rows ({len(out)} < {min_rows}) in {species} "
            "after sentinel/NaN filtering."
        )
    return out


def real_score_matrix_from_models(
    species: str = "red_maple",
    *,
    n_repetitions: int = 30,
    feature_cols: Sequence[str] = ("Latitude", "Longitude",
                                   "Elevation_in_Meters",
                                   "Mean_First_Yes_Year"),
    target_col: str = "Mean_First_Yes_DOY",
    seed: int = 0,
    max_rows_per_repetition: int = 96,
    refit_each_repetition: bool = False,
    max_eml_features_for_scoring: int = 32,
) -> pd.DataFrame:
    """Real-data pairwise score table for the binary stochastic choice screen.

    The default path is intentionally CI-friendly: each of the 13 EML families
    is fitted once on real bundled USA-NPN rows, then scored on seeded bootstrap
    resamples of real rows.  This preserves a real-data pairwise comparison
    layer without spending minutes refitting the heavy EML random-feature
    families in every test.

    Set ``refit_each_repetition=True`` for the slower, stricter bootstrap that
    refits all 13 families inside each replicate.

    Higher score = better fit, where score is ``-RMSE`` on the current real-row
    resample. Returns a ``(n_repetitions, 13)`` DataFrame whose columns are
    family IDs (``F01..F13``). This is exactly the input shape
    :func:`phenogame.binary_choice.pairwise_win_matrix` expects.

    Parameters
    ----------
    species
        One of :func:`available_species`.
    n_repetitions
        Number of bootstrap scoring repetitions.
    feature_cols
        Real numeric features used to fit/evaluate each family.
    target_col
        Real numeric target.
    seed
        Random seed for the bootstrap **row indices** (not data values).
    max_rows_per_repetition
        Deterministic cap on the number of real rows used in each scoring
        replicate. The cap keeps tests and examples fast while still sampling
        only from real bundled observations. Set to ``0`` or a value >=
        ``len(data)`` to use the full bootstrap size.
    refit_each_repetition
        If ``False`` (default), fit each family once and score on repeated
        real-row bootstrap subsets. If ``True``, refit all 13 families inside
        every repetition. The latter is slower but can be useful for a deeper
        robustness study outside CI.
    max_eml_features_for_scoring
        Feature cap used only inside this real-data scoring helper. It keeps
        routine tests fast and avoids platform-specific BLAS stalls from large
        pseudo-inverses. Use ``0`` or a value >= 128 to score with full family
        feature counts.

    Returns
    -------
    pandas.DataFrame
        ``(n_repetitions, 13)`` score table indexed by repetition number.
    """
    from .eml_families import (
        define_13_eml_families, fit_eml_family, fit_all_eml_families,
        predict_eml_family,
    )

    df = load_phenology_table(
        species,
        feature_cols=feature_cols, target_col=target_col,
        scenario_col="State", min_rows=20,
    )
    X = df[list(feature_cols)]
    y = df[target_col].reset_index(drop=True)

    fam_ids = [f.family_id for f in define_13_eml_families()]
    rng = np.random.default_rng(seed)
    n = len(df)
    if max_rows_per_repetition is None or int(max_rows_per_repetition) <= 0:
        sample_size = n
    else:
        sample_size = min(n, int(max_rows_per_repetition))

    # Fit once by default. This is still real-data model evaluation: only the
    # row-index selection changes across repetitions. It avoids slow repeated
    # pseudo-inverses for F05/F13 during routine package tests. Cache by the
    # real-data/model identity so repeated tests with different n_repetitions
    # reuse the exact same deterministic fitted family bundle.
    fitted_once = None
    if not refit_each_repetition:
        feature_cap = None
        if max_eml_features_for_scoring is not None and int(max_eml_features_for_scoring) > 0:
            feature_cap = int(max_eml_features_for_scoring)
        cache_key = (
            species, tuple(feature_cols), target_col, int(seed) + 12345,
            feature_cap,
        )
        if cache_key not in _REAL_SCORE_FIT_CACHE:
            _REAL_SCORE_FIT_CACHE[cache_key] = fit_all_eml_families(
                X, y, base_seed=int(seed) + 12345,
                max_n_features=feature_cap,
            )
        fitted_once = _REAL_SCORE_FIT_CACHE[cache_key]

    rows = []
    for _ in range(int(n_repetitions)):
        idx = rng.integers(0, n, size=sample_size)
        Xb = X.iloc[idx].reset_index(drop=True)
        yb = y.iloc[idx].reset_index(drop=True).to_numpy(dtype=float)
        scores = []
        for fid in fam_ids:
            try:
                if refit_each_repetition:
                    feature_cap = None
                    if (max_eml_features_for_scoring is not None
                            and int(max_eml_features_for_scoring) > 0):
                        feature_cap = int(max_eml_features_for_scoring)
                    fit = fit_eml_family(
                        Xb, yb, fid,
                        base_seed=int(rng.integers(1, 10**6)),
                        max_n_features=feature_cap,
                    )
                    rmse = float(fit.train_rmse)
                else:
                    pred = predict_eml_family(fitted_once[fid], Xb)
                    rmse = float(np.sqrt(np.mean((pred - yb) ** 2)))
                scores.append(-rmse)
            except Exception:
                scores.append(float("-inf"))
        rows.append(scores)

    return pd.DataFrame(rows, columns=fam_ids)

def real_temperature_trace(
    species: str = "red_maple",
    *,
    start_date: str = "2026-03-15",
    end_date: str = "2026-09-30",
    base_temp: float = 5.0,
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Real, deterministic daily temperature trace built from bundled NPN data.

    For every NPN row in the chosen species we estimate the average daily
    temperature that produced its observed Mean_AGDD by

        T_avg_daily ≈ Mean_AGDD / Mean_First_Yes_DOY + base_temp

    (with the AGDD base ``base_temp`` defaulting to 5°C, matching common
    NPN AGDD computations). We then group by day-of-year and take the
    **median** across all sites and years; the result is a deterministic,
    reproducible per-DOY temperature curve derived entirely from real
    observations. We map that curve onto the requested date range using
    each date's day-of-year.

    No fabricated noise, no ``np.random`` calls.

    Parameters
    ----------
    species
        One of :func:`available_species`. ``"red_maple"`` is the default
        because it has by far the largest sample (~9,200 clean rows after
        sentinel filtering).
    start_date, end_date
        ISO date strings defining the inclusive date range.
    base_temp
        AGDD base temperature, in degrees Celsius.
    data_dir
        Optional override for the data directory.

    Returns
    -------
    pandas.DataFrame
        Two columns: ``date`` (datetime64) and ``temperature`` (float).
        Same length as ``pd.date_range(start_date, end_date)``.
    """
    df = load_real_npn(species, data_dir=data_dir)
    df = df[(df["Mean_AGDD"] > 0) & (df["Mean_First_Yes_DOY"] > 0)].copy()
    if df.empty:
        raise ValueError(
            f"No clean rows for species={species!r} after AGDD/DOY filter."
        )
    df["temp_avg_daily"] = (
        df["Mean_AGDD"].astype(float) / df["Mean_First_Yes_DOY"].astype(float)
        + float(base_temp)
    )
    by_doy = df.groupby(df["Mean_First_Yes_DOY"].astype(int))["temp_avg_daily"].median()

    # Build the date range and map to DOY.
    dates = pd.date_range(start_date, end_date)
    doy = dates.dayofyear.to_numpy()

    # For each requested DOY, take the nearest available DOY in by_doy. This
    # is deterministic given the bundled CSVs.
    available = np.array(sorted(by_doy.index))
    temps = []
    for d in doy:
        nearest = int(available[np.argmin(np.abs(available - d))])
        temps.append(float(by_doy[nearest]))
    return pd.DataFrame({"date": dates, "temperature": np.array(temps, dtype=float)})
