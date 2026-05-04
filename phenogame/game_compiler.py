"""
phenogame.game_compiler — Build the 13 × S Algorithm-vs-Nature game.

Pipeline
--------

1. ``build_nature_scenarios(data)`` discretises selected columns of ``data``
   into ``S`` Nature scenarios.
2. ``learned_payoff_surface(models, scenarios)`` evaluates the 13 fitted
   EML families on the per-scenario representative rows.
3. ``construct_13_by_s_payoff_matrix(eml_results, scenarios)`` packages the
   results into a :class:`PayoffMatrix` with ``rows = 13 model choices``,
   ``columns = S Nature scenarios``.
4. ``compile_game_from_data(data, target_col, feature_cols)`` is the
   end-to-end convenience entry point.

The output is a :class:`phenogame.game.PayoffMatrix` so it plugs into the
existing ``zero_sum_minimax`` / Hedge ε-CCE machinery without changes.

This module deliberately treats the 13 EML families as *candidate payoff
generators*, not as 13 game players.  The two-player structure (Algorithm
vs Nature) is preserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .eml_families import (
    FittedEMLFamily,
    define_13_eml_families,
    fit_all_eml_families,
    predict_eml_family,
    rank_eml_families,
)
from .game import PayoffMatrix


# ────────────────────────────────────────────────────────────────────────────
# 1. Nature scenario construction
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class NatureScenarios:
    """Set of ``S`` Nature scenarios derived from data.

    Attributes
    ----------
    labels
        Length-``S`` list of human-readable scenario labels.
    representatives
        ``(S, n_features)`` DataFrame with one representative row per
        scenario, used to evaluate each candidate model.
    feature_columns
        Names of the feature columns used.
    scenario_columns
        Names of the columns used to build the scenarios.
    n_per_scenario
        Number of rows from ``data`` falling into each scenario.
    """

    labels: List[str]
    representatives: pd.DataFrame
    feature_columns: List[str]
    scenario_columns: List[str]
    n_per_scenario: List[int] = field(default_factory=list)

    def summary(self) -> str:
        return "\n".join([
            f"NatureScenarios: S={len(self.labels)}",
            f"  scenario columns: {', '.join(self.scenario_columns) or '(single bucket)'}",
            f"  feature columns: {', '.join(self.feature_columns)}",
            f"  rows per scenario: {self.n_per_scenario}",
        ])


def _safe_qcut_labels(s: pd.Series, n_bins: int) -> List[str]:
    """Return string bin labels for a numeric series; falls back gracefully."""
    s_clean = pd.to_numeric(s, errors="coerce").dropna()
    if s_clean.empty:
        return ["all"] * len(s)
    n_unique = s_clean.nunique()
    n_bins_eff = max(1, min(n_bins, n_unique))
    try:
        cats = pd.qcut(s_clean, q=n_bins_eff, duplicates="drop")
    except ValueError:
        cats = pd.cut(s_clean, bins=n_bins_eff, duplicates="drop")
    out = pd.Series(["unknown"] * len(s), index=s.index, dtype=object)
    out.loc[s_clean.index] = cats.astype(str)
    return out.tolist()


def build_nature_scenarios(
    data: pd.DataFrame,
    *,
    scenario_cols: Optional[Sequence[str]] = None,
    feature_cols: Optional[Sequence[str]] = None,
    n_bins: int = 3,
    max_scenarios: int = 12,
) -> NatureScenarios:
    """Discretise data into a finite set of Nature scenarios.

    Parameters
    ----------
    data
        Tabular dataset.  Each row is one observation.
    scenario_cols
        Columns whose joint discretisation defines Nature.  If omitted, a
        single ``"all"`` scenario is produced.
    feature_cols
        Feature columns later used to evaluate the candidate models.
        Defaults to all numeric columns excluding ``scenario_cols``.
    n_bins
        Quantile bins per scenario column (default 3 → low/med/high).
    max_scenarios
        Cap on the total number of scenarios.  If the cross-product exceeds
        this, only the largest-population scenarios are kept.

    Returns
    -------
    NatureScenarios
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if data.shape[0] == 0:
        raise ValueError("data must have at least one row.")

    df = data.copy()

    # Default feature columns: all numeric columns excluding scenario_cols.
    if feature_cols is None:
        scen_set = set(scenario_cols or [])
        feature_cols = [c for c in df.columns
                        if pd.api.types.is_numeric_dtype(df[c]) and c not in scen_set]
    feature_cols = list(feature_cols)
    if len(feature_cols) == 0:
        raise ValueError("At least one numeric feature column is required.")

    if not scenario_cols:
        # Single all-data scenario (no scenario_cols supplied).
        rep = pd.DataFrame(
            df[feature_cols].mean(numeric_only=True).to_dict(),
            index=[0],
        )
        return NatureScenarios(
            labels=["all"],
            representatives=rep,
            feature_columns=feature_cols,
            scenario_columns=[],
            n_per_scenario=[int(df.shape[0])],
        )

    scenario_cols = list(scenario_cols)
    bin_cols = []
    for c in scenario_cols:
        if c not in df.columns:
            raise ValueError(f"scenario column not in data: {c}")
        if pd.api.types.is_numeric_dtype(df[c]):
            df[f"_{c}_bin"] = _safe_qcut_labels(df[c], n_bins)
        else:
            df[f"_{c}_bin"] = df[c].astype(str)
        bin_cols.append(f"_{c}_bin")

    df["_scenario"] = df[bin_cols].astype(str).agg(" | ".join, axis=1)
    grouped = df.groupby("_scenario", sort=True)
    counts = grouped.size().sort_values(ascending=False)
    keep = list(counts.index[:max_scenarios])
    df_keep = df[df["_scenario"].isin(keep)].copy()

    rep_rows = []
    labels: List[str] = []
    n_per_scenario: List[int] = []
    for label, sub in df_keep.groupby("_scenario", sort=True):
        rep = sub[feature_cols].mean(numeric_only=True)
        rep_rows.append(rep)
        labels.append(str(label))
        n_per_scenario.append(int(sub.shape[0]))

    rep_df = pd.DataFrame(rep_rows).reset_index(drop=True)
    rep_df.columns = feature_cols
    return NatureScenarios(
        labels=labels,
        representatives=rep_df,
        feature_columns=feature_cols,
        scenario_columns=scenario_cols,
        n_per_scenario=n_per_scenario,
    )


# ────────────────────────────────────────────────────────────────────────────
# 2. Learned payoff surface (13 candidate models × S scenarios)
# ────────────────────────────────────────────────────────────────────────────

def learned_payoff_surface(
    models: Dict[str, FittedEMLFamily],
    scenarios: NatureScenarios,
) -> pd.DataFrame:
    """Evaluate each fitted EML family on each scenario's representative row.

    Parameters
    ----------
    models
        Mapping ``family_id -> FittedEMLFamily``.
    scenarios
        NatureScenarios instance.

    Returns
    -------
    pandas.DataFrame
        ``(13, S)`` predictions; rows are family IDs in fixed canonical
        order (``F01..F13``), columns are scenario labels.  All entries are
        finite; non-finite predictions are replaced with the family's
        per-row median.
    """
    family_order = [s.family_id for s in define_13_eml_families()]
    missing = [fid for fid in family_order if fid not in models]
    if missing:
        raise ValueError(f"Missing fitted families: {missing}")

    X_eval = scenarios.representatives[scenarios.feature_columns]
    rows = []
    for fid in family_order:
        preds = predict_eml_family(models[fid], X_eval)
        # Already nan-clean from predict_eml_family, but defend further:
        finite = np.isfinite(preds)
        if not np.all(finite):
            med = np.nanmedian(preds[finite]) if finite.any() else 0.0
            preds = np.where(finite, preds, med)
        rows.append(preds.astype(float))
    arr = np.vstack(rows)
    return pd.DataFrame(arr, index=family_order, columns=scenarios.labels)


# ────────────────────────────────────────────────────────────────────────────
# 3. 13 × S payoff matrix construction
# ────────────────────────────────────────────────────────────────────────────

def construct_13_by_s_payoff_matrix(
    eml_results: Dict[str, FittedEMLFamily],
    scenarios: NatureScenarios,
    *,
    metric: str = "eml_family_payoff",
) -> PayoffMatrix:
    """Construct a :class:`PayoffMatrix` of shape ``(13, S)``.

    Rows correspond to the 13 EML model choices (Algorithm/Selector's pure
    actions); columns to the S Nature scenarios.  The numerical entries are
    the predicted payoffs from :func:`learned_payoff_surface`.
    """
    surface = learned_payoff_surface(eml_results, scenarios)
    matrix = surface.to_numpy(dtype=float)
    if matrix.shape[0] != 13:
        raise RuntimeError(
            f"Internal: expected 13 rows in payoff surface, got {matrix.shape[0]}."
        )
    # Defensive finite-check.
    if not np.all(np.isfinite(matrix)):
        raise RuntimeError("Payoff matrix contains non-finite values after surface eval.")

    return PayoffMatrix(
        matrix=matrix,
        strategy_labels=list(surface.index),
        scenario_labels=list(surface.columns),
        func_id="eml_family_choice",
        params={"n_families": 13.0, "n_scenarios": float(matrix.shape[1])},
        metric=metric,
    )


# ────────────────────────────────────────────────────────────────────────────
# 4. End-to-end convenience entry point
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class CompiledGame:
    """End-to-end output of :func:`compile_game_from_data`."""
    payoff_matrix: PayoffMatrix
    families: Dict[str, FittedEMLFamily]
    family_ranking: pd.DataFrame
    scenarios: NatureScenarios
    payoff_surface: pd.DataFrame
    target_column: str

    def summary(self) -> str:
        m, n = self.payoff_matrix.matrix.shape
        return "\n".join([
            f"CompiledGame ({m} model rows × {n} Nature columns)",
            f"  target: {self.target_column}",
            self.scenarios.summary(),
            "  top 3 families by training RMSE:",
            *(f"    {row.family_id} ({row.name}) RMSE={row.train_rmse:.4f}"
              for row in self.family_ranking.head(3).itertuples()),
        ])


def compile_game_from_data(
    data: pd.DataFrame,
    target_col: str,
    feature_cols: Sequence[str],
    *,
    scenario_cols: Optional[Sequence[str]] = None,
    n_bins: int = 3,
    max_scenarios: int = 12,
    base_seed: int = 42,
) -> CompiledGame:
    """End-to-end pipeline: data → 13 EML families → 13×S payoff matrix.

    Parameters
    ----------
    data
        Tabular dataset.
    target_col
        Name of the response variable.
    feature_cols
        Feature columns.  Cleaned for NaN/inf before fitting.
    scenario_cols
        Columns defining Nature.  If omitted, a single ``"all"`` scenario
        is used.
    n_bins, max_scenarios
        Forwarded to :func:`build_nature_scenarios`.
    base_seed
        Random seed forwarded to :func:`fit_all_eml_families`.

    Returns
    -------
    CompiledGame
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    feature_cols = list(feature_cols)
    if target_col not in data.columns:
        raise ValueError(f"target_col not in data: {target_col!r}")
    missing = [c for c in feature_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Missing feature_cols: {missing}")

    work = data[[*feature_cols, target_col, *(scenario_cols or [])]].copy()
    work = work.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[*feature_cols, target_col]
    )
    if len(work) < 3:
        raise ValueError("Need at least 3 clean rows after dropna for fitting.")

    # 1. Fit the 13 candidate families on the full clean dataset.
    families = fit_all_eml_families(
        work[feature_cols], work[target_col], base_seed=base_seed,
    )
    ranking = rank_eml_families(families)

    # 2. Build Nature scenarios.
    scenarios = build_nature_scenarios(
        work,
        scenario_cols=scenario_cols,
        feature_cols=feature_cols,
        n_bins=n_bins,
        max_scenarios=max_scenarios,
    )

    # 3. Compile 13 × S payoff matrix.
    surface = learned_payoff_surface(families, scenarios)
    pm = construct_13_by_s_payoff_matrix(
        families, scenarios, metric=f"eml-family-prediction[{target_col}]",
    )

    return CompiledGame(
        payoff_matrix=pm,
        families=families,
        family_ranking=ranking,
        scenarios=scenarios,
        payoff_surface=surface,
        target_column=target_col,
    )
