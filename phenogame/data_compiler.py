"""Data-induced game compiler using learned EML-tree payoff surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .eml_tree import EMLPayoffFit, fit_eml_payoff_model
from .game import PayoffMatrix
from .steering import _discretize_column


@dataclass
class DataInducedGame:
    """Finite game compiled from tabular data and an EML-tree payoff model."""

    payoff_matrix: PayoffMatrix
    payoff_fit: EMLPayoffFit
    strategy_column: str
    scenario_column: str
    feature_columns: List[str]
    imputation: str

    def summary(self) -> str:
        return "\n".join([
            "Data-induced EML payoff game",
            f"  strategy column: {self.strategy_column}",
            f"  scenario column: {self.scenario_column}",
            f"  feature columns: {', '.join(self.feature_columns)}",
            f"  matrix shape: {self.payoff_matrix.matrix.shape}",
            f"  imputation: {self.imputation}",
            "",
            self.payoff_fit.summary(),
        ])


def compile_eml_payoff_game(
    data: pd.DataFrame,
    *,
    strategy_column: str,
    scenario_column: str,
    feature_columns: List[str],
    target_column: str,
    discretize_strategy: bool = True,
    discretize_scenario: bool = True,
    n_strategy_bins: int = 3,
    n_scenario_bins: int = 3,
    n_eml_features: int = 64,
    alpha: float = 1.0,
    seed: int = 42,
    imputation: str = "row_min",
) -> DataInducedGame:
    """Compile a finite payoff game from data using an EML-tree model.

    The learned model predicts the payoff/response target for each observation.
    Predicted values are then averaged by strategy × scenario to create the
    finite game consumed by PhenoGame's minimax/CCE routines.
    """
    required = [strategy_column, scenario_column, target_column, *feature_columns]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if imputation not in {"row_min", "zero", "drop"}:
        raise ValueError("imputation must be one of: 'row_min', 'zero', 'drop'.")

    fit = fit_eml_payoff_model(
        data, feature_columns, target_column,
        n_features=n_eml_features, alpha=alpha, seed=seed,
    )
    df = data.dropna(subset=feature_columns).copy()
    df["_eml_payoff"] = fit.model.predict(df[feature_columns])

    if discretize_strategy and pd.api.types.is_numeric_dtype(df[strategy_column]):
        df["_strategy"] = _discretize_column(df[strategy_column], n_strategy_bins)
    else:
        df["_strategy"] = df[strategy_column].astype(str)

    if discretize_scenario and pd.api.types.is_numeric_dtype(df[scenario_column]):
        df["_scenario"] = _discretize_column(df[scenario_column], n_scenario_bins)
    else:
        df["_scenario"] = df[scenario_column].astype(str)

    pivot = df.pivot_table(
        values="_eml_payoff", index="_strategy", columns="_scenario", aggfunc="mean"
    ).dropna(how="all", axis=0).dropna(how="all", axis=1)

    if imputation == "drop":
        pivot = pivot.dropna(axis=0).dropna(axis=1)
    elif imputation == "zero":
        pivot = pivot.fillna(0.0)
    else:
        for idx in pivot.index:
            row_min = pivot.loc[idx].min()
            pivot.loc[idx] = pivot.loc[idx].fillna(float(row_min) if pd.notna(row_min) else 0.0)

    if pivot.empty:
        raise ValueError("Compiled payoff matrix is empty after aggregation/imputation.")

    pm = PayoffMatrix(
        matrix=pivot.to_numpy(dtype=float),
        strategy_labels=[str(x) for x in pivot.index],
        scenario_labels=[str(x) for x in pivot.columns],
        func_id="eml_tree_payoff",
        params={"n_eml_features": float(n_eml_features), "alpha": float(alpha)},
        metric=f"EML-tree prediction of {target_column}",
    )
    return DataInducedGame(
        payoff_matrix=pm,
        payoff_fit=fit,
        strategy_column=strategy_column,
        scenario_column=scenario_column,
        feature_columns=list(feature_columns),
        imputation=imputation,
    )
