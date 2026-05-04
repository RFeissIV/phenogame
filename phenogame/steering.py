"""
phenogame.steering — Crop Steering ε-CCE Certificate.

Given tabular grow data, constructs a decision game between the grower
(management strategies) and nature (environmental scenarios), fits
response models to observed outcomes, computes a transparent payoff
matrix from a user-declared formula, and certifies the recommended
strategy as part of an approximate coarse correlated equilibrium.

The certificate says exactly this and nothing more:

    "Given the uploaded data, fitted response model, declared payoff
     formula, and scenario definitions, this strategy is part of a
     certified ε-CCE of the induced decision game."

It does NOT say "this proves the best grow recipe."

Pipeline:
  1. Ingest tabular data (CSV/DataFrame)
  2. Auto-detect or accept user-defined action columns and scenarios
  3. Fit response model to outcome column
  4. Construct payoff matrix via transparent weighted formula
  5. Run DPUU criteria + zero-sum minimax solution
  6. Run no-regret dynamics → ε-CCE certificate
  7. Return structured report with full provenance

Reference:
  Dillon, J. L. (1962). Applications of game theory in agricultural
  economics: Review and requiem. Aust. J. Agric. Econ., 6(2), 20-35.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

from .game import (
    PayoffMatrix, solve_all_criteria, nash_equilibrium,
    DPUUResult, NashResult, GameReport,
)
from .equilibrium import (
    certify_provable, certify_empirical, CCECertificate,
)
from .fit import fit_all, fit_single, FitReport


# ═══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════

@dataclass
class PayoffFormula:
    """Transparent, user-visible payoff formula.

    U = α·outcome + β·quality - γ·cost - δ·risk

    All weights and column mappings are recorded for full provenance.
    The user sees exactly this formula in the certificate.
    """
    outcome_col: str
    outcome_weight: float = 1.0
    quality_col: Optional[str] = None
    quality_weight: float = 0.0
    cost_col: Optional[str] = None
    cost_weight: float = 0.0
    risk_col: Optional[str] = None
    risk_weight: float = 0.0

    def compute(self, df: pd.DataFrame) -> np.ndarray:
        """Evaluate the payoff formula on a DataFrame.

        Returns a 1D array of payoff values, one per row.
        """
        u = self.outcome_weight * df[self.outcome_col].values.astype(float)
        if self.quality_col and self.quality_col in df.columns:
            u = u + self.quality_weight * df[self.quality_col].values.astype(float)
        if self.cost_col and self.cost_col in df.columns:
            u = u - self.cost_weight * df[self.cost_col].values.astype(float)
        if self.risk_col and self.risk_col in df.columns:
            u = u - self.risk_weight * df[self.risk_col].values.astype(float)
        return u

    def __str__(self) -> str:
        parts = [f"{self.outcome_weight:+.2f}·{self.outcome_col}"]
        if self.quality_col:
            parts.append(f"{self.quality_weight:+.2f}·{self.quality_col}")
        if self.cost_col:
            parts.append(f"{-self.cost_weight:+.2f}·{self.cost_col}")
        if self.risk_col:
            parts.append(f"{-self.risk_weight:+.2f}·{self.risk_col}")
        return "U = " + " ".join(parts)


@dataclass
class DataSummary:
    """Summary statistics of the uploaded dataset."""
    n_rows: int
    n_cols: int
    columns: List[str]
    action_columns: List[str]
    scenario_column: str
    n_strategies: int
    n_scenarios: int
    strategy_labels: List[str]
    scenario_labels: List[str]
    outcome_stats: Dict[str, float]


@dataclass
class SteeringCertificate:
    """Complete Crop Steering ε-CCE Certificate.

    Contains the full provenance chain from data to recommendation:
      1. Data summary
      2. Fitted response model (if applicable)
      3. Strategy and scenario definitions
      4. Payoff formula (transparent)
      5. Payoff matrix
      6. DPUU criteria results
      7. Zero-sum minimax solution
      8. ε-CCE certificate
      9. Recommended strategy with reasoning
     10. Limitations and caveats
    """
    # Provenance
    data_summary: DataSummary
    payoff_formula: PayoffFormula
    payoff_matrix: PayoffMatrix
    fit_report: Optional[FitReport]

    # Decision analysis
    criteria: Dict[str, DPUUResult]
    nash: NashResult
    certificate: CCECertificate

    # Recommendation
    recommended_strategy: str
    recommendation_reason: str
    estimated_payoff: float
    consensus: bool  # True if all DPUU criteria agree

    def summary(self) -> str:
        """Full human-readable certificate."""
        lines = [
            "=" * 64,
            "  CROP STEERING ε-CCE CERTIFICATE",
            "=" * 64,
            "",

            "1. DATA SUMMARY",
            f"   Rows: {self.data_summary.n_rows}  "
            f"Columns: {self.data_summary.n_cols}",
            f"   Strategies ({self.data_summary.n_strategies}): "
            f"{', '.join(self.data_summary.strategy_labels)}",
            f"   Scenarios ({self.data_summary.n_scenarios}):  "
            f"{', '.join(self.data_summary.scenario_labels)}",
            "",

            "2. PAYOFF FORMULA",
            f"   {self.payoff_formula}",
            "",

            "3. PAYOFF MATRIX",
            f"   {self.payoff_matrix}",
            "",

            "4. DECISION CRITERIA (Dillon 1962)",
        ]
        for key, result in self.criteria.items():
            lines.append(
                f"   {result.criterion:30s} → {result.best_strategy_label}"
                f"  (value={result.criterion_value:.4f})"
            )

        lines.extend([
            "",
            "5. ZERO-SUM MINIMAX SOLUTION",
            f"   {self.nash.summary()}",
            "",
            "6. ε-CCE CERTIFICATE",
            f"   Mode:      {self.certificate.mode}",
            f"   Iterations: {self.certificate.iterations}",
            f"   CCE gap:   {self.certificate.cce_gap:.6f}",
            f"   ε bound:   {self.certificate.epsilon:.6f}",
            f"   Certified: {'YES' if self.certificate.is_certified else 'NO'}",
            "",
            "7. RECOMMENDATION",
            f"   Strategy: {self.recommended_strategy}",
            f"   Estimated payoff: {self.estimated_payoff:.4f}",
            f"   Consensus: {'all criteria agree' if self.consensus else 'criteria disagree'}",
            f"   Reason: {self.recommendation_reason}",
            "",
            "8. LIMITATIONS",
            "   • The certificate is conditional on the uploaded data, the",
            "     declared payoff formula, and the scenario definitions.",
            "   • It does NOT prove this is the globally optimal strategy.",
            "   • The response model is fitted to observed data and may not",
            "     generalize to conditions outside the data range.",
        ])
        if self.certificate.mode == "empirical":
            lines.extend([
                "   • EMPIRICAL mode: the EML-derived regret transform has",
                "     not been formally proven to satisfy no-regret bounds.",
                "     The gap is measured, not guaranteed.",
            ])
        lines.extend([
            "   • Banzhaf/Shapley attribution is available separately via",
            "     the .attribute() method for driver-level explanation.",
            "",
            "CLAIM: Given the uploaded data, fitted response model, declared",
            "payoff formula, and scenario definitions, the recommended strategy",
            "is part of a certified ε-CCE of the induced decision game.",
            "=" * 64,
        ])
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# STRATEGY / SCENARIO CONSTRUCTION
# ═══════════════════════════════════════════════════════════════

def _build_strategy_label(row: pd.Series, action_cols: List[str]) -> str:
    """Create a human-readable strategy label from action column values."""
    parts = []
    for col in action_cols:
        val = row[col]
        if isinstance(val, float):
            parts.append(f"{col}={val:.2g}")
        else:
            parts.append(f"{col}={val}")
    return "_".join(parts)


def _discretize_column(series: pd.Series, n_bins: int = 3) -> pd.Series:
    """Discretize a continuous column into labeled bins."""
    try:
        binned = pd.qcut(series, q=n_bins, labels=False, duplicates="drop")
        labels = {0: "low", 1: "mid", 2: "high"}
        if n_bins == 2:
            labels = {0: "low", 1: "high"}
        return binned.map(lambda x: labels.get(int(x), f"bin{x}") if pd.notna(x) else "na")
    except (ValueError, TypeError):
        return series.astype(str)


def build_payoff_matrix_from_data(
    data: pd.DataFrame,
    action_columns: List[str],
    scenario_column: str,
    formula: PayoffFormula,
    discretize_actions: bool = True,
    discretize_scenarios: bool = True,
    n_action_bins: int = 3,
    n_scenario_bins: int = 3,
) -> Tuple[PayoffMatrix, DataSummary]:
    """Construct a payoff matrix from tabular grow data.

    Parameters
    ----------
    data : DataFrame
        Raw observation data with action, scenario, and outcome columns.
    action_columns : list of str
        Columns representing grower choices (irrigation, nitrogen, PPFD, etc.).
        If multiple, they are combined into composite strategy labels.
    scenario_column : str
        Column representing nature's state (temperature, VPD, week, etc.).
    formula : PayoffFormula
        The transparent payoff formula.
    discretize_actions : bool
        If True and action columns are continuous, discretize into bins.
    discretize_scenarios : bool
        If True and scenario column is continuous, discretize into bins.
    n_action_bins : int
        Number of bins for continuous action discretization.
    n_scenario_bins : int
        Number of bins for continuous scenario discretization.

    Returns
    -------
    (PayoffMatrix, DataSummary)
    """
    df = data.copy()

    # Discretize if needed
    for col in action_columns:
        if discretize_actions and pd.api.types.is_numeric_dtype(df[col]):
            df[f"_act_{col}"] = _discretize_column(df[col], n_action_bins)
        else:
            df[f"_act_{col}"] = df[col].astype(str)

    if discretize_scenarios and pd.api.types.is_numeric_dtype(df[scenario_column]):
        df["_scenario"] = _discretize_column(df[scenario_column], n_scenario_bins)
    else:
        df["_scenario"] = df[scenario_column].astype(str)

    # Build composite strategy labels
    act_cols = [f"_act_{c}" for c in action_columns]
    df["_strategy"] = df[act_cols].apply(
        lambda row: "_".join(str(v) for v in row), axis=1
    )

    # Compute payoff values
    df["_payoff"] = formula.compute(df)

    # Aggregate: mean payoff per (strategy, scenario) cell
    pivot = df.pivot_table(
        values="_payoff",
        index="_strategy",
        columns="_scenario",
        aggfunc="mean",
    )

    # Drop rows/cols with all NaN (impossible combinations)
    pivot = pivot.dropna(how="all", axis=0).dropna(how="all", axis=1)

    # Fill remaining NaN with row minimum (pessimistic imputation for
    # unobserved combinations — honest, conservative)
    for idx in pivot.index:
        row_min = pivot.loc[idx].min()
        pivot.loc[idx] = pivot.loc[idx].fillna(
            row_min if pd.notna(row_min) else 0.0
        )

    strategy_labels = list(pivot.index)
    scenario_labels = list(pivot.columns)
    matrix = pivot.values.astype(float)

    pm = PayoffMatrix(
        matrix=matrix,
        strategy_labels=strategy_labels,
        scenario_labels=scenario_labels,
        func_id="composite_payoff",
        params={},
        metric=str(formula),
    )

    summary = DataSummary(
        n_rows=len(data),
        n_cols=len(data.columns),
        columns=list(data.columns),
        action_columns=action_columns,
        scenario_column=scenario_column,
        n_strategies=len(strategy_labels),
        n_scenarios=len(scenario_labels),
        strategy_labels=strategy_labels,
        scenario_labels=scenario_labels,
        outcome_stats={
            "mean": float(data[formula.outcome_col].mean()),
            "std": float(data[formula.outcome_col].std()),
            "min": float(data[formula.outcome_col].min()),
            "max": float(data[formula.outcome_col].max()),
        },
    )

    return pm, summary


# ═══════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def crop_steering_certificate(
    data: Union[pd.DataFrame, str],
    action_columns: List[str],
    scenario_column: str,
    outcome_column: str,
    quality_column: Optional[str] = None,
    cost_column: Optional[str] = None,
    risk_column: Optional[str] = None,
    payoff_weights: Optional[Dict[str, float]] = None,
    mode: str = "provable",
    iterations: int = 10000,
    tau: float = 1.0,
    alpha: float = 0.5,
    seed: Optional[int] = 0,
    discretize_actions: bool = True,
    discretize_scenarios: bool = True,
    n_action_bins: int = 3,
    n_scenario_bins: int = 3,
    fit_driver_col: Optional[str] = None,
) -> SteeringCertificate:
    """Produce a Crop Steering ε-CCE Certificate from tabular data.

    This is the main entry point. Upload your data, declare your payoff
    formula, and receive a certified strategy recommendation.

    Parameters
    ----------
    data : DataFrame or str (path to CSV)
        Observation data. Minimum required columns: at least one action
        column, one scenario column, and one outcome column.
    action_columns : list of str
        Columns representing grower choices (e.g., ["irrigation_ml",
        "nitrogen_ec", "ppfd"]).
    scenario_column : str
        Column representing nature's state (e.g., "temperature",
        "vpd", "week").
    outcome_column : str
        Column with observed outcome (e.g., "yield_g", "growth_rate").
    quality_column : str, optional
        Column for quality score (added to payoff).
    cost_column : str, optional
        Column for cost (subtracted from payoff).
    risk_column : str, optional
        Column for risk/stress metric (subtracted from payoff).
    payoff_weights : dict, optional
        Override default weights. Keys: "outcome", "quality", "cost", "risk".
        Default: {"outcome": 1.0, "quality": 0.5, "cost": 0.3, "risk": 0.5}
    mode : str
        "provable" — Hedge with formal no-regret bound (recommended).
        "empirical" — EML-derived transform (measured, not guaranteed).
    iterations : int
        Number of rounds of no-regret dynamics.
    tau : float
        Temperature for EML transform (empirical mode only).
    alpha : float
        Hurwicz optimism coefficient (0=pessimist, 1=optimist).
    seed : int, optional
        Random seed for reproducibility.
    discretize_actions : bool
        Discretize continuous action columns into bins.
    discretize_scenarios : bool
        Discretize continuous scenario column into bins.
    n_action_bins : int
        Number of bins for action discretization.
    n_scenario_bins : int
        Number of bins for scenario discretization.
    fit_driver_col : str, optional
        If provided, also fit a response function to this driver vs.
        the outcome column (for EML metadata and model diagnostics).

    Returns
    -------
    SteeringCertificate with full provenance and recommendation.

    Example
    -------
    >>> cert = crop_steering_certificate(
    ...     data=grow_log,
    ...     action_columns=["ppfd", "irrigation_ml", "nitrogen_ec"],
    ...     scenario_column="temperature",
    ...     outcome_column="yield_g",
    ...     quality_column="quality_score",
    ...     cost_column="cost",
    ...     risk_column="stress",
    ...     payoff_weights={"outcome": 1.0, "quality": 0.5, "cost": 0.3, "risk": 0.5},
    ...     mode="provable",
    ... )
    >>> print(cert.summary())
    """
    # Load data if path
    if isinstance(data, str):
        data = pd.read_csv(data)

    # Validate columns
    required = [scenario_column, outcome_column] + action_columns
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Available: {list(data.columns)}")

    # Build payoff formula
    w = payoff_weights or {}
    formula = PayoffFormula(
        outcome_col=outcome_column,
        outcome_weight=w.get("outcome", 1.0),
        quality_col=quality_column,
        quality_weight=w.get("quality", 0.5) if quality_column else 0.0,
        cost_col=cost_column,
        cost_weight=w.get("cost", 0.3) if cost_column else 0.0,
        risk_col=risk_column,
        risk_weight=w.get("risk", 0.5) if risk_column else 0.0,
    )

    # Build payoff matrix
    pm, data_summary = build_payoff_matrix_from_data(
        data, action_columns, scenario_column, formula,
        discretize_actions=discretize_actions,
        discretize_scenarios=discretize_scenarios,
        n_action_bins=n_action_bins,
        n_scenario_bins=n_scenario_bins,
    )

    # Validate matrix is non-degenerate
    if pm.matrix.shape[0] < 2:
        raise ValueError(
            f"Need at least 2 strategies, got {pm.matrix.shape[0]}. "
            "Try increasing n_action_bins or providing more varied action data."
        )
    if pm.matrix.shape[1] < 2:
        raise ValueError(
            f"Need at least 2 scenarios, got {pm.matrix.shape[1]}. "
            "Try increasing n_scenario_bins or providing more varied scenario data."
        )

    # Optional response function fit
    fit_report = None
    if fit_driver_col and fit_driver_col in data.columns:
        x = data[fit_driver_col].values.astype(float)
        y = data[outcome_column].values.astype(float)
        # Normalize y to [0, 1] for response function fitting
        y_min, y_max = y.min(), y.max()
        if y_max > y_min:
            y_norm = (y - y_min) / (y_max - y_min)
        else:
            y_norm = np.zeros_like(y)
        try:
            fit_report = fit_all(x, y_norm)
        except RuntimeError:
            fit_report = None  # fitting failed, proceed without

    # DPUU criteria
    criteria = solve_all_criteria(pm, alpha)

    # Zero-sum minimax solution; stored in the legacy `nash` field for backward compatibility.
    nash = nash_equilibrium(pm)

    # CCE certification
    if mode == "provable":
        certificate = certify_provable(pm, iterations, seed)
    elif mode == "empirical":
        certificate = certify_empirical(pm, iterations, tau, seed)
    else:
        raise ValueError(f"mode must be 'provable' or 'empirical', got '{mode}'")

    # Determine recommendation
    # Priority: if all DPUU criteria agree, use that. Otherwise, use
    # Wald (most conservative) as the default recommendation.
    recommendations = {r.best_strategy_label for r in criteria.values()}
    consensus = len(recommendations) == 1

    if consensus:
        rec = recommendations.pop()
        rec_idx = pm.strategy_labels.index(rec)
        reason = "all four DPUU criteria agree on this strategy"
    else:
        # Default to Wald (conservative) for safety
        wald = criteria["wald"]
        rec = wald.best_strategy_label
        rec_idx = wald.best_strategy_index
        reason = (
            "DPUU criteria disagree; defaulting to Wald (maximin) for "
            "conservative safety. See full criteria for risk-tuned alternatives."
        )

    # Add Nash context to reason
    if nash.is_pure and nash.dominant_strategy == rec:
        reason += f". Zero-sum minimax confirms: {rec} is dominant."
    elif not nash.is_pure:
        reason += ". Zero-sum minimax suggests hedging (mixed strategy) for adversarial robustness."

    # Add certification context
    if certificate.is_certified:
        reason += f" Strategy is part of a certified ε-CCE (ε={certificate.epsilon:.4f})."
    else:
        reason += f" CCE certification did not pass (gap={certificate.cce_gap:.4f})."

    # Estimated payoff: Laplace (expected under equal probability)
    estimated_payoff = float(pm.matrix[rec_idx].mean())

    return SteeringCertificate(
        data_summary=data_summary,
        payoff_formula=formula,
        payoff_matrix=pm,
        fit_report=fit_report,
        criteria=criteria,
        nash=nash,
        certificate=certificate,
        recommended_strategy=rec,
        recommendation_reason=reason,
        estimated_payoff=estimated_payoff,
        consensus=consensus,
    )
