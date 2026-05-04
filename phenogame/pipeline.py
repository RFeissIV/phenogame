"""
phenogame.pipeline — Data loading and crop game pipeline.

Provides canonical functions for loading USA-NPN phenometrics
and running decision games on phenological data.

    npn = load_npn_csv("npn_grape.csv")
    result = run_phenology_game(npn, phenophase="Ripe fruits")

Design rules:
  - USA-NPN -9999 sentinel values are filtered explicitly.
  - Scenarios are derived from observed variability, not invented.
  - Mode is Hedge no-regret (Freund & Schapire 1999).
  - Claim: data-induced epsilon-CCE certificate, not a new proof.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

from .game import PayoffMatrix
from .equilibrium import CCECertificate
from .steering import (
    crop_steering_certificate, SteeringCertificate,
    PayoffFormula,
)


_DATA_DIR = Path(__file__).resolve().parent / "data"


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_npn_csv(
    filename: str = "npn_grape.csv",
    data_dir: Optional[Union[str, Path]] = None,
    phenophase: Optional[str] = None,
) -> pd.DataFrame:
    """Load USA-NPN site phenometrics CSV with explicit -9999 filtering.

    The USA-NPN uses -9999 as a sentinel for missing data. This function
    removes all rows where Mean_First_Yes_DOY or Mean_AGDD equals -9999.

    Parameters
    ----------
    filename : str
        CSV filename in the bundled data directory.
    data_dir : str or Path, optional
        Override the default phenogame/data/ directory.
    phenophase : str, optional
        Filter to a specific phenophase (e.g., "Ripe fruits").

    Returns
    -------
    DataFrame with -9999 sentinel values removed.

    Example
    -------
    >>> npn = load_npn_csv("npn_grape.csv")
    >>> ripe = load_npn_csv("npn_grape.csv", phenophase="Ripe fruits")
    """
    directory = Path(data_dir) if data_dir else _DATA_DIR
    path = directory / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            "Ensure phenogame/data/ contains the bundled CSV."
        )

    raw = pd.read_csv(path)

    # Explicit -9999 sentinel filtering
    df = raw[
        (raw["Mean_First_Yes_DOY"] != -9999) &
        (raw["Mean_First_Yes_DOY"] > 0) &
        (raw["Mean_AGDD"] != -9999) &
        (raw["Mean_AGDD"] > 0)
    ].copy()

    if phenophase:
        df = df[df["Phenophase_Description"] == phenophase].copy()

    return df


# ═══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════

@dataclass
class PhenologyGameResult:
    """Structured result of a phenology-driven decision game.

    Every field traces to either observed data or a transparent formula.

    Example
    -------
    >>> result = run_phenology_game(npn, phenophase="Ripe fruits")
    >>> result.recommended_strategy   # "standard"
    >>> result.cce_gap                # 0.008
    >>> result.certified              # True
    >>> result.doy_mean               # 198.0
    >>> result.doy_std                # 28.0
    """
    # Provenance
    phenophase: str
    n_obs: int
    doy_mean: float
    doy_std: float
    agdd_mean: float

    # Game structure
    strategies: Dict[str, int]    # name → target DOY
    scenarios: Dict[str, int]     # name → actual DOY
    payoff_formula: str           # human-readable formula

    # Results
    payoff_matrix: PayoffMatrix
    cce_gap: float
    epsilon: float
    certified: bool
    mode: str
    recommended_strategy: str
    consensus: bool               # True if all DPUU criteria agree
    nash_is_pure: bool  # backward-compatible alias for minimax_is_pure
    nash_strategy: Optional[str]  # backward-compatible alias for minimax_strategy

    # Full certificate for deep inspection
    certificate: SteeringCertificate

    def __repr__(self) -> str:
        return (
            f"PhenologyGameResult(phenophase='{self.phenophase}', "
            f"rec='{self.recommended_strategy}', "
            f"cce_gap={self.cce_gap:.4f}, "
            f"certified={self.certified}, "
            f"consensus={self.consensus})"
        )

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Phenophase: {self.phenophase} (n={self.n_obs})",
            f"Observed: DOY {self.doy_mean:.0f} +/- {self.doy_std:.0f}, "
            f"AGDD {self.agdd_mean:.0f}",
            "",
            "Strategies (harvest target DOY):",
        ]
        for name, doy in self.strategies.items():
            lines.append(f"  {name}: DOY {doy}")
        lines.append("")
        lines.append("Scenarios (actual DOY from observed variability):")
        for name, doy in self.scenarios.items():
            lines.append(f"  {name}: DOY {doy}")
        lines.extend([
            "",
            f"Payoff: {self.payoff_formula}",
            f"Recommended: {self.recommended_strategy}",
            f"DPUU consensus: {'YES' if self.consensus else 'NO'}",
            f"Zero-sum minimax pure: {'YES — ' + (self.nash_strategy or '') if self.nash_is_pure else 'NO (mixed)'}",
            f"CCE gap: {self.cce_gap:.6f}, epsilon: {self.epsilon:.6f}",
            f"Certified: {'YES' if self.certified else 'NO'}",
            f"Mode: {'Hedge no-regret' if self.mode == 'provable' else 'EML empirical'}"
            f" ({self.mode})",
        ])
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# GAME PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_phenology_game(
    npn_data: pd.DataFrame,
    phenophase: str = "Ripe fruits",
    strategy_offsets: Optional[Dict[str, float]] = None,
    scenario_offsets: Optional[Dict[str, float]] = None,
    cost_early: float = 1.0,
    cost_late: float = 3.0,
    mode: str = "provable",
    iterations: int = 10000,
    seed: Optional[int] = 0,
) -> PhenologyGameResult:
    """Run a management-timing decision game from NPN phenophase data.

    Uses asymmetric loss: being late costs more than being early.
    This reflects real grower economics — a missed frost event or
    overripe harvest is worse than a premature action.

    Uses MAD (median absolute deviation) instead of SD for robust
    variability estimation.

    Payoff formula (asymmetric):
      loss = cost_early * max(0, actual - target)   [acted too early]
           + cost_late  * max(0, target - actual)    [acted too late]
      U = 1 - loss / max_loss

    Parameters
    ----------
    npn_data : DataFrame
        USA-NPN data (already filtered by load_npn_csv).
    phenophase : str
        Phenophase to analyze.
    strategy_offsets : dict, optional
        Offsets from median DOY in units of MAD.
        Default: {"early": -1.0, "standard": 0, "late": +1.0}
    scenario_offsets : dict, optional
        Offsets from median DOY in units of MAD.
        Default: {"cool": +1.5, "normal": 0, "warm": -1.5}
    cost_early : float
        Per-day cost of acting too early (default 1.0).
    cost_late : float
        Per-day cost of acting too late (default 3.0).
    mode : str
        "provable" (Hedge) or "empirical" (EML-derived).
    iterations : int
        Rounds of no-regret dynamics.
    seed : int, optional
        Random seed.

    Returns
    -------
    PhenologyGameResult with full provenance.
    """
    # Validate
    required_cols = ["Phenophase_Description", "Mean_First_Yes_DOY", "Mean_AGDD"]
    missing = [c for c in required_cols if c not in npn_data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    ph = npn_data[npn_data["Phenophase_Description"] == phenophase].copy()
    if len(ph) < 2:
        raise ValueError(f"Need >=2 obs for '{phenophase}', got {len(ph)}.")

    if len(ph) < 10:
        import warnings
        warnings.warn(
            f"Only {len(ph)} observations for '{phenophase}'. "
            f"Confidence intervals are wide; interpret with caution.",
            UserWarning, stacklevel=2,
        )

    doy_values = ph["Mean_First_Yes_DOY"].values.astype(float)
    doy_median = float(np.median(doy_values))
    doy_mad = float(np.median(np.abs(doy_values - doy_median)))
    doy_mean = float(np.mean(doy_values))
    doy_std = float(np.std(doy_values, ddof=1)) if len(doy_values) > 1 else 0.0
    agdd_mean = float(ph["Mean_AGDD"].mean())

    # Fallback to SD if MAD is zero
    if doy_mad == 0:
        doy_mad = doy_std
    if doy_mad == 0 or np.isnan(doy_mad):
        raise ValueError(f"Zero variability for '{phenophase}' DOY.")

    # Strategies from median ± MAD
    s_off = strategy_offsets or {"early": -1.0, "standard": 0.0, "late": 1.0}
    strategies = {
        name: int(doy_median + offset * doy_mad)
        for name, offset in s_off.items()
    }

    # Scenarios from median ± MAD
    sc_off = scenario_offsets or {"cool": 1.5, "normal": 0.0, "warm": -1.5}
    scenarios = {
        name: int(doy_median + offset * doy_mad)
        for name, offset in sc_off.items()
    }

    # ASYMMETRIC LOSS: being late costs more than being early
    rows = []
    for strat_name, target_doy in strategies.items():
        for scen_name, actual_doy in scenarios.items():
            diff = actual_doy - target_doy
            if diff >= 0:
                # Event after action → acted too early, moderate cost
                raw_loss = cost_early * diff
            else:
                # Event before action → acted too late, high cost
                raw_loss = cost_late * abs(diff)
            rows.append({
                "strategy": strat_name,
                "scenario": scen_name,
                "raw_loss": raw_loss,
            })

    payoff_df = pd.DataFrame(rows)
    max_loss = payoff_df["raw_loss"].max()
    if max_loss == 0:
        raise ValueError("All strategies identical — no game to play.")

    payoff_df["payoff"] = 1.0 - payoff_df["raw_loss"] / max_loss
    formula_str = (
        f"U = 1 - loss/max_loss; "
        f"loss = {cost_early}*max(0,actual-target) [early] "
        f"+ {cost_late}*max(0,target-actual) [late]"
    )

    cert = crop_steering_certificate(
        data=payoff_df,
        action_columns=["strategy"],
        scenario_column="scenario",
        outcome_column="payoff",
        mode=mode, iterations=iterations, seed=seed,
        discretize_actions=False, discretize_scenarios=False,
    )

    return PhenologyGameResult(
        phenophase=phenophase, n_obs=len(ph),
        doy_mean=doy_mean, doy_std=doy_std, agdd_mean=agdd_mean,
        strategies=strategies, scenarios=scenarios,
        payoff_formula=formula_str,
        payoff_matrix=cert.payoff_matrix,
        cce_gap=cert.certificate.cce_gap,
        epsilon=cert.certificate.epsilon,
        certified=cert.certificate.is_certified,
        mode=cert.certificate.mode,
        recommended_strategy=cert.recommended_strategy,
        consensus=cert.consensus,
        nash_is_pure=cert.nash.is_pure,
        nash_strategy=cert.nash.dominant_strategy if cert.nash.is_pure else None,
        certificate=cert,
    )


# ═══════════════════════════════════════════════════════════════
# GENERIC CROP GAME PIPELINE
# ═══════════════════════════════════════════════════════════════
# These functions work with any tabular data, not just NPN.
# Use them when your data has action columns, a scenario column,
# and an outcome column (e.g., trial data, management experiments).
#
# Architecture:
#
#     for crop in ["corn", "soybean", "rice"]:
#         result = run_game_for_crop(crop_data[crop])
#
#     # or:
#     results = run_multi_crop({"corn": df1, "soy": df2})
#
# Each result is a CropGameResult:
#
#     CropGameResult(
#         crop="corn",
#         payoff_matrix=...,
#         cce_gap=0.0017,
#     )

@dataclass
class CropGameResult:
    """Structured result of a single-crop decision game.

    Contains full provenance from raw data to recommendation.
    Works with any tabular data — not tied to a specific data source.

    Example
    -------
    >>> result = run_game_for_crop(
    ...     data=trial_data,
    ...     crop="corn",
    ...     action_col="treatment",
    ...     scenario_col="year_type",
    ...     outcome_col="yield_normalized",
    ... )
    >>> result.crop                  # "corn"
    >>> result.recommended_strategy  # "high_N"
    >>> result.cce_gap               # 0.0023
    >>> result.certified             # True
    """
    crop: str
    n_obs: int
    strategies: List[str]
    scenarios: List[str]
    payoff_matrix: PayoffMatrix
    cce_gap: float
    epsilon: float
    certified: bool
    mode: str
    recommended_strategy: str
    consensus: bool
    certificate: SteeringCertificate

    def __repr__(self) -> str:
        return (
            f"CropGameResult(crop='{self.crop}', "
            f"rec='{self.recommended_strategy}', "
            f"cce_gap={self.cce_gap:.4f}, "
            f"certified={self.certified})"
        )

    def summary(self) -> str:
        """One-line summary for cross-crop tables."""
        return (
            f"{self.crop:12s}  rec={self.recommended_strategy:12s}  "
            f"gap={self.cce_gap:.4f}  "
            f"cert={'YES' if self.certified else 'NO':3s}  "
            f"consensus={'YES' if self.consensus else 'NO'}"
        )


def run_game_for_crop(
    data: pd.DataFrame,
    crop: str = "unknown",
    action_col: str = "action",
    scenario_col: str = "scenario",
    outcome_col: str = "outcome",
    mode: str = "provable",
    iterations: int = 10000,
    seed: Optional[int] = 0,
    discretize_actions: bool = False,
    discretize_scenarios: bool = False,
    **kwargs,
) -> CropGameResult:
    """Run the full decision game pipeline for a single crop.

    This is the canonical single-crop entry point for any tabular data.
    It wraps crop_steering_certificate() and returns a structured
    CropGameResult.

    Parameters
    ----------
    data : DataFrame
        Must contain action_col, scenario_col, and outcome_col.
    crop : str
        Label for this crop (for display and provenance).
    action_col : str
        Column with strategy/action labels.
    scenario_col : str
        Column with scenario labels.
    outcome_col : str
        Column with outcome values (payoff).
    mode : str
        "provable" (Hedge) or "empirical" (EML-derived).
    iterations : int
        Rounds of no-regret dynamics.
    seed : int, optional
        Random seed.
    discretize_actions : bool
        Discretize continuous action columns.
    discretize_scenarios : bool
        Discretize continuous scenario column.

    Returns
    -------
    CropGameResult with full provenance.

    Example
    -------
    >>> result = run_game_for_crop(df, crop="corn",
    ...     action_col="state", scenario_col="year_type",
    ...     outcome_col="normalized_yield", seed=42)
    >>> print(result)
    CropGameResult(crop='corn', rec='Iowa', cce_gap=0.0012, certified=True)
    """
    # Input validation
    required = [action_col, scenario_col, outcome_col]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}. Available: {list(data.columns)}"
        )

    if not pd.api.types.is_numeric_dtype(data[outcome_col]):
        raise ValueError(
            f"Outcome column '{outcome_col}' must be numeric, "
            f"got {data[outcome_col].dtype}"
        )

    if data[outcome_col].isna().any():
        import warnings
        n_na = data[outcome_col].isna().sum()
        warnings.warn(
            f"{n_na} NaN values in '{outcome_col}' will be dropped.",
            UserWarning, stacklevel=2,
        )
        data = data.dropna(subset=[outcome_col])

    cert = crop_steering_certificate(
        data=data,
        action_columns=[action_col],
        scenario_column=scenario_col,
        outcome_column=outcome_col,
        mode=mode,
        iterations=iterations,
        seed=seed,
        discretize_actions=discretize_actions,
        discretize_scenarios=discretize_scenarios,
        **kwargs,
    )

    return CropGameResult(
        crop=crop,
        n_obs=len(data),
        strategies=cert.payoff_matrix.strategy_labels,
        scenarios=cert.payoff_matrix.scenario_labels,
        payoff_matrix=cert.payoff_matrix,
        cce_gap=cert.certificate.cce_gap,
        epsilon=cert.certificate.epsilon,
        certified=cert.certificate.is_certified,
        mode=cert.certificate.mode,
        recommended_strategy=cert.recommended_strategy,
        consensus=cert.consensus,
        certificate=cert,
    )


def run_multi_crop(
    datasets: Dict[str, pd.DataFrame],
    action_col: str = "action",
    scenario_col: str = "scenario",
    outcome_col: str = "outcome",
    mode: str = "provable",
    iterations: int = 10000,
    seed: Optional[int] = 0,
    **kwargs,
) -> Dict[str, CropGameResult]:
    """Run decision games for multiple crops independently.

    Each crop is analyzed independently — no cross-crop mixing.

    Parameters
    ----------
    datasets : dict of {str: DataFrame}
        Keys are crop labels, values are DataFrames.
    action_col, scenario_col, outcome_col : str
        Column names (same for all crops).
    mode : str
        "provable" or "empirical".
    iterations : int
        Rounds of no-regret dynamics.
    seed : int, optional
        Same seed used for each crop for reproducibility.

    Returns
    -------
    Dict of {crop_label: CropGameResult}.

    Example
    -------
    >>> results = run_multi_crop(
    ...     {"corn": corn_df, "soy": soy_df},
    ...     action_col="treatment",
    ...     scenario_col="year_type",
    ...     outcome_col="yield_norm",
    ... )
    >>> for crop, r in results.items():
    ...     print(r.summary())
    """
    results = {}
    for label, data in datasets.items():
        results[label] = run_game_for_crop(
            data=data,
            crop=label,
            action_col=action_col,
            scenario_col=scenario_col,
            outcome_col=outcome_col,
            mode=mode,
            iterations=iterations,
            seed=seed,
            **kwargs,
        )
    return results
