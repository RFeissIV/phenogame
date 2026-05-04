"""
phenogame.game — Farmer vs Nature: game-theoretic crop decisions.

Constructs a two-person game where:
  Player 1 (Farmer): chooses management strategies (planting date, cultivar, etc.)
  Player 2 (Nature): chooses environmental states (weather scenarios)

The payoff matrix is built from calibrated phenological response functions.
Each cell = cumulative development (or stage-reached boolean, or yield proxy)
under that (strategy, scenario) pair.

Four classical DPUU criteria from Dillon (1962):
  Wald (maximin):      max_s min_n payoff(s, n)     — pessimistic
  Laplace:             max_s mean_n payoff(s, n)     — equal probability
  Hurwicz:             max_s [α·max_n + (1-α)·min_n] — tunable optimism
  Savage (minimax regret): min_s max_n regret(s, n)  — minimize worst-case regret

Banzhaf attribution:
  When multiple environmental drivers interact multiplicatively (the standard
  crop model assumption), Banzhaf values quantify each driver's marginal
  contribution to the developmental outcome. This answers "why was my crop
  late?" with coalitional attribution grounded in cooperative game theory.

Reference:
  Dillon, J. L. (1962). Applications of game theory in agricultural economics:
  Review and requiem. Australian J. Agric. Econ., 6(2), 20-35.

  Odrzywolek, A. (2026). All elementary functions from a single binary operator.
  arXiv:2603.21852.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from .response import RESPONSE_FUNCTIONS


# ═══════════════════════════════════════════════════════════════
# PAYOFF MATRIX CONSTRUCTION
# ═══════════════════════════════════════════════════════════════

@dataclass
class PayoffMatrix:
    """A Farmer vs Nature payoff matrix."""
    matrix: np.ndarray              # shape (n_strategies, n_scenarios)
    strategy_labels: List[str]      # farmer's choices
    scenario_labels: List[str]      # nature's states
    func_id: str                    # response function used
    params: Dict[str, float]        # calibrated parameters
    metric: str                     # what the payoff measures

    def __str__(self):
        df = pd.DataFrame(
            self.matrix,
            index=self.strategy_labels,
            columns=self.scenario_labels,
        )
        return f"Payoff matrix ({self.metric}):\n{df.to_string()}"


def build_payoff_matrix(
    env_scenarios: Dict[str, pd.DataFrame],
    planting_dates: List[str],
    func_id: str,
    params: Dict[str, float],
    target_threshold: float,
    target_date: str,
    date_col: str = "date",
    driver_col: str = "temperature",
    metric: str = "gdd",
) -> PayoffMatrix:
    """Build a Farmer vs Nature payoff matrix.

    Parameters
    ----------
    env_scenarios : dict
        Mapping of scenario name -> weather DataFrame.
        Example: {"dry_cool": df1, "normal": df2, "wet_warm": df3}
    planting_dates : list of str
        Farmer's strategy set (ISO date strings).
    func_id : str
        Calibrated response function ID.
    params : dict
        Calibrated parameters.
    target_threshold : float
        Cumulative development needed for target stage.
    target_date : str
        Deadline date.
    metric : str
        "gdd" = total accumulation, "feasible" = 0/1 reached target,
        "margin" = days of slack if feasible (negative if not).

    Returns
    -------
    PayoffMatrix with shape (n_planting_dates, n_scenarios).
    """
    rf = RESPONSE_FUNCTIONS[func_id]
    n_strat = len(planting_dates)
    n_scen = len(env_scenarios)
    scen_names = list(env_scenarios.keys())
    matrix = np.zeros((n_strat, n_scen))

    target_dt = pd.to_datetime(target_date)

    for i, pdate in enumerate(planting_dates):
        pdt = pd.to_datetime(pdate)
        for j, sname in enumerate(scen_names):
            env = env_scenarios[sname].copy()
            env[date_col] = pd.to_datetime(env[date_col])

            # Window: planting date to target date
            mask = (env[date_col] >= pdt) & (env[date_col] <= target_dt)
            window = env[mask]

            if len(window) == 0:
                matrix[i, j] = 0.0
                continue

            x = window[driver_col].values.astype(float)
            daily_rate = rf.fn(x, **params)
            cumul = float(np.sum(daily_rate))

            if metric == "gdd":
                matrix[i, j] = cumul
            elif metric == "feasible":
                matrix[i, j] = 1.0 if cumul >= target_threshold else 0.0
            elif metric == "margin":
                if cumul >= target_threshold:
                    # Find how many days before target we reach threshold
                    cs = np.cumsum(daily_rate)
                    crossing = np.where(cs >= target_threshold)[0]
                    if len(crossing) > 0:
                        days_to_target = len(daily_rate) - crossing[0]
                        matrix[i, j] = float(days_to_target)
                    else:
                        matrix[i, j] = 0.0
                else:
                    matrix[i, j] = -(target_threshold - cumul)

    return PayoffMatrix(
        matrix=matrix,
        strategy_labels=[str(d) for d in planting_dates],
        scenario_labels=scen_names,
        func_id=func_id,
        params=params,
        metric=metric,
    )


# ═══════════════════════════════════════════════════════════════
# DPUU CRITERIA (Dillon 1962, pp. 23-28)
# ═══════════════════════════════════════════════════════════════

@dataclass
class DPUUResult:
    """Result of a Decision Problem Under Uncertainty analysis."""
    criterion: str
    best_strategy_index: int
    best_strategy_label: str
    criterion_value: float
    all_values: np.ndarray  # criterion value for each strategy
    explanation: str


def wald_maximin(pm: PayoffMatrix) -> DPUUResult:
    """Wald criterion: maximize the minimum payoff.

    The most conservative criterion. Farmer assumes Nature will do her worst.
    Dillon (1962) p.27: "best for farmers who must consider short-run outcomes
    because of financial commitments."
    """
    row_mins = pm.matrix.min(axis=1)
    best = int(np.argmax(row_mins))
    return DPUUResult(
        criterion="Wald (maximin)",
        best_strategy_index=best,
        best_strategy_label=pm.strategy_labels[best],
        criterion_value=float(row_mins[best]),
        all_values=row_mins,
        explanation="Maximize worst-case outcome. Most conservative.",
    )


def laplace(pm: PayoffMatrix) -> DPUUResult:
    """Laplace criterion: maximize the expected payoff under equal probabilities.

    Assumes all of Nature's states are equally likely.
    Dillon (1962) p.27: "pertinent if the farmer is financially free to follow
    choices which may lead to highest long-run profits."
    """
    row_means = pm.matrix.mean(axis=1)
    best = int(np.argmax(row_means))
    return DPUUResult(
        criterion="Laplace (equal probability)",
        best_strategy_index=best,
        best_strategy_label=pm.strategy_labels[best],
        criterion_value=float(row_means[best]),
        all_values=row_means,
        explanation="Maximize expected payoff assuming equal probability of all scenarios.",
    )


def hurwicz(pm: PayoffMatrix, alpha: float = 0.5) -> DPUUResult:
    """Hurwicz criterion: weighted combination of best and worst cases.

    alpha = 1.0 is pure optimism (maximize best case).
    alpha = 0.0 is pure pessimism (same as Wald).
    Dillon (1962) p.27: "relevant for optimistic farmers who desire and can
    afford to gamble."
    """
    row_max = pm.matrix.max(axis=1)
    row_min = pm.matrix.min(axis=1)
    h_values = alpha * row_max + (1 - alpha) * row_min
    best = int(np.argmax(h_values))
    return DPUUResult(
        criterion=f"Hurwicz (alpha={alpha:.2f})",
        best_strategy_index=best,
        best_strategy_label=pm.strategy_labels[best],
        criterion_value=float(h_values[best]),
        all_values=h_values,
        explanation=f"Weighted optimism-pessimism (alpha={alpha:.2f}). "
                    f"alpha=1 = pure optimist, alpha=0 = pure pessimist.",
    )


def savage_regret(pm: PayoffMatrix) -> DPUUResult:
    """Savage-Niehans minimax regret criterion.

    For each scenario, compute regret = (best possible payoff) - (actual payoff).
    Then choose the strategy that minimizes the maximum regret.
    Dillon (1962) p.28: "appropriate for farmers who cannot completely ignore
    short-run outcomes but can give some weight to long-run considerations."
    """
    # Best payoff in each scenario (column max)
    col_max = pm.matrix.max(axis=0)
    # Regret matrix
    regret = col_max - pm.matrix
    # Maximum regret for each strategy
    max_regret = regret.max(axis=1)
    best = int(np.argmin(max_regret))
    return DPUUResult(
        criterion="Savage (minimax regret)",
        best_strategy_index=best,
        best_strategy_label=pm.strategy_labels[best],
        criterion_value=float(max_regret[best]),
        all_values=max_regret,
        explanation="Minimize worst-case regret (opportunity cost of not choosing "
                    "the best strategy for each scenario).",
    )


def solve_all_criteria(pm: PayoffMatrix, alpha: float = 0.5) -> Dict[str, DPUUResult]:
    """Solve all four DPUU criteria and return results."""
    return {
        "wald": wald_maximin(pm),
        "laplace": laplace(pm),
        "hurwicz": hurwicz(pm, alpha),
        "savage": savage_regret(pm),
    }


# ═══════════════════════════════════════════════════════════════
# ZERO-SUM MINIMAX SOLUTION (Farmer/Algorithm vs Nature)
# ═══════════════════════════════════════════════════════════════

@dataclass
class MinimaxResult:
    """Zero-sum minimax solution of the Farmer vs Nature game.

    This is the maximin solution for a two-player zero-sum game,
    computed via linear programming. It is NOT a general Nash
    equilibrium for arbitrary multi-player games.
    """
    farmer_strategy: np.ndarray     # mixed strategy probabilities
    nature_strategy: np.ndarray     # adversarial Nature's mixed strategy
    game_value: float               # guaranteed expected payoff
    strategy_labels: List[str]
    scenario_labels: List[str]
    is_pure: bool                   # True if equilibrium is a pure strategy
    dominant_strategy: Optional[str] = None  # label if pure

    def summary(self) -> str:
        lines = [
            f"Zero-sum minimax (game value = {self.game_value:.2f}):",
        ]
        if self.is_pure:
            lines.append(f"  Pure strategy: {self.dominant_strategy}")
        else:
            lines.append("  Mixed strategy (hedge across planting dates):")
            for label, prob in zip(self.strategy_labels, self.farmer_strategy):
                if prob > 0.005:
                    lines.append(f"    {label}: {prob:.1%}")
            lines.append("  Adversarial Nature (worst-case scenario mix):")
            for label, prob in zip(self.scenario_labels, self.nature_strategy):
                if prob > 0.005:
                    lines.append(f"    {label}: {prob:.1%}")
        return "\n".join(lines)


def zero_sum_minimax(pm: PayoffMatrix) -> MinimaxResult:
    """Compute the zero-sum minimax solution of the Farmer vs Nature game.

    Solves the maximin problem via linear programming:
      max v  s.t.  A^T p >= v·1,  p >= 0,  sum(p) = 1

    This finds the farmer's mixed strategy that maximizes the guaranteed
    expected payoff against the worst-case adversarial Nature.

    For a farmer, this answers: "If I split my acreage across planting
    windows, what allocation guarantees the best expected outcome
    regardless of what weather happens?"

    Dillon (1962) p.22: zero-sum games against Nature admit minimax
    solutions via linear programming.
    """
    from scipy.optimize import linprog

    A = pm.matrix  # (n_strategies, n_scenarios)
    m, n = A.shape

    # Farmer's maximin: max v s.t. A^T p >= v, p >= 0, sum(p) = 1
    # Rewrite as LP: min -v
    #   variables: [p_1, ..., p_m, v]
    #   constraints: A^T p - v >= 0  =>  -A^T p + v <= 0
    #   sum(p) = 1
    #   p >= 0, v unbounded (use large lower bound)

    # Objective: minimize -v (last variable)
    c = np.zeros(m + 1)
    c[-1] = -1.0

    # Inequality constraints: -A^T p + v <= 0  (for each scenario j)
    # A_ub @ x <= b_ub
    A_ub = np.zeros((n, m + 1))
    A_ub[:, :m] = -A.T  # -A^T p
    A_ub[:, -1] = 1.0   # + v
    b_ub = np.zeros(n)

    # Equality constraint: sum(p) = 1
    A_eq = np.zeros((1, m + 1))
    A_eq[0, :m] = 1.0
    b_eq = np.array([1.0])

    # Bounds: p_i >= 0, v unbounded below
    bounds = [(0, None)] * m + [(None, None)]

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method="highs")

    if not result.success:
        raise RuntimeError(f"Zero-sum minimax LP failed: {result.message}")

    farmer_p = result.x[:m]
    game_value = result.x[-1]

    # Clean tiny probabilities
    farmer_p[farmer_p < 1e-8] = 0.0
    farmer_p /= farmer_p.sum()

    # Nature's strategy: solve the dual (min w s.t. A q <= w, q >= 0, sum(q)=1)
    c_n = np.zeros(n + 1)
    c_n[-1] = 1.0  # minimize w

    A_ub_n = np.zeros((m, n + 1))
    A_ub_n[:, :n] = A       # A q
    A_ub_n[:, -1] = -1.0    # - w <= 0
    b_ub_n = np.zeros(m)

    A_eq_n = np.zeros((1, n + 1))
    A_eq_n[0, :n] = 1.0
    b_eq_n = np.array([1.0])

    bounds_n = [(0, None)] * n + [(None, None)]

    result_n = linprog(c_n, A_ub=A_ub_n, b_ub=b_ub_n, A_eq=A_eq_n, b_eq=b_eq_n,
                       bounds=bounds_n, method="highs")

    if result_n.success:
        nature_q = result_n.x[:n]
        nature_q[nature_q < 1e-8] = 0.0
        nature_q /= nature_q.sum()
    else:
        nature_q = np.ones(n) / n  # fallback to uniform

    # Check if pure strategy
    is_pure = np.max(farmer_p) > 0.99
    dominant = pm.strategy_labels[np.argmax(farmer_p)] if is_pure else None

    return MinimaxResult(
        farmer_strategy=farmer_p,
        nature_strategy=nature_q,
        game_value=float(game_value),
        strategy_labels=pm.strategy_labels,
        scenario_labels=pm.scenario_labels,
        is_pure=is_pure,
        dominant_strategy=dominant,
    )


# ═══════════════════════════════════════════════════════════════
# BANZHAF DRIVER ATTRIBUTION
# ═══════════════════════════════════════════════════════════════

def banzhaf_attribution(
    env: pd.DataFrame,
    driver_cols: List[str],
    func_ids: Dict[str, str],
    params_dict: Dict[str, Dict[str, float]],
    baseline_values: Dict[str, float],
    date_col: str = "date",
    aggregation: str = "product",
) -> pd.DataFrame:
    """Compute Banzhaf power indices for environmental drivers.

    Treats the multi-driver phenology system as a cooperative game where
    each driver is a player and the characteristic function v(S) is the
    cumulative development rate when only drivers in coalition S are active
    (others set to their baseline/neutral values).

    For n drivers, computes all 2^n coalitions (feasible for n <= 8).

    Parameters
    ----------
    env : DataFrame
        Environmental time series with columns for each driver.
    driver_cols : list of str
        Column names for each environmental driver (e.g., ["temperature",
        "photoperiod", "soil_moisture", "leaf_N"]).
    func_ids : dict
        Mapping of driver_col -> response function ID.
    params_dict : dict
        Mapping of driver_col -> calibrated parameters.
    baseline_values : dict
        Mapping of driver_col -> neutral/baseline value (e.g., optimal temp).
        When a driver is NOT in the coalition, its column is set to this value.
    aggregation : str
        How driver responses combine: "product" (multiplicative, standard crop model)
        or "sum" (additive).

    Returns
    -------
    DataFrame with columns: driver, banzhaf_value, banzhaf_normalized,
    shapley_value, marginal_contributions.
    """
    n = len(driver_cols)
    if n > 8:
        raise ValueError(f"Exact Banzhaf requires n <= 8 drivers, got {n}")

    n_coalitions = 2 ** n

    # Compute characteristic function for each coalition
    def v(coalition_mask: List[bool]) -> float:
        """Evaluate the coalition: active drivers use real data, inactive use baseline."""
        env_c = env.copy()
        for k, col in enumerate(driver_cols):
            if not coalition_mask[k]:
                env_c[col] = baseline_values[col]

        # Compute each driver's response scalar
        scalars = []
        for col in driver_cols:
            if col in func_ids:
                rf = RESPONSE_FUNCTIONS[func_ids[col]]
                x = env_c[col].values.astype(float)
                response = rf.fn(x, **params_dict[col])
                scalar = float(np.mean(response))
                scalars.append(scalar)

        if aggregation == "product":
            return float(np.prod(scalars)) if scalars else 0.0
        else:
            return float(np.sum(scalars)) if scalars else 0.0

    # Compute Banzhaf values
    banzhaf = np.zeros(n)

    for i in range(n):
        marginals = []
        # Iterate over all coalitions NOT containing i
        for bits in range(n_coalitions):
            if bits & (1 << i):
                continue  # skip coalitions containing i
            # Coalition without i
            mask_without = [(bits >> k) & 1 == 1 for k in range(n)]
            # Coalition with i
            mask_with = mask_without.copy()
            mask_with[i] = True

            v_with = v(mask_with)
            v_without = v(mask_without)
            marginals.append(v_with - v_without)

        banzhaf[i] = float(np.mean(marginals))

    # Normalize
    total = np.sum(np.abs(banzhaf))
    banzhaf_norm = banzhaf / total if total > 0 else banzhaf

    # Shapley values (exact for small n)
    from math import factorial
    shapley = np.zeros(n)
    for i in range(n):
        for bits in range(n_coalitions):
            if bits & (1 << i):
                continue
            mask_without = [(bits >> k) & 1 == 1 for k in range(n)]
            mask_with = mask_without.copy()
            mask_with[i] = True
            s = sum(mask_without)
            weight = factorial(s) * factorial(n - s - 1) / factorial(n)
            v_with = v(mask_with)
            v_without = v(mask_without)
            shapley[i] += weight * (v_with - v_without)

    shapley_norm = shapley / np.sum(np.abs(shapley)) if np.sum(np.abs(shapley)) > 0 else shapley

    return pd.DataFrame({
        "driver": driver_cols,
        "banzhaf_value": banzhaf,
        "banzhaf_normalized": banzhaf_norm,
        "shapley_value": shapley,
        "shapley_normalized": shapley_norm,
    })


# ═══════════════════════════════════════════════════════════════
# GAME SUMMARY
# ═══════════════════════════════════════════════════════════════

@dataclass
class GameReport:
    """Complete Farmer vs Nature analysis."""
    payoff: PayoffMatrix
    criteria: Dict[str, DPUUResult]
    nash: Optional[MinimaxResult] = None  # backward-compatible name for minimax result
    attribution: Optional[pd.DataFrame] = None

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "FARMER vs NATURE — Game-Theoretic Decision Report",
            "=" * 60,
            "",
            str(self.payoff),
            "",
            "Decision criteria (Dillon 1962):",
        ]
        for name, result in self.criteria.items():
            lines.append(
                f"  {result.criterion:30s} -> {result.best_strategy_label}"
                f"  (value={result.criterion_value:.2f})"
            )

        if self.nash is not None:
            lines.append("")
            lines.append(self.nash.summary())

        lines.append("")

        # Check consensus across DPUU criteria
        recommendations = set(r.best_strategy_label for r in self.criteria.values())
        if len(recommendations) == 1:
            rec = recommendations.pop()
            lines.append(f"CONSENSUS: All DPUU criteria recommend {rec}")
            if self.nash and self.nash.is_pure and self.nash.dominant_strategy == rec:
                lines.append(f"  Zero-sum minimax confirms: {rec} is dominant.")
            elif self.nash and not self.nash.is_pure:
                lines.append("  Zero-sum minimax suggests hedging (mixed strategy) for adversarial robustness.")
        else:
            lines.append("DIVERGENCE: Criteria disagree. Recommendation depends on risk attitude.")
            for name, result in self.criteria.items():
                lines.append(f"  {result.criterion}: {result.best_strategy_label}")

        if self.attribution is not None:
            lines.append("")
            lines.append("Driver attribution (Banzhaf):")
            for _, row in self.attribution.iterrows():
                pct = abs(row["banzhaf_normalized"]) * 100
                lines.append(f"  {row['driver']:20s} {pct:5.1f}%")

        return "\n".join(lines)

# Backward-compatible aliases. The "Nash" name is retained for callers that
# already use it, but new code should prefer the explicit zero-sum names
# below — we only solve the two-player zero-sum minimax problem here, not a
# general Nash equilibrium.
NashResult = MinimaxResult
nash_equilibrium = zero_sum_minimax

# Explicit, non-overclaiming aliases (preferred):
zero_sum_minimax_equilibrium = zero_sum_minimax
zero_sum_minimax_solution = zero_sum_minimax
