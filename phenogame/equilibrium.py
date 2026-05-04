"""
phenogame.equilibrium — No-regret dynamics and ε-CCE certification.

Given a payoff matrix (from game.py), runs no-regret learning dynamics
and certifies whether the empirical joint distribution is an approximate
coarse correlated equilibrium (ε-CCE).

Two modes:

  Mode 1 — Provable (Hedge/multiplicative weights):
    Uses the standard Hedge algorithm with learning rate η_t = sqrt(2 ln|A|/t).
    Formal no-regret bound: R_T ≤ O(sqrt(ln|A|/T)).
    Output: formally bounded ε-CCE certificate.

  Mode 2 — Empirical (EML-derived transform):
    Uses ψ_EML(R) = max(exp(R/τ) - ln(1 + exp(-R/τ)), 0).
    This transform is elementary (EML-reducible) and empirically validated
    but does NOT yet have a formal no-regret proof.
    Output: empirical ε-CCE certificate (not formally proven).

The distinction matters. Do not conflate the two.

Theorem (Hedge → CCE):
  If both players use Hedge with η_t = sqrt(2 ln|A_i|/t), then after T rounds:
    max_i max_{a'} (1/T) Σ_t [u_i(a', a_{-i,t}) - u_i(a_{i,t}, a_{-i,t})] ≤ ε_T
  where ε_T = max_i sqrt(2 ln|A_i| / T).
  Therefore μ_T is an ε_T-CCE.

Reference:
  Freund, Y. & Schapire, R. (1999). Adaptive game playing using
  multiplicative weights. Games and Economic Behavior, 29, 79-103.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from .game import PayoffMatrix


@dataclass
class CCECertificate:
    """Equilibrium certificate from no-regret dynamics.

    Note on ``epsilon`` semantics
    -----------------------------
    The meaning of ``epsilon`` differs between modes:

    - In ``mode="provable"`` (Hedge dynamics), ``epsilon`` is the **formal
      Freund-Schapire upper bound** ``√(2 ln|A| / T)`` — a *guaranteed*
      regret bound from the theorem.
    - In ``mode="empirical"`` (EML-derived dynamics), ``epsilon`` is the
      **measured** ``cce_gap`` itself. The EML transform has no formal
      no-regret proof (audit finding F9), so no theoretical bound is
      reported; the field is reused to surface what was actually achieved.

    This deliberate dual use is documented here and in
    :func:`phenogame.regret_panel.compare_regret_transforms`, which uses
    a separate ``epsilon_target`` parameter to avoid the ambiguity when
    comparing transforms head-to-head.
    """
    # Core outputs
    empirical_joint: np.ndarray       # μ_T: shape (m, n)
    cce_gap: float                    # max deviation from CCE constraints
    epsilon: float                    # provable: formal bound; empirical: measured gap
    is_certified: bool                # True if gap ≤ ε (provable) or gap ≤ 0.05 (empirical)
    mode: str                         # "provable" or "empirical"

    # Dynamics metadata
    iterations: int
    player1_regrets: np.ndarray       # per-action external regret, player 1
    player2_regrets: np.ndarray       # per-action external regret, player 2
    max_regret_p1: float
    max_regret_p2: float

    # Payoff context
    strategy_labels: List[str]
    scenario_labels: List[str]
    game_value_nash: Optional[float] = None

    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"EQUILIBRIUM CERTIFICATE ({self.mode.upper()} MODE)",
            "=" * 60,
            "",
            f"Iterations: {self.iterations}",
            f"CCE gap: {self.cce_gap:.6f}",
            f"ε bound: {self.epsilon:.6f}",
            f"Certified ε-CCE: {'YES' if self.is_certified else 'NO'}",
            "",
            f"Player 1 max external regret: {self.max_regret_p1:.6f}",
            f"Player 2 max external regret: {self.max_regret_p2:.6f}",
            "",
        ]
        if self.mode == "provable":
            lines.append(
                "This certificate is backed by the Hedge no-regret theorem:"
            )
            lines.append(
                f"  ε_T = max_i sqrt(2 ln|A_i| / T) = {self.epsilon:.6f}"
            )
            lines.append(
                "  (Freund & Schapire 1999, Theorem 1)"
            )
        else:
            lines.append(
                "This certificate is EMPIRICAL. The EML-derived regret transform"
            )
            lines.append(
                "has not been formally proven to satisfy no-regret bounds."
            )
            lines.append(
                "The gap is measured, not guaranteed."
            )

        lines.append("")
        lines.append("Empirical joint distribution μ_T:")
        import pandas as pd
        df = pd.DataFrame(
            self.empirical_joint,
            index=self.strategy_labels,
            columns=self.scenario_labels,
        )
        lines.append(df.to_string())
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# CCE GAP COMPUTATION
# ═══════════════════════════════════════════════════════════════

def compute_cce_gap(mu: np.ndarray, U1: np.ndarray, U2: np.ndarray) -> float:
    """Compute the coarse correlated equilibrium gap.

    For each player i and each deviation action a'_i:
      gap_i(a') = Σ_{a} μ(a) [u_i(a'_i, a_{-i}) - u_i(a)]

    CCE gap = max over all players and deviations of gap_i(a').

    If gap ≤ 0, μ is an exact CCE.
    If gap ≤ ε, μ is an ε-CCE.
    """
    m, n = mu.shape
    gaps = []

    # Player 1 deviations: for each alternative action a1'
    for a1_prime in range(m):
        # Expected payoff under deviation: Σ_{a1,a2} μ(a1,a2) u1(a1', a2)
        dev_payoff = np.sum(mu * U1[a1_prime, :][np.newaxis, :])
        # Expected payoff under μ: Σ_{a1,a2} μ(a1,a2) u1(a1, a2)
        current_payoff = np.sum(mu * U1)
        gaps.append(dev_payoff - current_payoff)

    # Player 2 deviations: for each alternative action a2'
    for a2_prime in range(n):
        dev_payoff = np.sum(mu * U2[:, a2_prime][:, np.newaxis])
        current_payoff = np.sum(mu * U2)
        gaps.append(dev_payoff - current_payoff)

    return float(max(gaps))


# ═══════════════════════════════════════════════════════════════
# MODE 1: PROVABLE — Hedge / Multiplicative Weights
# ═══════════════════════════════════════════════════════════════

def _hedge_update(weights: np.ndarray, payoffs: np.ndarray,
                  eta: float) -> np.ndarray:
    """Standard Hedge/multiplicative weights update.

    w_{a}(t+1) = w_{a}(t) · exp(η · u(a, opponent_action))
    Then normalize to probability simplex.
    """
    log_weights = np.log(weights + 1e-30) + eta * payoffs
    log_weights -= log_weights.max()  # numerical stability
    new_weights = np.exp(log_weights)
    return new_weights / new_weights.sum()


def certify_provable(
    pm: PayoffMatrix,
    iterations: int = 10000,
    seed: Optional[int] = 0,
) -> CCECertificate:
    """Run Hedge dynamics and produce a formally certified ε-CCE.

    Uses the standard Hedge algorithm (multiplicative weights) with
    decreasing learning rate η_t = sqrt(2 ln|A| / t).

    The formal bound guarantees:
      ε_T = max_i sqrt(2 ln|A_i| / T)

    This is a PROVEN bound (Freund & Schapire 1999).

    Reproducibility
    ---------------
    ``seed`` defaults to ``0`` so calls without an explicit seed are still
    bit-reproducible. Pass ``seed=None`` to draw from system entropy
    (non-deterministic) for genuine Monte-Carlo runs.
    """
    rng = np.random.default_rng(seed)

    A = pm.matrix  # Player 1's payoff (Farmer)
    m, n = A.shape
    U1 = A           # Farmer payoff
    U2 = -A          # Nature payoff (zero-sum)

    # Initialize uniform
    w1 = np.ones(m) / m
    w2 = np.ones(n) / n

    # Track empirical joint distribution
    joint_counts = np.zeros((m, n))

    # Track cumulative regret for each action
    cum_payoff_1 = np.zeros(m)
    cum_payoff_2 = np.zeros(n)
    cum_realized_1 = 0.0
    cum_realized_2 = 0.0

    for t in range(1, iterations + 1):
        eta = np.sqrt(2.0 * np.log(max(m, n)) / t)

        # Sample actions from current mixed strategies
        a1 = rng.choice(m, p=w1)
        a2 = rng.choice(n, p=w2)

        # Record joint play
        joint_counts[a1, a2] += 1

        # Payoffs for all actions against opponent's choice
        payoffs_1 = U1[:, a2]    # player 1's payoff for each action vs a2
        payoffs_2 = U2[a1, :]    # player 2's payoff for each action vs a1

        # Track cumulative regret
        cum_payoff_1 += payoffs_1
        cum_payoff_2 += payoffs_2
        cum_realized_1 += U1[a1, a2]
        cum_realized_2 += U2[a1, a2]

        # Hedge update
        w1 = _hedge_update(w1, payoffs_1, eta)
        w2 = _hedge_update(w2, payoffs_2, eta)

    # Empirical joint distribution
    mu_T = joint_counts / iterations

    # External regret per action
    regrets_1 = (cum_payoff_1 - cum_realized_1) / iterations
    regrets_2 = (cum_payoff_2 - cum_realized_2) / iterations

    # Formal ε bound
    epsilon_formal = max(
        np.sqrt(2.0 * np.log(m) / iterations),
        np.sqrt(2.0 * np.log(n) / iterations),
    )

    # Actual CCE gap
    gap = compute_cce_gap(mu_T, U1, U2)

    return CCECertificate(
        empirical_joint=mu_T,
        cce_gap=gap,
        epsilon=epsilon_formal,
        is_certified=(gap <= epsilon_formal + 1e-10),
        mode="provable",
        iterations=iterations,
        player1_regrets=regrets_1,
        player2_regrets=regrets_2,
        max_regret_p1=float(np.max(regrets_1)),
        max_regret_p2=float(np.max(regrets_2)),
        strategy_labels=pm.strategy_labels,
        scenario_labels=pm.scenario_labels,
    )


# ═══════════════════════════════════════════════════════════════
# MODE 2: EMPIRICAL — EML-derived regret transform
# ═══════════════════════════════════════════════════════════════

def _eml_regret_transform(R: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """EML-derived regret transform.

    ψ_EML(R) = max(exp(R/τ) - ln(1 + exp(-R/τ)), 0)

    This is an elementary function (EML-reducible) that maps regret
    to action weights. It combines exponential amplification of positive
    regret with logarithmic dampening.

    IMPORTANT: This transform is empirically validated but does NOT have
    a formal no-regret proof. Use certify_provable() for formal guarantees.
    """
    R_scaled = np.clip(R / tau, -50, 50)
    psi = np.exp(R_scaled) - np.log(1.0 + np.exp(-R_scaled))
    return np.maximum(psi, 1e-12)


def certify_empirical(
    pm: PayoffMatrix,
    iterations: int = 10000,
    tau: float = 1.0,
    seed: Optional[int] = 0,
) -> CCECertificate:
    """Run EML-derived regret dynamics and measure (not guarantee) ε-CCE.

    Uses the EML regret transform ψ_EML(R) to generate action weights
    from cumulative regret. The empirical joint distribution is tested
    against CCE constraints.

    WARNING: This mode produces an EMPIRICAL certificate only.
    The EML transform has not been formally proven to satisfy no-regret bounds.
    The CCE gap is measured from the simulation, not guaranteed by theory.
    For formal guarantees, use certify_provable().

    Reproducibility
    ---------------
    ``seed`` defaults to ``0`` so calls without an explicit seed are still
    bit-reproducible. Pass ``seed=None`` for non-deterministic runs.
    """
    rng = np.random.default_rng(seed)

    A = pm.matrix
    m, n = A.shape
    U1 = A
    U2 = -A

    # Cumulative regret per action
    cum_regret_1 = np.zeros(m)
    cum_regret_2 = np.zeros(n)

    joint_counts = np.zeros((m, n))
    cum_realized_1 = 0.0
    cum_realized_2 = 0.0
    cum_payoff_1 = np.zeros(m)
    cum_payoff_2 = np.zeros(n)

    for t in range(1, iterations + 1):
        # Generate weights from cumulative regret via EML transform
        w1_raw = _eml_regret_transform(cum_regret_1, tau)
        w1 = w1_raw / w1_raw.sum()

        w2_raw = _eml_regret_transform(cum_regret_2, tau)
        w2 = w2_raw / w2_raw.sum()

        # Sample actions
        a1 = rng.choice(m, p=w1)
        a2 = rng.choice(n, p=w2)

        joint_counts[a1, a2] += 1

        # Update cumulative regret
        # External regret for action a: u(a, opponent) - u(played, opponent)
        for a in range(m):
            cum_regret_1[a] += U1[a, a2] - U1[a1, a2]
        for a in range(n):
            cum_regret_2[a] += U2[a1, a] - U2[a1, a2]

        cum_payoff_1 += U1[:, a2]
        cum_payoff_2 += U2[a1, :]
        cum_realized_1 += U1[a1, a2]
        cum_realized_2 += U2[a1, a2]

    mu_T = joint_counts / iterations
    regrets_1 = (cum_payoff_1 - cum_realized_1) / iterations
    regrets_2 = (cum_payoff_2 - cum_realized_2) / iterations

    gap = compute_cce_gap(mu_T, U1, U2)

    # Empirical ε: measured, not guaranteed
    epsilon_empirical = max(float(np.max(regrets_1)), float(np.max(regrets_2)))

    return CCECertificate(
        empirical_joint=mu_T,
        cce_gap=gap,
        epsilon=max(gap, 0.0),  # report actual gap as ε (honest)
        is_certified=(gap <= 0.05),  # practical threshold
        mode="empirical",
        iterations=iterations,
        player1_regrets=regrets_1,
        player2_regrets=regrets_2,
        max_regret_p1=float(np.max(regrets_1)),
        max_regret_p2=float(np.max(regrets_2)),
        strategy_labels=pm.strategy_labels,
        scenario_labels=pm.scenario_labels,
    )


# ═══════════════════════════════════════════════════════════════
# Naming clarity: this package solves the zero-sum minimax problem
# (Algorithm/Selector vs Nature). The names below make that
# explicit. They are exact aliases of the existing implementations.
# ═══════════════════════════════════════════════════════════════

# Re-export under explicit, non-overclaiming names so downstream code
# can write `zero_sum_minimax_equilibrium(pm)` / `zero_sum_minimax_solution(pm)`
# instead of the ambiguous "nash_equilibrium". The underlying object is
# identical; only the name signals scope (zero-sum two-player only).
from .game import zero_sum_minimax as zero_sum_minimax_equilibrium  # noqa: E402
from .game import zero_sum_minimax as zero_sum_minimax_solution     # noqa: E402
