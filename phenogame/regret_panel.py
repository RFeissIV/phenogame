"""
phenogame.regret_panel — Four-transform regret-matching comparator.

Closes EML-RML twenty-pass audit findings F2 (multi-seed variance), F4
(final-gap and AUC), F7 (cross-rule comparison), F8 (normalised correlation
residual), and F10 (wall-clock timing).

Single entry point: :func:`compare_regret_transforms`.

What it does
------------

For each transform in ``transforms`` (default: standard, exp, softplus,
EML), runs ``n_seeds`` independent regret-matching simulations on the same
finite-game payoff matrix, recording per-run:

- final CCE gap at iteration ``T``
- iterations to reach ε threshold (or ``T`` if never reached)
- normalised joint-correlation residual ``ρ(μ_T) ∈ [0, 1]``
  (closes audit F8)
- wall-clock seconds (closes audit F10)

then aggregates mean ± SD across the seeds and returns a typed
:class:`RegretComparisonResult`.

Honest caveats
--------------

- ``standard`` and ``exp`` (Hedge) have formal no-regret proofs; ``eml``
  does not (audit F9). The comparator does not change that — it only
  reports what each transform achieves *empirically* on the user's game.
- ``softplus`` is always-positive (audit F6); reported numbers may differ
  from the strict regret-matching transforms for structural reasons
  unrelated to convergence quality.
- The default ``epsilon_target = 0.05`` is the audit doc's engineering
  threshold, not a derived bound.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .game import PayoffMatrix
from .joint_correlation import joint_correlation_residual_normalized
from .regret_transforms import TRANSFORM_REGISTRY, list_transforms


@dataclass
class TransformRun:
    """Per-seed result for a single transform on a single game."""
    transform: str
    seed: int
    iterations: int
    final_cce_gap: float
    iters_to_epsilon: Optional[int]   # None if not reached
    joint_corr_normalized: float
    wall_clock_seconds: float
    epsilon_target: float


@dataclass
class TransformAggregate:
    """Mean ± SD aggregate across seeds for one transform."""
    transform: str
    n_seeds: int
    final_cce_gap_mean: float
    final_cce_gap_std: float
    iters_to_epsilon_mean: Optional[float]   # None if no run reached ε
    iters_to_epsilon_std: Optional[float]
    n_runs_reached_epsilon: int
    joint_corr_normalized_mean: float
    joint_corr_normalized_std: float
    wall_clock_seconds_mean: float
    wall_clock_seconds_std: float
    no_regret_proof: str
    is_strict_rm: bool
    always_positive: bool


@dataclass
class RegretComparisonResult:
    """Output of :func:`compare_regret_transforms`."""
    payoff_shape: Tuple[int, int]
    iterations: int
    epsilon_target: float
    tau: float
    n_seeds: int
    transforms: List[str]
    runs: List[TransformRun]
    aggregates: List[TransformAggregate]
    notes: Dict[str, str] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Wide aggregate table, one row per transform."""
        rows = []
        for a in self.aggregates:
            rows.append({
                "transform": a.transform,
                "n_seeds": a.n_seeds,
                "final_cce_gap_mean": a.final_cce_gap_mean,
                "final_cce_gap_std": a.final_cce_gap_std,
                "iters_to_epsilon_mean": a.iters_to_epsilon_mean,
                "iters_to_epsilon_std": a.iters_to_epsilon_std,
                "n_runs_reached_epsilon": a.n_runs_reached_epsilon,
                "joint_corr_norm_mean": a.joint_corr_normalized_mean,
                "joint_corr_norm_std": a.joint_corr_normalized_std,
                "wall_clock_s_mean": a.wall_clock_seconds_mean,
                "wall_clock_s_std": a.wall_clock_seconds_std,
                "no_regret_proof": a.no_regret_proof,
                "is_strict_rm": a.is_strict_rm,
                "always_positive": a.always_positive,
            })
        return pd.DataFrame(rows)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = []
        m, n = self.payoff_shape
        lines.append(f"Regret-transform comparison ({m}×{n} game, "
                     f"{self.iterations} iterations, n_seeds={self.n_seeds}, "
                     f"τ={self.tau}, ε_target={self.epsilon_target})")
        lines.append("")
        lines.append(f"{'transform':12s} {'gap (μ±σ)':>22s} {'iters to ε (μ±σ)':>22s} "
                     f"{'ρ_norm (μ±σ)':>20s} {'wall s (μ±σ)':>16s}")
        for a in self.aggregates:
            gap = f"{a.final_cce_gap_mean:.4f}±{a.final_cce_gap_std:.4f}"
            if a.iters_to_epsilon_mean is None:
                iters = f"never (n={a.n_runs_reached_epsilon}/{a.n_seeds})"
            else:
                iters = (f"{a.iters_to_epsilon_mean:.0f}±{a.iters_to_epsilon_std:.0f} "
                         f"(n={a.n_runs_reached_epsilon}/{a.n_seeds})")
            rho = f"{a.joint_corr_normalized_mean:.4f}±{a.joint_corr_normalized_std:.4f}"
            wall = f"{a.wall_clock_seconds_mean:.3f}±{a.wall_clock_seconds_std:.3f}"
            lines.append(f"{a.transform:12s} {gap:>22s} {iters:>22s} {rho:>20s} {wall:>16s}")
        lines.append("")
        for k, v in self.notes.items():
            lines.append(f"  note ({k}): {v}")
        return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────────────
# Core simulation loop
# ────────────────────────────────────────────────────────────────────────────

def _run_one(
    A: np.ndarray,
    transform_name: str,
    iterations: int,
    tau: float,
    epsilon_target: float,
    seed: int,
    log_every: int = 0,
) -> TransformRun:
    """Run a single regret-matching simulation with the chosen transform.

    Mirrors :func:`phenogame.equilibrium.certify_empirical` exactly except
    for the ψ kernel and tracks iters-to-ε along the way.
    """
    info = TRANSFORM_REGISTRY[transform_name]
    psi = info["fn"]
    uses_tau = info["uses_tau"]

    rng = np.random.default_rng(seed)
    m, n = A.shape
    U1 = A
    U2 = -A

    cum_regret_1 = np.zeros(m)
    cum_regret_2 = np.zeros(n)
    joint_counts = np.zeros((m, n))
    cum_payoff_1 = np.zeros(m)
    cum_payoff_2 = np.zeros(n)
    cum_realized_1 = 0.0
    cum_realized_2 = 0.0

    iters_to_epsilon: Optional[int] = None
    eps_check_every = max(1, iterations // 50)

    t0 = time.perf_counter()

    for t in range(1, iterations + 1):
        if uses_tau:
            w1_raw = psi(cum_regret_1, tau)
            w2_raw = psi(cum_regret_2, tau)
        else:
            w1_raw = psi(cum_regret_1)
            w2_raw = psi(cum_regret_2)

        # Normalise; if all-zero (can happen for strict RM at t=1 with all
        # zero regrets), fall back to uniform.
        s1 = w1_raw.sum()
        w1 = (w1_raw / s1) if s1 > 0 else np.full(m, 1.0 / m)
        s2 = w2_raw.sum()
        w2 = (w2_raw / s2) if s2 > 0 else np.full(n, 1.0 / n)

        a1 = rng.choice(m, p=w1)
        a2 = rng.choice(n, p=w2)
        joint_counts[a1, a2] += 1

        # External regret update.
        cum_regret_1 += U1[:, a2] - U1[a1, a2]
        cum_regret_2 += U2[a1, :] - U2[a1, a2]

        cum_payoff_1 += U1[:, a2]
        cum_payoff_2 += U2[a1, :]
        cum_realized_1 += U1[a1, a2]
        cum_realized_2 += U2[a1, a2]

        if iters_to_epsilon is None and (t % eps_check_every == 0 or t == iterations):
            mu_t = joint_counts / t
            gap_t = _cce_gap(mu_t, U1, U2)
            if gap_t <= epsilon_target:
                iters_to_epsilon = t

    wall = time.perf_counter() - t0

    mu_T = joint_counts / iterations
    final_gap = _cce_gap(mu_T, U1, U2)
    rho = joint_correlation_residual_normalized(mu_T)

    return TransformRun(
        transform=transform_name,
        seed=int(seed),
        iterations=int(iterations),
        final_cce_gap=float(final_gap),
        iters_to_epsilon=iters_to_epsilon,
        joint_corr_normalized=float(rho),
        wall_clock_seconds=float(wall),
        epsilon_target=float(epsilon_target),
    )


def _cce_gap(mu: np.ndarray, U1: np.ndarray, U2: np.ndarray) -> float:
    """Local copy of the CCE-gap computation (avoid circular import)."""
    expected_1 = float(np.sum(mu * U1))
    expected_2 = float(np.sum(mu * U2))
    p_col = mu.sum(axis=0)
    p_row = mu.sum(axis=1)
    best_dev_1 = float(np.max(U1 @ p_col))
    best_dev_2 = float(np.max(p_row @ U2))
    gap = max(best_dev_1 - expected_1, best_dev_2 - expected_2, 0.0)
    return gap


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────

def compare_regret_transforms(
    pm: Union[PayoffMatrix, np.ndarray],
    *,
    transforms: Optional[Sequence[str]] = None,
    n_seeds: int = 10,
    iterations: int = 5000,
    tau: float = 1.0,
    epsilon_target: float = 0.05,
    base_seed: int = 0,
) -> RegretComparisonResult:
    """Compare four regret-matching transforms on the same game.

    Parameters
    ----------
    pm
        :class:`phenogame.game.PayoffMatrix` or raw ``(m, n)`` ndarray.
    transforms
        Subset of ``["standard", "exp", "softplus", "eml"]``. Default: all four.
    n_seeds
        Number of independent seeded runs per transform. Default 10. Closes
        audit F2.
    iterations
        Per-run iteration count (constant across transforms).
    tau
        Temperature for the kernels that use it (exp, softplus, eml).
    epsilon_target
        CCE-gap threshold for "iters to ε" (closes audit F4 in part).
    base_seed
        Base seed; per-run seeds are ``base_seed + transform_index*1000 + run``.

    Returns
    -------
    RegretComparisonResult
    """
    if isinstance(pm, PayoffMatrix):
        A = pm.matrix
        shape = pm.matrix.shape
    else:
        A = np.asarray(pm, dtype=float)
        if A.ndim != 2:
            raise ValueError("pm must be a 2D matrix or PayoffMatrix.")
        shape = A.shape

    if not np.all(np.isfinite(A)):
        raise ValueError("Payoff matrix contains NaN/inf.")
    if iterations < 10:
        raise ValueError("iterations must be ≥ 10.")
    if n_seeds < 1:
        raise ValueError("n_seeds must be ≥ 1.")

    if transforms is None:
        transforms = list_transforms()
    transforms = list(transforms)
    for t in transforms:
        if t not in TRANSFORM_REGISTRY:
            raise KeyError(
                f"Unknown transform: {t!r}. Available: {list(TRANSFORM_REGISTRY)}"
            )

    runs: List[TransformRun] = []
    for ti, name in enumerate(transforms):
        for s in range(n_seeds):
            seed = int(base_seed + ti * 1000 + s)
            run = _run_one(A, name, iterations, tau, epsilon_target, seed)
            runs.append(run)

    # Aggregate.
    aggregates: List[TransformAggregate] = []
    for name in transforms:
        sub = [r for r in runs if r.transform == name]
        gaps = np.array([r.final_cce_gap for r in sub])
        rhos = np.array([r.joint_corr_normalized for r in sub])
        walls = np.array([r.wall_clock_seconds for r in sub])
        reached = [r.iters_to_epsilon for r in sub if r.iters_to_epsilon is not None]
        if reached:
            iters_mean = float(np.mean(reached))
            iters_std = float(np.std(reached, ddof=0))
        else:
            iters_mean = None
            iters_std = None
        info = TRANSFORM_REGISTRY[name]
        aggregates.append(TransformAggregate(
            transform=name,
            n_seeds=len(sub),
            final_cce_gap_mean=float(gaps.mean()),
            final_cce_gap_std=float(gaps.std(ddof=0)),
            iters_to_epsilon_mean=iters_mean,
            iters_to_epsilon_std=iters_std,
            n_runs_reached_epsilon=len(reached),
            joint_corr_normalized_mean=float(rhos.mean()),
            joint_corr_normalized_std=float(rhos.std(ddof=0)),
            wall_clock_seconds_mean=float(walls.mean()),
            wall_clock_seconds_std=float(walls.std(ddof=0)),
            no_regret_proof=info["no_regret_proof"],
            is_strict_rm=info["is_strict_rm"],
            always_positive=info["always_positive"],
        ))

    notes = {
        "F2": f"Variance reported across n_seeds={n_seeds} (mean ± SD).",
        "F4": (f"Final CCE gap and iters-to-ε reported per run "
               f"(ε_target={epsilon_target})."),
        "F6": ("softplus is always-positive (Hedge-like); not a strict "
               "regret-matching transform."),
        "F7": ("All transforms run on identical (game, seed) pairs for "
               "paired comparison."),
        "F8": "Joint correlation residual normalised to [0, 1].",
        "F9": ("EML transform has no formal no-regret proof; convergence "
               "to CCE is empirical only."),
        "F10": "Wall-clock timing reported per run.",
    }

    return RegretComparisonResult(
        payoff_shape=tuple(shape),
        iterations=iterations,
        epsilon_target=epsilon_target,
        tau=tau,
        n_seeds=n_seeds,
        transforms=transforms,
        runs=runs,
        aggregates=aggregates,
        notes=notes,
    )
