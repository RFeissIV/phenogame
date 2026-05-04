"""
phenogame.bsc_game_bridge — Binary Stochastic Choice ↔ Cooperative game bridge.

This module has two deliberately separated pieces.

1. An **external-flow coalition summary**
   ``v(S) = sum_{i in S, j notin S} P[i,j]``. This is a descriptive
   cooperative-game diagnostic of outward pairwise preference flow.

2. A theorem-safe **Gilboa-Monderer u=0 necessary inequality**.  For a
   supplied coefficient matrix ``a_ij``, the theorem with auxiliary game
   ``u = 0`` gives the valid but generally loose bound::

       sum_{i != j} a_ij p_ij <= sum_i max_{S subset N\\{i}} sum_{j in S} a_ij

   which reduces to the sum of positive off-diagonal coefficients in each
   row.  This is a diagnostic necessary-condition audit only; it does not
   prove full rationalisability.

Public API
----------

- :func:`coalition_value_from_pairwise(P, coalition)`
- :func:`coalition_game(P)`
- :func:`marginal_contribution(v, i, N)`
- :func:`max_marginal_contributions(v, N)`
- :func:`gilboa_monderer_u0_bound(P, a_ij=None)`
- :func:`binary_choice_game_certificate(P)`
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, FrozenSet, Iterable, Optional, Union

import numpy as np
import pandas as pd

from .binary_choice import check_binary_choice_constraints


ArrayLike = Union[np.ndarray, pd.DataFrame]


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _validate_P(P: ArrayLike) -> np.ndarray:
    if isinstance(P, pd.DataFrame):
        arr = P.to_numpy(dtype=float)
    else:
        arr = np.asarray(P, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("P must be a square 2D array.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("P contains NaN/inf.")
    return arr


# ────────────────────────────────────────────────────────────────────────────
# Coalition value
# ────────────────────────────────────────────────────────────────────────────

def coalition_value_from_pairwise(
    P: ArrayLike,
    coalition: Iterable[int],
) -> float:
    """Coalition value ``v(S)`` from pairwise preferences ``P``.

    We use the natural symmetric construction::

        v(S) = sum_{i in S, j notin S} P[i, j]

    so ``v(empty) = 0``, ``v(N)`` is the total off-diagonal mass directed
    out of all-vs-none pairs (which by symmetry equals the sum of all
    off-diagonal ``P[i, j]`` for ``i in S, j notin S``).  This is the
    standard "external preference flow" coalition value used in the
    binary-choice / cooperative-game correspondence.

    Parameters
    ----------
    P
        ``(n, n)`` preference probability matrix.
    coalition
        Iterable of indices in ``{0, ..., n-1}`` defining ``S``.

    Returns
    -------
    float
        The coalition value ``v(S)``.
    """
    arr = _validate_P(P)
    n = arr.shape[0]
    S = sorted(set(int(i) for i in coalition))
    for i in S:
        if i < 0 or i >= n:
            raise ValueError(f"Coalition index out of range: {i}")
    if len(S) == 0:
        return 0.0
    Sset = set(S)
    out = sum(arr[i, j] for i in S for j in range(n) if j not in Sset)
    return float(out)


def coalition_game(P: ArrayLike) -> Dict[FrozenSet[int], float]:
    """Enumerate the coalitional game ``v: 2^N → R`` from ``P``.

    Returns a dict keyed by frozenset coalitions.  For ``n`` ≥ ~16 this
    function will be expensive; callers should prefer the marginal-only
    routines below for large ``n``.

    Parameters
    ----------
    P
        ``(n, n)`` preference probability matrix.

    Returns
    -------
    dict
        Mapping ``frozenset(coalition) -> v(coalition)``.
    """
    arr = _validate_P(P)
    n = arr.shape[0]
    if n > 20:
        raise ValueError(
            f"coalition_game enumerates 2^n coalitions; n={n} is too large. "
            "Use marginal_contribution / max_marginal_contributions instead."
        )
    out: Dict[FrozenSet[int], float] = {}
    indices = list(range(n))
    for r in range(0, n + 1):
        for S in combinations(indices, r):
            out[frozenset(S)] = coalition_value_from_pairwise(arr, S)
    return out


# ────────────────────────────────────────────────────────────────────────────
# Marginal contributions
# ────────────────────────────────────────────────────────────────────────────

def marginal_contribution(
    v: Dict[FrozenSet[int], float],
    i: int,
    N: Iterable[int],
    *,
    coalition: Optional[Iterable[int]] = None,
) -> float:
    """Marginal contribution of ``i`` to a coalition (default: ``N \\ {i}``).

    Parameters
    ----------
    v
        Coalitional game as returned by :func:`coalition_game`.
    i
        Player index.
    N
        Player set.
    coalition
        Coalition to which ``i`` is added.  If omitted, ``N \\ {i}`` is used.

    Returns
    -------
    float
        ``v(coalition ∪ {i}) - v(coalition)``.
    """
    Nset = frozenset(int(x) for x in N)
    if coalition is None:
        S = Nset - {int(i)}
    else:
        S = frozenset(int(x) for x in coalition) - {int(i)}
    if int(i) not in Nset:
        raise ValueError(f"Player {i} not in N.")
    if S not in v or (S | {int(i)}) not in v:
        raise KeyError("v does not contain the required coalitions.")
    return float(v[S | {int(i)}] - v[S])


def max_marginal_contributions(
    v: Dict[FrozenSet[int], float],
    N: Iterable[int],
) -> Dict[int, float]:
    """Compute ``v_i* = max_{S ⊂ N, i ∉ S} [v(S ∪ {i}) − v(S)]`` for each ``i``.

    Parameters
    ----------
    v
        Coalitional game.
    N
        Player set.

    Returns
    -------
    dict
        ``{i: v_i*}`` for every ``i`` in ``N``.
    """
    Nset = sorted(int(x) for x in N)
    n = len(Nset)
    if n == 0:
        return {}
    out: Dict[int, float] = {}
    for i in Nset:
        others = [j for j in Nset if j != i]
        best = -np.inf
        for r in range(0, len(others) + 1):
            for S in combinations(others, r):
                sset = frozenset(S)
                if sset not in v or (sset | {i}) not in v:
                    raise KeyError(
                        "v does not contain all required coalitions for max_marginal_contributions."
                    )
                m = float(v[sset | {i}] - v[sset])
                if m > best:
                    best = m
        out[i] = best if np.isfinite(best) else 0.0
    return out




def gilboa_monderer_u0_bound(
    P: ArrayLike,
    *,
    a_ij: Optional[ArrayLike] = None,
    tol: float = 1e-8,
) -> Dict[str, object]:
    """Compute the Gilboa-Monderer equivalence inequality for ``u = 0``.

    This is the theorem-valid conservative bound:

    ``lhs = sum_{i != j} a_ij p_ij``

    ``rhs = sum_i max_{S subset N\\{i}} sum_{j in S} a_ij``

    Because ``u = 0``, the inner maximum is simply the sum of positive
    off-diagonal coefficients in row ``i``.  The bound is therefore valid
    for every supplied coefficient matrix ``A`` but can be loose.  It should
    be reported as a necessary-condition diagnostic, not as a full
    rationalisability proof.
    """
    arr = _validate_P(P)
    n = arr.shape[0]
    if a_ij is None:
        A = np.ones((n, n), dtype=float) - np.eye(n, dtype=float)
    else:
        if isinstance(a_ij, pd.DataFrame):
            A = a_ij.to_numpy(dtype=float)
        else:
            A = np.asarray(a_ij, dtype=float)
        if A.shape != (n, n):
            raise ValueError("a_ij must have the same shape as P.")
        if not np.all(np.isfinite(A)):
            raise ValueError("a_ij contains NaN/inf.")

    A_off = A.copy()
    np.fill_diagonal(A_off, 0.0)
    lhs = float(np.sum(A_off * arr))
    row_positive_bounds = np.maximum(A_off, 0.0).sum(axis=1)
    rhs = float(row_positive_bounds.sum())
    return {
        "lhs": lhs,
        "rhs": rhs,
        "holds": bool(lhs <= rhs + tol),
        "slack": float(rhs - lhs),
        "row_positive_bounds": [float(x) for x in row_positive_bounds],
        "auxiliary_game": "u=0",
        "note": (
            "Gilboa-Monderer equivalence inequality evaluated with u=0. "
            "This is theorem-valid but generally loose; passing it is not a "
            "full rationalisability proof."
        ),
    }


# ────────────────────────────────────────────────────────────────────────────
# Diagnostic / necessary-condition certificate
# ────────────────────────────────────────────────────────────────────────────

def binary_choice_game_certificate(
    P: ArrayLike,
    *,
    a_ij: Optional[ArrayLike] = None,
    tol: float = 1e-8,
    max_n: int = 13,
) -> Dict[str, object]:
    """Diagnostic certificate from binary stochastic choice via cooperative game.

    Computes:

    1. The binary stochastic choice constraint audit
       (:func:`phenogame.binary_choice.check_binary_choice_constraints`).
    2. A descriptive external-flow coalitional game
       ``v(S)=sum_{i in S,j notin S}P[i,j]``.
    3. The theorem-valid Gilboa-Monderer ``u=0`` inequality returned by
       :func:`gilboa_monderer_u0_bound`.

    The result is **diagnostic only**.  Passing these checks is not
    sufficient for full rationalisability; failure flags inconsistent
    pairwise structure or incompatible weights.

    Parameters
    ----------
    P
        ``(n, n)`` preference probability matrix.
    a_ij
        Optional ``(n, n)`` weight matrix.  Defaults to ``1`` off-diagonal,
        ``0`` on-diagonal.
    tol
        Numerical tolerance.
    max_n
        Refuse to enumerate the cooperative game above this size, for
        runtime safety.  The default of ``13`` accommodates the canonical
        EML pipeline (13 candidate families) which requires ``8192``
        coalitions and is well within budget.

    Returns
    -------
    dict
        Certificate fields:

        - ``n_alternatives``
        - ``constraints``               (binary-choice audit)
        - ``v_N``                        ``v(N)``
        - ``max_marginal_contributions`` ``{i: v_i*}``
        - ``sum_max_marginals``          sum_i v_i*
        - ``lhs``                        sum_{i!=j} a_ij p_ij
        - ``rhs``                        sum_i v_i* + v(N)
        - ``inequality_holds``           bool, ``lhs <= rhs + tol``
        - ``slack``                      ``rhs - lhs``
        - ``note``                       Conservative scope statement.
    """
    arr = _validate_P(P)
    n = arr.shape[0]
    if n > max_n:
        raise ValueError(
            f"binary_choice_game_certificate: n={n} exceeds max_n={max_n}; "
            "this enumerates 2^n coalitions."
        )

    if a_ij is None:
        A = np.ones((n, n), dtype=float) - np.eye(n, dtype=float)
    else:
        if isinstance(a_ij, pd.DataFrame):
            A = a_ij.to_numpy(dtype=float)
        else:
            A = np.asarray(a_ij, dtype=float)
        if A.shape != (n, n):
            raise ValueError("a_ij must have the same shape as P.")
        if not np.all(np.isfinite(A)):
            raise ValueError("a_ij contains NaN/inf.")

    # 1. Constraint audit on P.
    constraints = check_binary_choice_constraints(arr, tol=tol)

    # 2. Build cooperative game.
    v = coalition_game(arr)
    N = list(range(n))
    v_N = float(v[frozenset(N)])

    # 3. Per-player maximum marginal contributions.
    v_max = max_marginal_contributions(v, N)
    sum_v_max = float(sum(v_max.values()))

    # 4. Gilboa-Monderer theorem-valid u=0 necessary inequality.
    gm = gilboa_monderer_u0_bound(arr, a_ij=A, tol=tol)
    lhs = float(gm["lhs"])
    rhs = float(gm["rhs"])
    holds = bool(gm["holds"])
    slack = float(gm["slack"])

    return {
        "n_alternatives": int(n),
        "constraints": constraints,
        "v_N": v_N,
        "max_marginal_contributions": {int(k): float(val) for k, val in v_max.items()},
        "sum_max_marginals": sum_v_max,
        "lhs": lhs,
        "rhs": rhs,
        "inequality_holds": holds,
        "slack": slack,
        "gilboa_monderer_u0": gm,
        "external_flow_note": (
            "v_N and max_marginal_contributions are computed from the "
            "external-flow coalition diagnostic v(S)=sum_{i in S,j notin S}P[i,j]."
        ),
        "note": (
            "Diagnostic / necessary-condition audit. lhs/rhs/inequality_holds "
            "use the theorem-valid Gilboa-Monderer u=0 bound. Passing this "
            "bound is not sufficient for full rationalisability; failure flags "
            "inconsistent pairwise structure or incompatible weights."
        ),
    }
