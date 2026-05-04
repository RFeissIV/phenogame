"""
phenogame.binary_choice — Pairwise stochastic-choice screen for EML model families.

This module implements the binary stochastic choice layer used to rank the
13 candidate EML model families before they are wired into the finite
Farmer/Algorithm vs Nature game.

Mathematical reference
----------------------
We follow the binary stochastic choice / pairwise probability framework
discussed in Gilboa & Monderer (and subsequent work) where, for a finite
set of alternatives N = {1, ..., 13}, one defines a pairwise preference
probability::

    p_ij = Prob(model i outperforms model j across bootstrap/scenario evals)

The classical *binary stochastic choice* consistency conditions are:

1. Complementarity:                p_ij + p_ji = 1
2. Non-negativity:                 p_ij >= 0
3. Triangle / weak-stochastic-transitivity:  p_ij + p_jk + p_ki <= 2

These are *necessary* conditions for rationalisable pairwise choice.  Failures
flag inconsistent / cyclical preference structure in the underlying scores.

This is a *diagnostic certificate* / *necessary-condition audit* / *pairwise
consistency screen* — not a proof of full rationalisability, and not a new
equilibrium theorem.

Public API
----------

- :func:`pairwise_win_matrix(scores)`
- :func:`preference_probabilities(score_matrix)`
- :func:`check_binary_choice_constraints(P)`
- :func:`triangle_violations(P)`
- :func:`stochastic_choice_summary(P)`
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd


ArrayLike = Union[np.ndarray, pd.DataFrame]


# ────────────────────────────────────────────────────────────────────────────
# 1. Pairwise win matrix from scores
# ────────────────────────────────────────────────────────────────────────────

def pairwise_win_matrix(scores: pd.DataFrame) -> pd.DataFrame:
    """Compute a pairwise win-count matrix from a score table.

    Each row of ``scores`` is one evaluation/bootstrap repetition; each column
    is one candidate model.  Entry ``W[i, j]`` is the number of repetitions in
    which model ``i`` strictly outperforms model ``j``.  Ties contribute 0.5
    to both ``W[i, j]`` and ``W[j, i]`` (Massey-style tie handling), which keeps
    ``W + W.T`` equal to ``n_repetitions`` on off-diagonals and preserves the
    complementarity property below downstream.

    Parameters
    ----------
    scores
        DataFrame of shape ``(n_repetitions, n_models)`` with finite values.

    Returns
    -------
    pandas.DataFrame
        Square, model-indexed win matrix.

    Raises
    ------
    TypeError
        If ``scores`` is not a DataFrame.
    ValueError
        If ``scores`` is empty, has fewer than 2 columns, or has non-finite
        entries.
    """
    if not isinstance(scores, pd.DataFrame):
        raise TypeError("scores must be a pandas DataFrame.")
    if scores.shape[0] == 0 or scores.shape[1] < 2:
        raise ValueError("scores must have >=1 row and >=2 columns.")
    arr = scores.to_numpy(dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError("scores contains NaN/inf; clean before calling.")

    n_models = arr.shape[1]
    W = np.zeros((n_models, n_models), dtype=float)
    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                continue
            si = arr[:, i]
            sj = arr[:, j]
            W[i, j] = float(np.sum(si > sj) + 0.5 * np.sum(si == sj))
    return pd.DataFrame(W, index=scores.columns, columns=scores.columns)


def preference_probabilities(score_matrix: ArrayLike) -> np.ndarray:
    """Convert a pairwise win matrix to a preference probability matrix ``P``.

    Given ``W`` of shape ``(n, n)``, define::

        P[i, j] = W[i, j] / (W[i, j] + W[j, i])         if i != j
        P[i, i] = 0.5                                   (by convention)

    Empty (zero-zero) cells are filled with 0.5 (no preference).  The result
    automatically satisfies ``P + P.T = 1`` on off-diagonals, modulo tie
    handling.

    Parameters
    ----------
    score_matrix
        ``(n, n)`` win matrix as a DataFrame or ndarray.

    Returns
    -------
    numpy.ndarray
        ``(n, n)`` preference probability matrix with all entries in [0, 1].

    Raises
    ------
    ValueError
        If ``score_matrix`` is not square or contains non-finite values.
    """
    if isinstance(score_matrix, pd.DataFrame):
        W = score_matrix.to_numpy(dtype=float)
    else:
        W = np.asarray(score_matrix, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("score_matrix must be a square 2D array.")
    if not np.all(np.isfinite(W)):
        raise ValueError("score_matrix contains NaN/inf.")

    n = W.shape[0]
    P = np.full((n, n), 0.5, dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            denom = W[i, j] + W[j, i]
            if denom > 0:
                P[i, j] = float(W[i, j] / denom)
            else:
                P[i, j] = 0.5
    # Defensive clamp to handle any numerical drift.
    P = np.clip(P, 0.0, 1.0)
    return P


# ────────────────────────────────────────────────────────────────────────────
# 2. Constraint checking
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


def check_binary_choice_constraints(P: ArrayLike, *, tol: float = 1e-8) -> Dict[str, object]:
    """Diagnostic check of binary stochastic choice constraints.

    Verifies the three necessary consistency conditions from the docstring of
    this module: complementarity, non-negativity, and the triangle inequality
    ``p_ij + p_jk + p_ki <= 2`` for all distinct ``i, j, k``.

    Parameters
    ----------
    P
        ``(n, n)`` preference probability matrix.
    tol
        Numerical tolerance applied to all checks.

    Returns
    -------
    dict
        Diagnostic summary with the following keys:

        - ``complementarity_ok`` : bool
        - ``max_complementarity_error`` : float
        - ``nonnegativity_ok`` : bool
        - ``min_value`` : float
        - ``max_value`` : float
        - ``triangle_ok`` : bool
        - ``n_triangle_violations`` : int
        - ``max_triangle_excess`` : float
        - ``all_passed`` : bool
        - ``tolerance`` : float
        - ``note`` : str
    """
    arr = _validate_P(P)
    n = arr.shape[0]

    # Complementarity
    comp_err = float(np.max(np.abs(arr + arr.T - 1.0))) if n > 0 else 0.0
    comp_ok = bool(comp_err <= tol + 1e-12)

    # Non-negativity (and upper bound)
    min_v = float(arr.min()) if n > 0 else 0.0
    max_v = float(arr.max()) if n > 0 else 1.0
    nn_ok = bool(min_v >= -tol and max_v <= 1.0 + tol)

    # Triangle: p_ij + p_jk + p_ki <= 2
    n_violations = 0
    max_excess = 0.0
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            for k in range(n):
                if k == i or k == j:
                    continue
                s = arr[i, j] + arr[j, k] + arr[k, i]
                excess = s - 2.0
                if excess > tol:
                    n_violations += 1
                    if excess > max_excess:
                        max_excess = float(excess)
    tri_ok = bool(n_violations == 0)

    return {
        "complementarity_ok": comp_ok,
        "max_complementarity_error": comp_err,
        "nonnegativity_ok": nn_ok,
        "min_value": min_v,
        "max_value": max_v,
        "triangle_ok": tri_ok,
        "n_triangle_violations": int(n_violations),
        "max_triangle_excess": float(max_excess),
        "all_passed": bool(comp_ok and nn_ok and tri_ok),
        "tolerance": float(tol),
        "note": (
            "Diagnostic / necessary-condition audit of binary stochastic "
            "choice consistency. Does NOT prove full rationalisability."
        ),
    }


def triangle_violations(P: ArrayLike, *, tol: float = 1e-8) -> pd.DataFrame:
    """Return all triples ``(i, j, k)`` that violate the triangle inequality.

    Parameters
    ----------
    P
        ``(n, n)`` preference probability matrix.
    tol
        Numerical tolerance.

    Returns
    -------
    pandas.DataFrame
        Columns: ``i, j, k, p_ij, p_jk, p_ki, sum, excess``.  Empty when no
        violations are present.
    """
    arr = _validate_P(P)
    n = arr.shape[0]
    rows = []
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            for k in range(n):
                if k == i or k == j:
                    continue
                s = arr[i, j] + arr[j, k] + arr[k, i]
                excess = s - 2.0
                if excess > tol:
                    rows.append({
                        "i": i, "j": j, "k": k,
                        "p_ij": float(arr[i, j]),
                        "p_jk": float(arr[j, k]),
                        "p_ki": float(arr[k, i]),
                        "sum": float(s),
                        "excess": float(excess),
                    })
    return pd.DataFrame(rows, columns=["i", "j", "k", "p_ij", "p_jk", "p_ki", "sum", "excess"])


# ────────────────────────────────────────────────────────────────────────────
# 3. Aggregate summary used by the certificate
# ────────────────────────────────────────────────────────────────────────────

def stochastic_choice_summary(P: ArrayLike, *,
                              labels: Optional[List[str]] = None,
                              tol: float = 1e-8) -> Dict[str, object]:
    """High-level summary of pairwise preferences and consistency.

    Computes Borda-style scores ``B_i = sum_{j != i} P[i, j]`` (the row sums
    over off-diagonal entries), produces a ranking, and bundles the constraint
    audit.

    Parameters
    ----------
    P
        ``(n, n)`` preference probability matrix.
    labels
        Optional labels for the ``n`` alternatives; defaults to ``"m0"..."m{n-1}"``.
    tol
        Numerical tolerance forwarded to :func:`check_binary_choice_constraints`.

    Returns
    -------
    dict
        Summary with keys:

        - ``n_alternatives`` : int
        - ``labels`` : list[str]
        - ``borda_scores`` : list[float]
        - ``ranking`` : list[str]   (best -> worst)
        - ``constraints`` : dict     (output of :func:`check_binary_choice_constraints`)
        - ``note`` : str
    """
    arr = _validate_P(P)
    n = arr.shape[0]
    if labels is None:
        labels = [f"m{i}" for i in range(n)]
    if len(labels) != n:
        raise ValueError("labels length must equal P dimension.")

    # Borda-style score: sum of row off-diagonal preferences.
    eye = np.eye(n, dtype=bool)
    masked = np.where(eye, 0.0, arr)
    borda = masked.sum(axis=1)
    order = np.argsort(-borda)
    ranking = [labels[idx] for idx in order]

    constraints = check_binary_choice_constraints(arr, tol=tol)
    return {
        "n_alternatives": int(n),
        "labels": list(labels),
        "borda_scores": [float(x) for x in borda],
        "ranking": ranking,
        "constraints": constraints,
        "note": (
            "Borda-style row sums over P provide an aggregate ranking; "
            "the constraint audit flags pairwise inconsistencies."
        ),
    }
