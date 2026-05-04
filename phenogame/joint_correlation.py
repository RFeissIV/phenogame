"""
phenogame.joint_correlation — Normalized joint-distribution correlation residual.

Closes EML-RML twenty-pass audit finding F8: the audit's R sibling pipeline
computes a raw L1 distance ``sum(|q − p_i p_j|)`` whose maximum scales with
game dimensions, so the fixed 0.10 threshold is not comparable across
2×2 / 3×3 / 5×5 games. This module normalises the residual to ``[0, 1]``.

Definition
----------

Given an empirical joint distribution ``q`` of shape ``(m, n)`` over
player-1 actions × player-2 actions:

.. math::
    \\rho(q) = \\frac{1}{2\\,(1 - 1/\\min(m, n))}\\,
              \\sum_{i, j} \\bigl|q_{ij} - q_{i\\cdot} q_{\\cdot j}\\bigr|

where ``q_{i⋅} = ∑_j q_{ij}`` and ``q_{⋅j} = ∑_i q_{ij}`` are the marginals.

Properties
----------

- ``ρ(q) = 0`` iff ``q`` factorises (product-form independent play).
- ``ρ(q) ∈ [0, 1]`` for any valid joint distribution.
- The normaliser ``2(1 − 1/min(m, n))`` is the supremum of the unnormalised
  L1 distance over all joint distributions on a finite ``m × n`` support
  with given marginal shapes; it makes the residual directly comparable
  across game dimensions.

This is **not** the same quantity as the binary-stochastic-choice triangle
audit in :mod:`phenogame.binary_choice`. That module operates on pairwise
preference probabilities ``P[i, j] = Pr(i beats j)`` over candidate models;
this module operates on the empirical joint play distribution over the
two-player action space of the data-induced game. The two live next to
each other in PhenoGame for a reason: they answer different questions.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd


ArrayLike = Union[np.ndarray, pd.DataFrame]


def joint_correlation_residual_normalized(empirical_joint: ArrayLike) -> float:
    """Normalised L1 residual ``ρ(q) ∈ [0, 1]``; 0 ⇔ q factorises.

    Parameters
    ----------
    empirical_joint
        Empirical joint distribution ``q`` of shape ``(m, n)``. Entries
        must be non-negative and sum to 1 (within tolerance).

    Returns
    -------
    float
        Normalised correlation residual in ``[0, 1]``.

    Raises
    ------
    ValueError
        If the input is not 2D, not non-negative, or does not sum to 1.
    """
    if isinstance(empirical_joint, pd.DataFrame):
        q = empirical_joint.to_numpy(dtype=float)
    else:
        q = np.asarray(empirical_joint, dtype=float)
    if q.ndim != 2:
        raise ValueError("empirical_joint must be 2D.")
    if not np.all(np.isfinite(q)):
        raise ValueError("empirical_joint contains NaN/inf.")
    if q.min() < -1e-12:
        raise ValueError("empirical_joint contains negative entries.")
    total = q.sum()
    if not np.isclose(total, 1.0, atol=1e-9):
        raise ValueError(
            f"empirical_joint does not sum to 1 (sum={total:.6e}). "
            "Pass a probability distribution."
        )

    m, n = q.shape
    if m < 2 or n < 2:
        # Degenerate support — no correlation possible by definition.
        return 0.0

    p_row = q.sum(axis=1, keepdims=True)
    p_col = q.sum(axis=0, keepdims=True)
    product = p_row @ p_col  # outer product of marginals
    raw = float(np.sum(np.abs(q - product)))

    denom = 2.0 * (1.0 - 1.0 / float(min(m, n)))
    if denom <= 0:
        return 0.0
    rho = raw / denom
    # Clamp for numerical safety.
    return float(min(max(rho, 0.0), 1.0))


def joint_correlation_residual_raw(empirical_joint: ArrayLike) -> float:
    """Unnormalised L1 distance between ``q`` and its product factorisation.

    Provided for direct comparison with the audit doc's R reference
    implementation. Use :func:`joint_correlation_residual_normalized` for
    cross-dimension comparisons.
    """
    if isinstance(empirical_joint, pd.DataFrame):
        q = empirical_joint.to_numpy(dtype=float)
    else:
        q = np.asarray(empirical_joint, dtype=float)
    if q.ndim != 2:
        raise ValueError("empirical_joint must be 2D.")
    p_row = q.sum(axis=1, keepdims=True)
    p_col = q.sum(axis=0, keepdims=True)
    return float(np.sum(np.abs(q - p_row @ p_col)))
