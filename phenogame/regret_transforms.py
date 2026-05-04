"""
phenogame.regret_transforms — Regret-to-weight transforms for no-regret dynamics.

Four transforms, all of the same shape ``ψ : R^A → R^A_+`` mapping a vector
of cumulative external regrets to non-negative action weights (which are
then normalised into a probability distribution before sampling):

- :func:`psi_standard`     — Hart-Mas-Colell positive-part regret matching
- :func:`psi_exponential`  — exp(z), the Hedge / multiplicative-weights kernel
- :func:`psi_softplus`     — log(1 + exp(z)), Hedge-like with continuous floor
- :func:`psi_eml`          — exp(z) − log(1 + exp(−z)), the EML-RML transform

The four are exposed as a comparator family so the user can rerun the same
data-induced game under multiple regret-matching dynamics and report which
one converges fastest / produces the most-correlated empirical play / has the
tightest seed-to-seed variance.

Mathematical claims
-------------------
- :func:`psi_standard` has a formal external-regret bound (Hart & Mas-Colell
  2000; Blum & Mansour 2007). Time-averaged play converges to a CCE.
- :func:`psi_exponential` and Hedge with this kernel have the
  Freund-Schapire (1999) bound ``R_T ≤ √(2 T ln |A|)``, so time-averaged
  play converges to ε-CCE with ``ε = √(2 ln|A| / T)``.
- :func:`psi_softplus` is **always positive** for finite inputs (the
  log(1 + exp(z)) floor never reaches zero). It is therefore Hedge-like
  with a continuous floor, **not** a strict regret-matching transform; this
  matches finding F6 of the EML-RML twenty-pass audit.
- :func:`psi_eml` has no formal no-regret proof; convergence to CCE is
  empirically observed but not theoretically guaranteed (audit finding F9).

References
----------
Hart, S. and Mas-Colell, A. (2000). A simple adaptive procedure leading
    to correlated equilibrium. Econometrica 68(5).
Blum, A. and Mansour, Y. (2007). From external to internal regret.
    Journal of Machine Learning Research.
Freund, Y. and Schapire, R. E. (1999). Adaptive game playing using
    multiplicative weights. Games and Economic Behavior 29.
"""

from __future__ import annotations

import numpy as np


_CLIP = 50.0  # match equilibrium.py's existing clip range for numerical safety


def _clip(z: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(z, dtype=float), -_CLIP, _CLIP)


def psi_standard(R: np.ndarray) -> np.ndarray:
    """Standard Hart-Mas-Colell positive-part regret matching.

    .. math:: \\psi_{\\text{std}}(R) = \\max(R, 0)

    The textbook regret-matching transform. Has a formal external-regret
    bound (Hart-Mas-Colell 2000; Blum-Mansour 2007); time-averaged play
    converges to CCE.

    Parameters
    ----------
    R
        Cumulative external regret vector.

    Returns
    -------
    numpy.ndarray
        Non-negative weights of the same shape as ``R``.
    """
    return np.maximum(np.asarray(R, dtype=float), 0.0)


def psi_exponential(R: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """Exponential / Hedge / multiplicative-weights kernel.

    .. math:: \\psi_{\\text{exp}}(R) = \\exp(R / \\tau)

    Yields the Hedge no-regret dynamic when normalised. Has the
    Freund-Schapire (1999) bound ``R_T ≤ √(2 T ln |A|)``.

    Parameters
    ----------
    R
        Cumulative external regret vector.
    tau
        Temperature; smaller values are more concentrated on the best action.

    Returns
    -------
    numpy.ndarray
        Strictly positive weights of the same shape as ``R``.
    """
    return np.exp(_clip(np.asarray(R, dtype=float) / float(tau)))


def psi_softplus(R: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """Softplus regret kernel ``log(1 + exp(z))``.

    .. math:: \\psi_{\\text{sp}}(R) = \\log(1 + \\exp(R / \\tau))

    Note (audit finding F6)
    -----------------------
    softplus is **always positive** for finite inputs — every action,
    including those with arbitrarily-negative regret, gets non-zero weight.
    This makes it structurally Hedge-like with a continuous floor, **not**
    a strict regret-matching transform. Including it as one of four
    comparators is fine but the user must understand that convergence
    differences vs the strict-RM transforms (standard, exp clamped, EML)
    may reflect this structural difference and not the mathematical shape
    of the kernel.

    Parameters
    ----------
    R
        Cumulative external regret vector.
    tau
        Temperature.

    Returns
    -------
    numpy.ndarray
        Strictly positive weights of the same shape as ``R``.
    """
    z = _clip(np.asarray(R, dtype=float) / float(tau))
    # Numerically stable softplus.
    return np.where(z > 0, z + np.log1p(np.exp(-z)), np.log1p(np.exp(z)))


def psi_eml(R: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """EML-RML transform from the EML-RML twenty-pass audit.

    .. math:: \\psi_{\\text{EML}}(R) = \\max\\bigl(\\exp(R/\\tau)
              - \\log(1 + \\exp(-R/\\tau)),\\ 0\\bigr)

    The argument convention matches the audited R reference implementation
    (``psi_eml_regret`` in the EML-RML R pipeline) and PhenoGame's existing
    private ``_eml_regret_transform`` in ``equilibrium.py``.

    No-regret status (audit finding F9)
    -----------------------------------
    No formal no-regret proof is known for :math:`\\psi_{\\text{EML}}`. The
    transform is non-negative by construction (zero-clamp), super-linear
    for large positive z, and asymptotically exp-like, which makes Hannan
    consistency *plausible* — but the substantive condition (sufficient
    weight on high-regret actions) is not formally proven for this exact
    kernel. The :func:`phenogame.equilibrium.certify_empirical` certificate
    therefore reports the *measured* gap, not a *guaranteed* bound.

    Parameters
    ----------
    R
        Cumulative external regret vector.
    tau
        Temperature.

    Returns
    -------
    numpy.ndarray
        Non-negative weights of the same shape as ``R``.
    """
    z = _clip(np.asarray(R, dtype=float) / float(tau))
    # exp(z) - log(1 + exp(-z)) = exp(z) - softplus(-z), max with 0.
    minus_z_softplus = np.where(-z > 0,
                                -z + np.log1p(np.exp(z)),
                                np.log1p(np.exp(-z)))
    return np.maximum(np.exp(z) - minus_z_softplus, 0.0)


# Public registry consumed by the comparator panel.
TRANSFORM_REGISTRY = {
    "standard": {
        "fn": psi_standard,
        "uses_tau": False,
        "no_regret_proof": "Hart-Mas-Colell 2000; Blum-Mansour 2007",
        "is_strict_rm": True,
        "always_positive": False,
    },
    "exp": {
        "fn": psi_exponential,
        "uses_tau": True,
        "no_regret_proof": "Freund-Schapire 1999",
        "is_strict_rm": False,
        "always_positive": True,
    },
    "softplus": {
        "fn": psi_softplus,
        "uses_tau": True,
        "no_regret_proof": "informal; Hedge-like with continuous floor (audit F6)",
        "is_strict_rm": False,
        "always_positive": True,
    },
    "eml": {
        "fn": psi_eml,
        "uses_tau": True,
        "no_regret_proof": "empirical only; not formally proven (audit F9)",
        "is_strict_rm": True,
        "always_positive": False,
    },
}


def list_transforms() -> list:
    """Return the canonical ordering of transform names."""
    return ["standard", "exp", "softplus", "eml"]
