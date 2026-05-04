"""
phenogame.eml_families — 13 candidate EML-derived model families.

Defines a fixed library of 13 candidate function grammars / payoff
generators used as the action set for the binary-stochastic-choice
ranking layer.  These are *candidate models*, not 13 separate game
players: the game itself remains two-player (Algorithm/Selector vs Nature).

Conservative scope
------------------
Each family is a deliberately simple regressor on top of features built
from the EML operator ``eml(x, y) = exp(x) - ln(y)``.  We keep the
families small, interpretable, and numerically safe.  The intent is:

- They are candidate **payoff generators**, not new equilibrium concepts.
- They are NOT a claim of expressive completeness or a new theorem.
- They are NOT exhaustive — they are a fixed, auditable shortlist.

Public API
----------

- :func:`define_13_eml_families`
- :func:`fit_eml_family`
- :func:`fit_all_eml_families`
- :func:`predict_eml_family`
- :func:`rank_eml_families`
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .eml_tree import EMLTreeRegressor, eml


ArrayLike = Union[np.ndarray, pd.DataFrame]


# ────────────────────────────────────────────────────────────────────────────
# Family registry
# ────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EMLFamilySpec:
    """Description of one candidate EML-derived model family."""
    family_id: str
    name: str
    description: str
    # Hyperparameters consumed by fit_eml_family.  Only the EMLTreeRegressor
    # ones are used here; a "linear" backbone bypasses random EML features.
    backbone: str  # "eml_tree" or "linear" or "ridge_eml"
    n_features: int = 0
    alpha: float = 1.0
    # Whether to include the operator eml(x_i, |x_j|+1) explicitly as feature
    use_explicit_eml_pair: bool = False


def define_13_eml_families() -> List[EMLFamilySpec]:
    """Return the canonical list of 13 candidate EML-derived model families.

    The set is fixed and ordered.  IDs are stable strings of the form
    ``"F01"..."F13"`` so they can be referenced safely in certificates.

    Family sketch
    -------------

    =====  ======================  ========================================
    ID     Name                    Idea
    =====  ======================  ========================================
    F01    linear_baseline         Plain ridge linear, no EML features
    F02    eml_pairs_small         16 random EML features
    F03    eml_pairs_medium        32 random EML features
    F04    eml_pairs_large         64 random EML features
    F05    eml_pairs_xlarge        128 random EML features
    F06    eml_explicit_pair       Linear + explicit eml(x_i,|x_j|+1) col
    F07    eml_low_alpha           Like F03 but very low ridge penalty
    F08    eml_high_alpha          Like F03 but high ridge penalty
    F09    eml_pairs_seed1         32 features, alt seed for diversity
    F10    eml_pairs_seed2         32 features, second alt seed
    F11    eml_compact_lowreg      16 features, low ridge penalty
    F12    eml_compact_highreg     16 features, high ridge penalty
    F13    eml_dense_balanced      96 features, mid-range ridge penalty
    =====  ======================  ========================================
    """
    return [
        EMLFamilySpec("F01", "linear_baseline",
                      "Ridge linear regression baseline (no EML features).",
                      backbone="linear", n_features=0, alpha=1.0),
        EMLFamilySpec("F02", "eml_pairs_small",
                      "EMLTree with 16 random pairwise EML features.",
                      backbone="eml_tree", n_features=16, alpha=1.0),
        EMLFamilySpec("F03", "eml_pairs_medium",
                      "EMLTree with 32 random pairwise EML features.",
                      backbone="eml_tree", n_features=32, alpha=1.0),
        EMLFamilySpec("F04", "eml_pairs_large",
                      "EMLTree with 64 random pairwise EML features.",
                      backbone="eml_tree", n_features=64, alpha=1.0),
        EMLFamilySpec("F05", "eml_pairs_xlarge",
                      "EMLTree with 128 random pairwise EML features.",
                      backbone="eml_tree", n_features=128, alpha=1.0),
        EMLFamilySpec("F06", "eml_explicit_pair",
                      "Linear + explicit eml(x0,|x1|+1) feature column.",
                      backbone="ridge_eml", n_features=0, alpha=1.0,
                      use_explicit_eml_pair=True),
        EMLFamilySpec("F07", "eml_low_alpha",
                      "EMLTree (32 features) with low ridge penalty.",
                      backbone="eml_tree", n_features=32, alpha=0.1),
        EMLFamilySpec("F08", "eml_high_alpha",
                      "EMLTree (32 features) with high ridge penalty.",
                      backbone="eml_tree", n_features=32, alpha=10.0),
        EMLFamilySpec("F09", "eml_pairs_seed1",
                      "EMLTree (32 features) with alternate random seed (1).",
                      backbone="eml_tree", n_features=32, alpha=1.0),
        EMLFamilySpec("F10", "eml_pairs_seed2",
                      "EMLTree (32 features) with alternate random seed (2).",
                      backbone="eml_tree", n_features=32, alpha=1.0),
        EMLFamilySpec("F11", "eml_compact_lowreg",
                      "EMLTree (16 features) with low ridge penalty.",
                      backbone="eml_tree", n_features=16, alpha=0.1),
        EMLFamilySpec("F12", "eml_compact_highreg",
                      "EMLTree (16 features) with high ridge penalty.",
                      backbone="eml_tree", n_features=16, alpha=10.0),
        EMLFamilySpec("F13", "eml_dense_balanced",
                      "EMLTree (96 features) with moderate ridge penalty.",
                      backbone="eml_tree", n_features=96, alpha=2.0),
    ]


# ────────────────────────────────────────────────────────────────────────────
# Fitted model wrapper
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class FittedEMLFamily:
    """Result of fitting one EML family to (X, y)."""
    spec: EMLFamilySpec
    backbone: str
    estimator: Any  # EMLTreeRegressor or _RidgeLinear
    feature_names: List[str]
    n_rows: int
    train_rmse: float
    effective_n_features: int = 0
    note: str = (
        "Candidate EML-derived payoff generator. Not a new equilibrium concept."
    )


# ────────────────────────────────────────────────────────────────────────────
# Tiny ridge linear backbone (used by F01 and F06)
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class _RidgeLinear:
    alpha: float = 1.0
    use_explicit_eml_pair: bool = False
    coef_: Optional[np.ndarray] = field(default=None, init=False)
    y_mean_: float = field(default=0.0, init=False)
    x_mean_: Optional[np.ndarray] = field(default=None, init=False)
    x_std_: Optional[np.ndarray] = field(default=None, init=False)

    def _design(self, X: np.ndarray) -> np.ndarray:
        cols = [np.ones(X.shape[0])]
        cols.extend([X[:, j] for j in range(X.shape[1])])
        if self.use_explicit_eml_pair and X.shape[1] >= 2:
            cols.append(eml(X[:, 0], np.abs(X[:, 1]) + 1.0))
        Z = np.column_stack(cols)
        return np.nan_to_num(Z, nan=0.0, posinf=1e6, neginf=-1e6)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_RidgeLinear":
        self.x_mean_ = X.mean(axis=0)
        self.x_std_ = X.std(axis=0)
        self.x_std_[self.x_std_ == 0] = 1.0
        Xs = (X - self.x_mean_) / self.x_std_
        Z = self._design(Xs)
        self.y_mean_ = float(y.mean())
        yc = y - self.y_mean_
        penalty = self.alpha * np.eye(Z.shape[1])
        penalty[0, 0] = 0.0
        lhs = Z.T @ Z + penalty
        rhs = Z.T @ yc
        try:
            self.coef_ = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            self.coef_ = np.linalg.pinv(lhs) @ rhs
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.x_mean_ is None or self.x_std_ is None:
            raise RuntimeError("Ridge backbone not fitted.")
        Xs = (X - self.x_mean_) / self.x_std_
        return self.y_mean_ + self._design(Xs) @ self.coef_


# ────────────────────────────────────────────────────────────────────────────
# Fitting
# ────────────────────────────────────────────────────────────────────────────

def _coerce_X(X: ArrayLike) -> Tuple[np.ndarray, List[str]]:
    if isinstance(X, pd.DataFrame):
        names = list(X.columns)
        arr = X.to_numpy(dtype=float)
    else:
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        names = [f"x{j}" for j in range(arr.shape[1])]
    if arr.ndim != 2:
        raise ValueError("X must be 2D.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("X contains NaN/inf.")
    return arr, names


def _coerce_y(y: ArrayLike) -> np.ndarray:
    arr = np.asarray(y, dtype=float).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError("y contains NaN/inf.")
    return arr


def fit_eml_family(
    X: ArrayLike,
    y: ArrayLike,
    family_id: str,
    *,
    base_seed: int = 42,
    max_n_features: Optional[int] = None,
) -> FittedEMLFamily:
    """Fit a single EML family by ID.

    Parameters
    ----------
    X
        ``(n_rows, n_features)`` design matrix (DataFrame or ndarray).
    y
        ``(n_rows,)`` target vector.
    family_id
        One of ``"F01"..."F13"`` (see :func:`define_13_eml_families`).
    base_seed
        Base random seed; per-family seed offsets are added internally to make
        the families F09 and F10 noticeably different.
    max_n_features
        Optional cap for EML random features. This is used by fast real-data
        scoring fixtures to avoid expensive pseudo-inverses while preserving
        the same 13-family identities. ``None`` keeps the family specification
        unchanged.

    Returns
    -------
    FittedEMLFamily
        Fitted estimator with metadata.

    Raises
    ------
    KeyError
        If ``family_id`` is unknown.
    ValueError
        If inputs are non-finite or shapes are inconsistent.
    """
    families = {f.family_id: f for f in define_13_eml_families()}
    if family_id not in families:
        raise KeyError(f"Unknown family_id={family_id!r}; expected one of {list(families)}.")
    spec = families[family_id]

    X_arr, names = _coerce_X(X)
    y_arr = _coerce_y(y)
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("X and y must have the same number of rows.")
    if X_arr.shape[0] < 3:
        raise ValueError("Need at least 3 rows to fit an EML family.")

    # Family-specific seed offset (for diversity in F09/F10).
    seed_offset = {"F09": 1001, "F10": 2002}.get(family_id, 0)
    seed = int(base_seed + seed_offset)

    effective_n_features = 0
    if spec.backbone == "linear" or spec.backbone == "ridge_eml":
        est = _RidgeLinear(alpha=spec.alpha,
                           use_explicit_eml_pair=spec.use_explicit_eml_pair)
        est.fit(X_arr, y_arr)
        pred = est.predict(X_arr)
        effective_n_features = 1 if spec.use_explicit_eml_pair else 0
    else:  # eml_tree
        n_features = spec.n_features
        if max_n_features is not None:
            n_features = min(n_features, int(max_n_features))
        effective_n_features = int(n_features)
        est = EMLTreeRegressor(n_features=n_features,
                               alpha=spec.alpha,
                               seed=seed,
                               standardize=True)
        # EMLTreeRegressor accepts both ndarray and DataFrame; reuse names.
        if isinstance(X, pd.DataFrame):
            est.fit(X, y_arr)
            pred = est.predict(X)
        else:
            est.fit(X_arr, y_arr)
            pred = est.predict(X_arr)

    rmse = float(np.sqrt(np.mean((pred - y_arr) ** 2)))
    return FittedEMLFamily(
        spec=spec,
        backbone=spec.backbone,
        estimator=est,
        feature_names=names,
        n_rows=int(X_arr.shape[0]),
        train_rmse=rmse,
        effective_n_features=effective_n_features,
    )


def fit_all_eml_families(
    X: ArrayLike,
    y: ArrayLike,
    *,
    base_seed: int = 42,
    max_n_features: Optional[int] = None,
) -> Dict[str, FittedEMLFamily]:
    """Fit all 13 EML families and return them keyed by family_id."""
    out: Dict[str, FittedEMLFamily] = {}
    for spec in define_13_eml_families():
        out[spec.family_id] = fit_eml_family(
            X, y, spec.family_id, base_seed=base_seed,
            max_n_features=max_n_features,
        )
    return out


def predict_eml_family(model: FittedEMLFamily, X: ArrayLike) -> np.ndarray:
    """Predict y for new X with a fitted family.

    Parameters
    ----------
    model
        FittedEMLFamily as returned by :func:`fit_eml_family`.
    X
        New design matrix.

    Returns
    -------
    numpy.ndarray
        ``(n_rows,)`` predictions.  Always finite (NaN/inf are clipped to
        large finite values inside the estimator design steps).
    """
    if isinstance(model.estimator, EMLTreeRegressor):
        # Estimator handles DataFrame/ndarray.
        pred = model.estimator.predict(X)
    else:
        X_arr, _ = _coerce_X(X)
        pred = model.estimator.predict(X_arr)
    pred = np.nan_to_num(np.asarray(pred, dtype=float),
                         nan=0.0, posinf=1e6, neginf=-1e6)
    return pred


def rank_eml_families(
    results: Dict[str, FittedEMLFamily],
) -> pd.DataFrame:
    """Rank fitted families by training RMSE (lower is better).

    Parameters
    ----------
    results
        Mapping ``family_id -> FittedEMLFamily``.

    Returns
    -------
    pandas.DataFrame
        Sorted by ``train_rmse`` ascending. Columns include both the
        canonical ``n_features`` from the family specification and the
        ``effective_n_features`` actually used after any runtime cap.
    """
    rows = []
    for fid, m in results.items():
        rows.append({
            "family_id": fid,
            "name": m.spec.name,
            "backbone": m.spec.backbone,
            "n_features": m.spec.n_features,
            "effective_n_features": m.effective_n_features,
            "alpha": m.spec.alpha,
            "train_rmse": m.train_rmse,
            "n_rows": m.n_rows,
        })
    df = pd.DataFrame(rows)
    df = df.sort_values("train_rmse", ascending=True).reset_index(drop=True)
    return df
