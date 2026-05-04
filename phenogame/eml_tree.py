"""EML-tree payoff learning.

This module makes the EML component operational without overclaiming.  It uses
random, shallow EML-composition features as a nonlinear basis and fits a ridge
linear model on top.  The result is a data-induced payoff/response model that
can be used upstream of finite-game construction.

Important scope boundary: this is a practical function class, not a new
no-regret or equilibrium theorem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def eml(x: np.ndarray, y: np.ndarray, *, clip: float = 30.0) -> np.ndarray:
    """Numerically stable EML operator ``exp(x) - ln(y)``.

    Inputs are clipped/shifted to keep real-valued evaluation finite.  This is a
    modeling implementation choice for noisy tabular data, not an algebraic
    identity claim over all real/complex inputs.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    safe_x = np.clip(x_arr, -clip, clip)
    safe_y = np.maximum(np.abs(y_arr), 1e-12)
    return np.exp(safe_x) - np.log(safe_y)


@dataclass
class EMLTreeFeature:
    """One randomly generated shallow EML feature."""

    left_col: int
    right_col: int
    left_scale: float
    right_scale: float
    bias: float = 1.0

    def transform(self, X: np.ndarray) -> np.ndarray:
        left = self.left_scale * X[:, self.left_col]
        right = np.abs(self.right_scale * X[:, self.right_col]) + self.bias
        return eml(left, right)

    def expression(self, feature_names: List[str]) -> str:
        return (
            f"eml({self.left_scale:.3g}*{feature_names[self.left_col]}, "
            f"abs({self.right_scale:.3g}*{feature_names[self.right_col]})+{self.bias:.3g})"
        )


@dataclass
class EMLTreeRegressor:
    """Ridge regressor over EML-tree random features.

    Parameters
    ----------
    n_features:
        Number of EML-tree features to generate.
    alpha:
        Ridge penalty.  Larger values reduce variance.
    seed:
        Reproducible random seed.
    standardize:
        Whether to standardize input columns before feature generation.
    """

    n_features: int = 64
    alpha: float = 1.0
    seed: int = 42
    standardize: bool = True
    feature_names_: List[str] = field(default_factory=list, init=False)
    features_: List[EMLTreeFeature] = field(default_factory=list, init=False)
    coef_: Optional[np.ndarray] = field(default=None, init=False)
    x_mean_: Optional[np.ndarray] = field(default=None, init=False)
    x_std_: Optional[np.ndarray] = field(default=None, init=False)
    y_mean_: float = field(default=0.0, init=False)

    def _as_array(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            if not self.feature_names_:
                self.feature_names_ = list(X.columns)
            arr = X.to_numpy(dtype=float)
        else:
            arr = np.asarray(X, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            if not self.feature_names_:
                self.feature_names_ = [f"x{i}" for i in range(arr.shape[1])]
        if arr.ndim != 2:
            raise ValueError("X must be a 2D array or DataFrame.")
        if not np.all(np.isfinite(arr)):
            raise ValueError("X contains non-finite values.")
        return arr

    def _scale(self, X: np.ndarray, fit: bool) -> np.ndarray:
        if not self.standardize:
            return X
        if fit:
            self.x_mean_ = X.mean(axis=0)
            self.x_std_ = X.std(axis=0)
            self.x_std_[self.x_std_ == 0] = 1.0
        if self.x_mean_ is None or self.x_std_ is None:
            raise RuntimeError("Model has not been fitted.")
        return (X - self.x_mean_) / self.x_std_

    def _design(self, Xs: np.ndarray) -> np.ndarray:
        cols = [np.ones(Xs.shape[0])]
        cols.extend([Xs[:, j] for j in range(Xs.shape[1])])
        cols.extend([feat.transform(Xs) for feat in self.features_])
        Z = np.column_stack(cols)
        return np.nan_to_num(Z, nan=0.0, posinf=1e6, neginf=-1e6)

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "EMLTreeRegressor":
        X_arr = self._as_array(X)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have the same number of rows.")
        if X_arr.shape[0] < 3:
            raise ValueError("Need at least 3 rows to fit EMLTreeRegressor.")
        if not np.all(np.isfinite(y_arr)):
            raise ValueError("y contains non-finite values.")

        Xs = self._scale(X_arr, fit=True)
        rng = np.random.default_rng(self.seed)
        n_cols = Xs.shape[1]
        self.features_ = [
            EMLTreeFeature(
                left_col=int(rng.integers(0, n_cols)),
                right_col=int(rng.integers(0, n_cols)),
                left_scale=float(rng.normal(0.0, 0.75)),
                right_scale=float(rng.normal(0.0, 0.75)),
                bias=1.0,
            )
            for _ in range(self.n_features)
        ]
        Z = self._design(Xs)
        self.y_mean_ = float(y_arr.mean())
        yc = y_arr - self.y_mean_
        penalty = self.alpha * np.eye(Z.shape[1])
        penalty[0, 0] = 0.0
        lhs = Z.T @ Z + penalty
        rhs = Z.T @ yc
        try:
            self.coef_ = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            self.coef_ = np.linalg.pinv(lhs) @ rhs
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model has not been fitted.")
        X_arr = self._as_array(X)
        Xs = self._scale(X_arr, fit=False)
        return self.y_mean_ + self._design(Xs) @ self.coef_

    def score_rmse(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> float:
        pred = self.predict(X)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        return float(np.sqrt(np.mean((pred - y_arr) ** 2)))

    def expressions(self, max_terms: int = 10) -> List[str]:
        names = self.feature_names_ or []
        return [f.expression(names) for f in self.features_[:max_terms]]


@dataclass
class EMLPayoffFit:
    """Fitted EML-tree payoff model plus audit metadata."""

    model: EMLTreeRegressor
    feature_columns: List[str]
    target_column: str
    n_rows: int
    rmse: float
    note: str = (
        "EML-tree features learn a payoff/response surface used upstream of "
        "finite-game construction; this is not a new equilibrium proof."
    )

    def summary(self) -> str:
        lines = [
            "EML-tree payoff fit",
            f"  rows: {self.n_rows}",
            f"  features: {', '.join(self.feature_columns)}",
            f"  target: {self.target_column}",
            f"  training RMSE: {self.rmse:.6f}",
            f"  generated EML features: {len(self.model.features_)}",
            f"  note: {self.note}",
        ]
        return "\n".join(lines)


def fit_eml_payoff_model(
    data: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    *,
    n_features: int = 64,
    alpha: float = 1.0,
    seed: int = 42,
) -> EMLPayoffFit:
    """Fit an EML-tree payoff/response model from tabular data."""
    missing = [c for c in [*feature_columns, target_column] if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    clean = data[[*feature_columns, target_column]].dropna().copy()
    if len(clean) < 3:
        raise ValueError("Need at least 3 complete rows to fit an EML payoff model.")
    model = EMLTreeRegressor(n_features=n_features, alpha=alpha, seed=seed)
    model.fit(clean[feature_columns], clean[target_column])
    return EMLPayoffFit(
        model=model,
        feature_columns=list(feature_columns),
        target_column=target_column,
        n_rows=len(clean),
        rmse=model.score_rmse(clean[feature_columns], clean[target_column]),
    )


# ────────────────────────────────────────────────────────────────────────────
# Simple binary EML tree representation (for EML-tree fitting on top of
# tabular data).  This is intentionally lightweight: a binary tree of
# eml(left, right) nodes with leaves equal to feature columns or constants.
#
# It coexists with EMLTreeRegressor, which is a flat random-feature model.
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class EMLNode:
    """One node in a simple binary EML expression tree.

    A node is either:

    - a *leaf*: ``kind == "feature"`` (with ``feature_index``) or
      ``kind == "constant"`` (with ``constant``),
    - an *internal node*: ``kind == "eml"`` with two children. The internal
      node represents ``eml(left.evaluate(X), right.evaluate(X))``.

    Numerical safety follows :func:`eml`: inputs are clipped, the right
    operand is taken in absolute value plus a small bias, and ``log`` is
    only ever applied to strictly positive numbers.
    """

    kind: str  # "feature" | "constant" | "eml"
    feature_index: Optional[int] = None
    constant: float = 0.0
    left: Optional["EMLNode"] = None
    right: Optional["EMLNode"] = None
    bias: float = 1.0

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the (sub)tree on ``X`` and return a 1D array.

        Parameters
        ----------
        X
            ``(n_rows, n_features)`` ndarray.
        """
        if self.kind == "feature":
            if self.feature_index is None:
                raise ValueError("feature node missing feature_index.")
            col = X[:, self.feature_index]
            return np.asarray(col, dtype=float)
        if self.kind == "constant":
            return np.full(X.shape[0], float(self.constant))
        if self.kind == "eml":
            if self.left is None or self.right is None:
                raise ValueError("eml node missing children.")
            left = self.left.evaluate(X)
            right = self.right.evaluate(X)
            right_safe = np.abs(right) + float(self.bias)
            return eml(left, right_safe)
        raise ValueError(f"Unknown EMLNode kind: {self.kind!r}.")

    def expression(self, feature_names: Optional[List[str]] = None) -> str:
        """Pretty-print the node as a small expression string."""
        if self.kind == "feature":
            if feature_names is not None and self.feature_index is not None:
                return feature_names[self.feature_index]
            return f"x{self.feature_index}"
        if self.kind == "constant":
            return f"{self.constant:.3g}"
        if self.kind == "eml":
            assert self.left is not None and self.right is not None
            return (f"eml({self.left.expression(feature_names)}, "
                    f"abs({self.right.expression(feature_names)})+{self.bias:.3g})")
        return "?"


@dataclass
class EMLTree:
    """A small binary EML expression tree with a linear readout.

    Multiple :class:`EMLNode` subtrees are evaluated as features and combined
    with a learned ridge regression on top.  This object is the binary-tree
    counterpart to :class:`EMLTreeRegressor` and is intended for inspection
    of the learned structure rather than maximum predictive performance.
    """

    nodes: List[EMLNode]
    feature_names: List[str] = field(default_factory=list)
    coef_: Optional[np.ndarray] = field(default=None)
    intercept_: float = field(default=0.0)
    alpha: float = 1.0

    def _design(self, X: np.ndarray) -> np.ndarray:
        cols = [np.ones(X.shape[0])]
        for node in self.nodes:
            cols.append(node.evaluate(X))
        Z = np.column_stack(cols)
        return np.nan_to_num(Z, nan=0.0, posinf=1e6, neginf=-1e6)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EMLTree":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        if not np.all(np.isfinite(X_arr)):
            raise ValueError("X contains NaN/inf.")
        if not np.all(np.isfinite(y_arr)):
            raise ValueError("y contains NaN/inf.")
        Z = self._design(X_arr)
        penalty = self.alpha * np.eye(Z.shape[1])
        penalty[0, 0] = 0.0
        ymean = float(y_arr.mean())
        yc = y_arr - ymean
        lhs = Z.T @ Z + penalty
        rhs = Z.T @ yc
        try:
            beta = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(lhs) @ rhs
        self.coef_ = beta
        self.intercept_ = ymean
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("EMLTree not fitted.")
        X_arr = np.asarray(X, dtype=float)
        if not np.all(np.isfinite(X_arr)):
            raise ValueError("X contains NaN/inf.")
        Z = self._design(X_arr)
        out = self.intercept_ + Z @ self.coef_
        return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)

    def expressions(self) -> List[str]:
        return [n.expression(self.feature_names) for n in self.nodes]


def _build_random_eml_node(
    rng: np.random.Generator, n_cols: int, depth: int,
) -> EMLNode:
    """Build a small random binary EML node up to a max depth."""
    if depth == 0 or rng.random() < 0.5:
        # Leaf: either a feature reference or a small constant.
        if rng.random() < 0.85 and n_cols > 0:
            return EMLNode(kind="feature",
                           feature_index=int(rng.integers(0, n_cols)))
        return EMLNode(kind="constant", constant=float(rng.normal(0.0, 0.5)))
    left = _build_random_eml_node(rng, n_cols, depth - 1)
    right = _build_random_eml_node(rng, n_cols, depth - 1)
    return EMLNode(kind="eml", left=left, right=right, bias=1.0)


def fit_eml_tree(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    *,
    n_nodes: int = 16,
    max_depth: int = 2,
    alpha: float = 1.0,
    seed: int = 42,
) -> EMLTree:
    """Fit a small ensemble of random binary EML expression trees.

    Parameters
    ----------
    X
        ``(n_rows, n_features)`` design matrix.
    y
        ``(n_rows,)`` target.
    n_nodes
        Number of random EML subtrees to draw.
    max_depth
        Maximum depth of each subtree (1 = a single eml() of two leaves).
    alpha
        Ridge penalty for the linear readout.
    seed
        Random seed.

    Returns
    -------
    EMLTree
        Fitted tree with a learned linear readout.
    """
    if isinstance(X, pd.DataFrame):
        feature_names = list(X.columns)
        X_arr = X.to_numpy(dtype=float)
    else:
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        feature_names = [f"x{j}" for j in range(X_arr.shape[1])]
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("X and y must have same number of rows.")
    if X_arr.shape[0] < 3:
        raise ValueError("Need at least 3 rows to fit an EMLTree.")

    rng = np.random.default_rng(seed)
    n_cols = X_arr.shape[1]
    nodes = [_build_random_eml_node(rng, n_cols, max_depth) for _ in range(n_nodes)]
    tree = EMLTree(nodes=nodes, feature_names=feature_names, alpha=alpha)
    tree.fit(X_arr, y_arr)
    return tree


def evaluate_eml_tree(
    tree: EMLTree,
    X: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    """Evaluate a fitted :class:`EMLTree` on new inputs."""
    if isinstance(X, pd.DataFrame):
        X_arr = X.to_numpy(dtype=float)
    else:
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
    return tree.predict(X_arr)
