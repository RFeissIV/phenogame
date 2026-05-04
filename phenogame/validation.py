"""Train/test validation and baseline model comparison.

This module addresses two audit recommendations:

* Compare EMLTreeRegressor against simple baselines (naive mean, ridge linear,
  optional random forest if scikit-learn is available).
* Provide held-out evaluation metrics (RMSE, MAE, R^2) and a measure of
  *action stability* — how often the EML-induced game's recommended strategy
  agrees with itself across bootstrap-style resamples of the training data.

Scope boundary (kept conservative on purpose):

* These are *empirical* model-quality checks on tabular data. They are not
  agronomic field validations.
* Random forest is loaded lazily so the package keeps a small dependency
  footprint. If scikit-learn is not installed the RF row is simply skipped.
* Action stability is a frequency, not a probability bound.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .eml_tree import EMLTreeRegressor


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(diff * diff)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(diff)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 1e-18:
        # Degenerate target (constant). Define R^2 as 0.0 to be safe.
        return 0.0
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# Baseline regressors (intentionally tiny; no sklearn requirement)
# ---------------------------------------------------------------------------

@dataclass
class _NaiveMeanRegressor:
    mean_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_NaiveMeanRegressor":
        self.mean_ = float(np.mean(np.asarray(y, dtype=float)))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        n = np.asarray(X).shape[0]
        return np.full(n, self.mean_, dtype=float)


@dataclass
class _RidgeLinearRegressor:
    """Closed-form ridge regression on standardized features."""
    alpha: float = 1.0
    coef_: Optional[np.ndarray] = field(default=None, init=False)
    intercept_: float = field(default=0.0, init=False)
    x_mean_: Optional[np.ndarray] = field(default=None, init=False)
    x_std_: Optional[np.ndarray] = field(default=None, init=False)
    y_mean_: float = field(default=0.0, init=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_RidgeLinearRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.x_mean_ = X.mean(axis=0)
        self.x_std_ = X.std(axis=0)
        self.x_std_[self.x_std_ == 0] = 1.0
        Xs = (X - self.x_mean_) / self.x_std_
        self.y_mean_ = float(y.mean())
        yc = y - self.y_mean_
        n_features = Xs.shape[1]
        A = Xs.T @ Xs + self.alpha * np.eye(n_features)
        self.coef_ = np.linalg.pinv(A) @ Xs.T @ yc
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Ridge model not fitted.")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        Xs = (X - self.x_mean_) / self.x_std_
        return self.y_mean_ + Xs @ self.coef_


def _maybe_random_forest(seed: int = 42) -> Optional[Any]:
    """Return a sklearn RandomForestRegressor if available, else None."""
    try:  # pragma: no cover - optional dependency
        from sklearn.ensemble import RandomForestRegressor
    except Exception:  # pragma: no cover
        return None
    return RandomForestRegressor(
        n_estimators=100, max_depth=None, random_state=seed, n_jobs=1
    )


# ---------------------------------------------------------------------------
# Train/test splitting
# ---------------------------------------------------------------------------

def train_test_split_indices(
    n: int, *, test_fraction: float = 0.25, seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic random train/test index split.

    Returns
    -------
    (train_idx, test_idx) : np.ndarray, np.ndarray
        Disjoint integer arrays whose union is ``range(n)``.
    """
    if n < 4:
        raise ValueError("Need at least 4 rows to form a meaningful train/test split.")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1).")
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    n_test = max(1, int(round(test_fraction * n)))
    if n_test >= n:
        n_test = n - 1
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    return np.sort(train_idx), np.sort(test_idx)


# ---------------------------------------------------------------------------
# Multi-model comparison
# ---------------------------------------------------------------------------

@dataclass
class ModelScore:
    """Held-out metrics for a single model."""
    name: str
    n_train: int
    n_test: int
    train_rmse: float
    test_rmse: float
    test_mae: float
    test_r2: float
    note: str = ""

    def as_row(self) -> Dict[str, Any]:
        return {
            "model": self.name,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "train_rmse": self.train_rmse,
            "test_rmse": self.test_rmse,
            "test_mae": self.test_mae,
            "test_r2": self.test_r2,
            "note": self.note,
        }


@dataclass
class BaselineComparison:
    """Result of comparing EMLTreeRegressor against simple baselines."""
    target_column: str
    feature_columns: List[str]
    scores: List[ModelScore]
    best_by: str = "test_rmse"

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([s.as_row() for s in self.scores])

    def best(self) -> ModelScore:
        if not self.scores:
            raise RuntimeError("No models scored.")
        return min(self.scores, key=lambda s: getattr(s, self.best_by))

    def summary(self) -> str:
        lines = [
            "Baseline comparison",
            f"  target: {self.target_column}",
            f"  features: {', '.join(self.feature_columns)}",
            "  scores (lower test_rmse is better):",
        ]
        for s in self.scores:
            lines.append(
                f"    {s.name:18s}  train_rmse={s.train_rmse:.4f}  "
                f"test_rmse={s.test_rmse:.4f}  test_mae={s.test_mae:.4f}  "
                f"test_r2={s.test_r2:.4f}"
            )
        try:
            lines.append(f"  best by {self.best_by}: {self.best().name}")
        except RuntimeError:
            pass
        return "\n".join(lines)


def compare_baselines(
    data: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    *,
    test_fraction: float = 0.25,
    seed: int = 42,
    n_eml_features: int = 64,
    eml_alpha: float = 1.0,
    ridge_alpha: float = 1.0,
    include_random_forest: bool = True,
) -> BaselineComparison:
    """Compare EMLTreeRegressor against simple baselines on a single split.

    Models scored:

    * ``naive_mean`` — predicts the training mean (sanity floor).
    * ``ridge_linear`` — linear regression with L2 penalty.
    * ``eml_tree_ridge`` — EMLTreeRegressor (this package's model).
    * ``random_forest`` — sklearn RandomForestRegressor (if available).

    All four are trained on the same train/test split and scored with the same
    metrics so that the comparison is honest.
    """
    cols = [*feature_columns, target_column]
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    clean = data[cols].dropna().copy()
    if len(clean) < 4:
        raise ValueError("Need at least 4 rows after dropping NA to compare baselines.")

    train_idx, test_idx = train_test_split_indices(
        len(clean), test_fraction=test_fraction, seed=seed
    )
    X = clean[list(feature_columns)].to_numpy(dtype=float)
    y = clean[target_column].to_numpy(dtype=float)
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    scores: List[ModelScore] = []

    def _record(name: str, model: Any, note: str) -> None:
        pred_tr = np.asarray(model.predict(X_tr), dtype=float)
        pred_te = np.asarray(model.predict(X_te), dtype=float)
        scores.append(ModelScore(
            name=name,
            n_train=len(train_idx), n_test=len(test_idx),
            train_rmse=_rmse(y_tr, pred_tr),
            test_rmse=_rmse(y_te, pred_te),
            test_mae=_mae(y_te, pred_te),
            test_r2=_r2(y_te, pred_te),
            note=note,
        ))

    # 1. naive mean
    nm = _NaiveMeanRegressor().fit(X_tr, y_tr)
    _record("naive_mean", nm, "constant prediction")

    # 2. ridge linear
    rl = _RidgeLinearRegressor(alpha=ridge_alpha).fit(X_tr, y_tr)
    _record("ridge_linear", rl, f"ridge alpha={ridge_alpha}")

    # 3. EML-tree ridge
    eml = EMLTreeRegressor(n_features=n_eml_features, alpha=eml_alpha, seed=seed)
    eml.fit(X_tr, y_tr)
    _record("eml_tree_ridge", eml, f"n_features={n_eml_features}, alpha={eml_alpha}")

    # 4. optional random forest
    if include_random_forest:
        rf = _maybe_random_forest(seed=seed)
        if rf is not None:
            rf.fit(X_tr, y_tr)
            _record("random_forest", rf, "sklearn RandomForestRegressor")

    return BaselineComparison(
        target_column=target_column,
        feature_columns=list(feature_columns),
        scores=scores,
    )


# ---------------------------------------------------------------------------
# Action stability
# ---------------------------------------------------------------------------

@dataclass
class ActionStabilityResult:
    """How often the data-induced game's recommendation is unchanged across resamples."""
    n_resamples: int
    base_action: str
    action_counts: Dict[str, int]
    stability: float            # frequency of the base action
    most_common_action: str     # most frequently recommended across resamples
    most_common_frequency: float

    def summary(self) -> str:
        lines = [
            "Action stability under bootstrap resampling",
            f"  resamples: {self.n_resamples}",
            f"  base recommendation: {self.base_action}",
            f"  base-action stability: {self.stability:.3f}",
            f"  most common action: {self.most_common_action} "
            f"(freq={self.most_common_frequency:.3f})",
            "  action counts:",
        ]
        for name, count in sorted(self.action_counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {name}: {count}")
        return "\n".join(lines)


def action_stability(
    data: pd.DataFrame,
    *,
    strategy_column: str,
    scenario_column: str,
    feature_columns: Sequence[str],
    target_column: str,
    n_resamples: int = 20,
    seed: int = 42,
    n_eml_features: int = 32,
    alpha: float = 1.0,
    iterations: int = 50,
    n_strategy_bins: int = 3,
    n_scenario_bins: int = 3,
    imputation: str = "row_min",
) -> ActionStabilityResult:
    """Bootstrap the data, recompile the EML-induced game, and track recommendations.

    The recommendation rule is the **largest player-1 marginal of the
    empirical joint** returned by Hedge no-regret dynamics on the data-induced
    payoff matrix. This is the action chosen most often by the no-regret
    learner in expectation.

    Returned ``stability`` is the empirical frequency that bootstrap resamples
    return the same recommendation as the full-data baseline. It is a
    descriptive statistic, **not** a coverage probability or confidence bound.
    Treat it as a robustness indicator only.
    """
    # Local imports to avoid module-level cycles.
    from .data_compiler import compile_eml_payoff_game
    from .equilibrium import certify_provable

    if n_resamples < 2:
        raise ValueError("n_resamples must be >= 2.")
    cols = list({strategy_column, scenario_column, target_column, *feature_columns})
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    clean = data.dropna(subset=cols).reset_index(drop=True)
    if len(clean) < 4:
        raise ValueError("Need at least 4 complete rows for action stability.")

    def _recommend(df: pd.DataFrame, sub_seed: int) -> str:
        game = compile_eml_payoff_game(
            df,
            strategy_column=strategy_column,
            scenario_column=scenario_column,
            feature_columns=list(feature_columns),
            target_column=target_column,
            n_eml_features=n_eml_features,
            alpha=alpha,
            seed=sub_seed,
            n_strategy_bins=n_strategy_bins,
            n_scenario_bins=n_scenario_bins,
            imputation=imputation,
        )
        cert = certify_provable(game.payoff_matrix, iterations=iterations, seed=sub_seed)
        # Derive the player-1 marginal from the empirical joint and pick the
        # strategy with largest mass.  This is the no-regret-dynamics analogue
        # of "what action would I play most often?".
        joint = np.asarray(cert.empirical_joint, dtype=float)
        if joint.ndim != 2 or joint.size == 0:
            return "ERROR"
        marg = joint.sum(axis=1)
        idx = int(np.argmax(marg))
        labels = cert.strategy_labels or game.payoff_matrix.strategy_labels
        if 0 <= idx < len(labels):
            return str(labels[idx])
        return f"strategy_{idx}"

    # Base recommendation on the full data.
    base_action = _recommend(clean, sub_seed=int(seed))

    rng = np.random.default_rng(int(seed))
    counts: Dict[str, int] = {}
    n = len(clean)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boot = clean.iloc[idx].reset_index(drop=True)
        try:
            action = _recommend(boot, sub_seed=int(seed) + 1 + i)
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            action = "ERROR"
        counts[action] = counts.get(action, 0) + 1

    most_common_action, most_common_count = max(counts.items(), key=lambda kv: kv[1])
    stability = counts.get(base_action, 0) / float(n_resamples)
    return ActionStabilityResult(
        n_resamples=n_resamples,
        base_action=base_action,
        action_counts=counts,
        stability=float(stability),
        most_common_action=most_common_action,
        most_common_frequency=float(most_common_count) / float(n_resamples),
    )


__all__ = [
    "train_test_split_indices",
    "ModelScore",
    "BaselineComparison",
    "compare_baselines",
    "ActionStabilityResult",
    "action_stability",
]
