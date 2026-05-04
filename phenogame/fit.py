"""
phenogame.fit — Calibrate response functions from field observations.

Fits all 13 response function templates to user data via scipy.optimize.curve_fit,
ranks by AIC/BIC, and returns calibrated parameters with EML metadata.
"""

import numpy as np
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from scipy.optimize import curve_fit
from .response import RESPONSE_FUNCTIONS, ResponseFunction


@dataclass
class FitResult:
    """Result of fitting a single response function to data."""
    func_id: str
    func_name: str
    params: Dict[str, float]
    residuals: np.ndarray
    mse: float
    rmse: float
    r_squared: float
    aic: float
    bic: float
    eml_K: int
    eml_ops: List[str]
    eml_equation: str
    converged: bool
    equation_with_params: str


@dataclass
class FitReport:
    """Complete report from fitting all candidate functions."""
    results: List[FitResult]  # sorted by AIC
    best: FitResult
    x_data: np.ndarray
    y_data: np.ndarray
    n_converged: int
    n_total: int

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"PhenoGame Fit Report: {self.n_converged}/{self.n_total} models converged",
            f"Best model: {self.best.func_name}",
            f"  Equation: {self.best.eml_equation}",
            f"  Parameters: {self.best.params}",
            f"  R² = {self.best.r_squared:.6f}, RMSE = {self.best.rmse:.6f}, AIC = {self.best.aic:.2f}",
            f"  EML complexity: K ≤ {self.best.eml_K}",
            "",
            "Rankings (by AIC):",
        ]
        for i, r in enumerate(self.results[:8]):
            flag = " ***" if r.func_id == self.best.func_id else ""
            lines.append(
                f"  {i+1}. {r.func_name:30s} AIC={r.aic:8.2f}  R²={r.r_squared:.4f}  K≤{r.eml_K}{flag}"
            )
        return "\n".join(lines)


def _make_wrapper(rf: ResponseFunction):
    """Create a curve_fit-compatible wrapper: f(x, p0, p1, ...) -> y."""
    names = rf.param_names

    def wrapper(x, *args):
        kwargs = dict(zip(names, args))
        return rf.fn(x, **kwargs)

    return wrapper


def _aic(n: int, k: int, mse: float) -> float:
    """Akaike Information Criterion."""
    if mse <= 0:
        return -np.inf
    return n * np.log(mse) + 2 * k


def _bic(n: int, k: int, mse: float) -> float:
    """Bayesian Information Criterion."""
    if mse <= 0:
        return -np.inf
    return n * np.log(mse) + k * np.log(n)


def _format_equation(rf: ResponseFunction, params: Dict[str, float]) -> str:
    """Format equation with fitted parameter values substituted."""
    eq = rf.equation
    for name, val in params.items():
        # Replace parameter names with values in the equation string
        eq = eq + f"  [{name}={val:.4g}]"
    return eq


def fit_single(
    x: np.ndarray,
    y: np.ndarray,
    func_id: str,
    p0: Optional[Dict[str, float]] = None,
    maxfev: int = 5000,
) -> Optional[FitResult]:
    """Fit a single response function to data.

    Parameters
    ----------
    x : array-like
        Independent variable (temperature, GDD, photoperiod, etc.).
    y : array-like
        Observed response values.
    func_id : str
        ID of the response function to fit.
    p0 : dict, optional
        Initial parameter guesses. Uses defaults if not provided.
    maxfev : int
        Maximum function evaluations for optimizer.

    Returns
    -------
    FitResult or None if fitting fails.
    """
    rf = RESPONSE_FUNCTIONS[func_id]
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    # Initial guesses
    if p0 is None:
        p0_vals = [rf.param_defaults[name] for name in rf.param_names]
    else:
        p0_vals = [p0.get(name, rf.param_defaults[name]) for name in rf.param_names]

    # Bounds
    lower = [rf.param_bounds[name][0] for name in rf.param_names]
    upper = [rf.param_bounds[name][1] for name in rf.param_names]

    wrapper = _make_wrapper(rf)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(
                wrapper, x, y, p0=p0_vals,
                bounds=(lower, upper), maxfev=maxfev,
                method="trf"
            )

        fitted_params = dict(zip(rf.param_names, popt))
        y_pred = rf.fn(x, **fitted_params)
        residuals = y - y_pred
        mse = float(np.mean(residuals ** 2))
        rmse = float(np.sqrt(mse))
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        k = len(rf.param_names)

        return FitResult(
            func_id=rf.id,
            func_name=rf.name,
            params=fitted_params,
            residuals=residuals,
            mse=mse,
            rmse=rmse,
            r_squared=r2,
            aic=_aic(n, k, mse),
            bic=_bic(n, k, mse),
            eml_K=rf.eml_K,
            eml_ops=rf.eml_ops,
            eml_equation=rf.equation,
            converged=True,
            equation_with_params=_format_equation(rf, fitted_params),
        )

    except (RuntimeError, ValueError, TypeError):
        return None


def fit_all(
    x: np.ndarray,
    y: np.ndarray,
    func_ids: Optional[List[str]] = None,
    maxfev: int = 5000,
) -> FitReport:
    """Fit all (or selected) response functions to data and rank by AIC.

    Parameters
    ----------
    x : array-like
        Independent variable.
    y : array-like
        Observed response values.
    func_ids : list of str, optional
        Subset of function IDs to try. Defaults to all 13.
    maxfev : int
        Maximum function evaluations per fit.

    Returns
    -------
    FitReport with ranked results.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if func_ids is None:
        func_ids = list(RESPONSE_FUNCTIONS.keys())

    results = []
    for fid in func_ids:
        result = fit_single(x, y, fid, maxfev=maxfev)
        if result is not None:
            results.append(result)

    # Sort by AIC (lower is better)
    results.sort(key=lambda r: r.aic)

    if not results:
        raise RuntimeError("No response function converged. Check your data ranges.")

    return FitReport(
        results=results,
        best=results[0],
        x_data=x,
        y_data=y,
        n_converged=len(results),
        n_total=len(func_ids),
    )
