"""
phenogame.certificate — End-to-end audit certificate for EML game pipeline.

Bundles the data summary, EML family scores, pairwise preference matrix,
binary stochastic choice checks, Nature scenarios, payoff matrix, minimax
solution, ε-CCE solution, bootstrap stability, sensitivity tipping points,
recommendation, and limitations into one structured certificate.

Public API
----------

- :func:`generate_game_audit_certificate`
- :func:`export_certificate_json`
- :func:`export_certificate_markdown`
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# Default limitations included in every certificate.
DEFAULT_LIMITATIONS: List[str] = [
    "This is a research alpha and not a validated agronomic prescription.",
    "It does not prove a new Nash theorem.",
    "EML trees are used as function generators / payoff learners only.",
    "The binary stochastic choice layer is a diagnostic / model-selection "
    "bridge, not a full rationalisability proof.",
    "Results are conditional on data quality, the supplied feature/target "
    "columns, and the chosen scenario discretisation.",
]


def _to_jsonable(obj: Any) -> Any:
    """Best-effort conversion of common scientific Python objects to JSON."""
    if obj is None:
        return None
    if isinstance(obj, (bool, int, float, str)):
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, np.ndarray):
        return [_to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        # Replace non-finite floats with None for JSON.
        return obj.replace([np.inf, -np.inf], np.nan).where(pd.notna(obj), None).to_dict(orient="split")
    if isinstance(obj, pd.Series):
        return _to_jsonable(obj.to_dict())
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_to_jsonable(x) for x in obj]
    if is_dataclass(obj):
        return _to_jsonable(asdict(obj))
    # Fallback: try string repr.
    try:
        return str(obj)
    except Exception:
        return None


# ────────────────────────────────────────────────────────────────────────────
# Certificate generation
# ────────────────────────────────────────────────────────────────────────────

def generate_game_audit_certificate(result: Any) -> Dict[str, Any]:
    """Build a structured audit certificate dictionary from a result bundle.

    Parameters
    ----------
    result
        Either:

        - An object with the canonical attributes used by this package
          (``compiled_game``, ``family_scores``, ``preference_matrix``,
          ``constraint_audit``, ``minimax``, ``cce``, ``bootstrap``,
          ``sensitivity``, ``recommendation``), or
        - A plain ``dict`` carrying the same keys.

        Missing keys produce ``null`` sections rather than errors, so this
        function is robust against partial pipelines.

    Returns
    -------
    dict
        Certificate with the sections required by the package contract:
        data summary, 13 EML family scores, pairwise preference matrix P,
        binary stochastic choice checks, Nature scenarios, payoff matrix,
        minimax solution, ε-CCE solution, bootstrap stability, sensitivity
        tipping points, recommendation, limitations.
    """

    def get(name: str, default: Any = None) -> Any:
        if isinstance(result, dict):
            return result.get(name, default)
        return getattr(result, name, default)

    compiled = get("compiled_game")
    family_scores = get("family_scores")
    P = get("preference_matrix")
    constraint_audit = get("constraint_audit")
    minimax = get("minimax")
    cce = get("cce")
    bootstrap = get("bootstrap")
    sensitivity = get("sensitivity")
    recommendation = get("recommendation")
    limitations = get("limitations") or DEFAULT_LIMITATIONS

    # Data summary
    data_summary: Dict[str, Any] = {}
    if compiled is not None:
        scen = getattr(compiled, "scenarios", None)
        pm = getattr(compiled, "payoff_matrix", None)
        target = getattr(compiled, "target_column", None)
        data_summary = {
            "target_column": target,
            "n_scenarios": int(len(scen.labels)) if scen is not None else None,
            "scenario_labels": list(scen.labels) if scen is not None else None,
            "feature_columns": list(scen.feature_columns) if scen is not None else None,
            "n_per_scenario": list(scen.n_per_scenario) if scen is not None else None,
            "payoff_matrix_shape": list(pm.matrix.shape) if pm is not None else None,
        }

    # 13 EML family scores
    eml_family_scores: Optional[Any] = None
    if family_scores is not None:
        eml_family_scores = _to_jsonable(family_scores)

    # Pairwise preference matrix
    P_section: Optional[Any] = None
    if P is not None:
        P_section = _to_jsonable(P)

    # Nature scenarios + payoff matrix
    scenarios_section: Optional[Any] = None
    payoff_section: Optional[Any] = None
    if compiled is not None:
        scen = getattr(compiled, "scenarios", None)
        if scen is not None:
            scenarios_section = {
                "labels": list(scen.labels),
                "feature_columns": list(scen.feature_columns),
                "scenario_columns": list(scen.scenario_columns),
                "n_per_scenario": list(scen.n_per_scenario),
                "representatives": _to_jsonable(scen.representatives),
            }
        pm = getattr(compiled, "payoff_matrix", None)
        if pm is not None:
            payoff_section = {
                "shape": list(pm.matrix.shape),
                "strategy_labels": list(pm.strategy_labels),
                "scenario_labels": list(pm.scenario_labels),
                "matrix": _to_jsonable(pm.matrix),
                "metric": pm.metric,
            }

    # Minimax
    minimax_section: Optional[Any] = None
    if minimax is not None:
        minimax_section = {
            "game_value": _to_jsonable(getattr(minimax, "game_value", None)),
            "is_pure": bool(getattr(minimax, "is_pure", False)),
            "dominant_strategy": getattr(minimax, "dominant_strategy", None),
            "strategy_labels": list(getattr(minimax, "strategy_labels", []) or []),
            "scenario_labels": list(getattr(minimax, "scenario_labels", []) or []),
            "farmer_strategy": _to_jsonable(getattr(minimax, "farmer_strategy", None)),
            "nature_strategy": _to_jsonable(getattr(minimax, "nature_strategy", None)),
        }

    # CCE
    cce_section: Optional[Any] = None
    if cce is not None:
        cce_section = {
            "mode": getattr(cce, "mode", None),
            "iterations": int(getattr(cce, "iterations", 0) or 0),
            "cce_gap": _to_jsonable(getattr(cce, "cce_gap", None)),
            "epsilon": _to_jsonable(getattr(cce, "epsilon", None)),
            "is_certified": bool(getattr(cce, "is_certified", False)),
            "max_regret_p1": _to_jsonable(getattr(cce, "max_regret_p1", None)),
            "max_regret_p2": _to_jsonable(getattr(cce, "max_regret_p2", None)),
        }

    # Bootstrap
    bootstrap_section: Optional[Any] = None
    if bootstrap is not None:
        bootstrap_section = _to_jsonable(bootstrap)

    # Sensitivity
    sensitivity_section: Optional[Any] = None
    if sensitivity is not None:
        sensitivity_section = _to_jsonable(sensitivity)

    # Recommendation
    recommendation_section: Optional[Any] = None
    if recommendation is not None:
        recommendation_section = _to_jsonable(recommendation)

    return {
        "schema": "phenogame.certificate.v1",
        "data_summary": data_summary,
        "eml_family_scores": eml_family_scores,
        "preference_matrix": P_section,
        "binary_stochastic_choice_checks": _to_jsonable(constraint_audit),
        "nature_scenarios": scenarios_section,
        "payoff_matrix": payoff_section,
        "minimax_solution": minimax_section,
        "cce_solution": cce_section,
        "bootstrap_stability": bootstrap_section,
        "sensitivity_tipping_points": sensitivity_section,
        "recommendation": recommendation_section,
        "limitations": list(limitations),
    }


def export_certificate_json(certificate: Dict[str, Any], path: str, *, indent: int = 2) -> None:
    """Write a certificate dictionary to JSON.

    Non-finite floats and unsupported objects are coerced via the same
    sanitiser used inside :func:`generate_game_audit_certificate`.
    """
    safe = _to_jsonable(certificate)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(safe, f, indent=indent, default=str)


def export_certificate_markdown(certificate: Dict[str, Any], path: str) -> None:
    """Write a human-readable Markdown summary of the certificate."""
    lines: List[str] = []
    lines.append("# PhenoGame audit certificate")
    lines.append("")
    lines.append(f"_Schema:_ `{certificate.get('schema', 'phenogame.certificate.v1')}`")
    lines.append("")

    # Data summary
    ds = certificate.get("data_summary") or {}
    lines.append("## Data summary")
    lines.append("")
    if ds:
        for k, v in ds.items():
            lines.append(f"- **{k}**: {v}")
    else:
        lines.append("_(no data summary supplied)_")
    lines.append("")

    # Family scores
    lines.append("## 13 EML family scores")
    lines.append("")
    fs = certificate.get("eml_family_scores")
    if fs is None:
        lines.append("_(no family scores supplied)_")
    else:
        # Render either dict-of-records or generic JSON.
        if isinstance(fs, dict) and "data" in fs and "columns" in fs:
            cols = fs.get("columns", [])
            lines.append("| " + " | ".join(str(c) for c in cols) + " |")
            lines.append("|" + "|".join("---" for _ in cols) + "|")
            for row in fs.get("data", []):
                lines.append("| " + " | ".join(str(x) for x in row) + " |")
        else:
            lines.append("```json")
            lines.append(json.dumps(fs, indent=2, default=str)[:4000])
            lines.append("```")
    lines.append("")

    # Constraint audit
    lines.append("## Binary stochastic choice checks")
    lines.append("")
    bsc = certificate.get("binary_stochastic_choice_checks")
    if bsc is None:
        lines.append("_(not run)_")
    else:
        lines.append("```json")
        lines.append(json.dumps(bsc, indent=2, default=str)[:4000])
        lines.append("```")
    lines.append("")

    # Minimax
    lines.append("## Minimax solution (zero-sum, Algorithm vs Nature)")
    lines.append("")
    mm = certificate.get("minimax_solution")
    if mm is None:
        lines.append("_(not run)_")
    else:
        lines.append(f"- **Game value**: {mm.get('game_value')}")
        lines.append(f"- **Pure?**: {mm.get('is_pure')}")
        lines.append(f"- **Dominant strategy**: {mm.get('dominant_strategy')}")
    lines.append("")

    # CCE
    lines.append("## ε-CCE solution")
    lines.append("")
    cs = certificate.get("cce_solution")
    if cs is None:
        lines.append("_(not run)_")
    else:
        lines.append(f"- **Mode**: {cs.get('mode')}")
        lines.append(f"- **Iterations**: {cs.get('iterations')}")
        lines.append(f"- **CCE gap**: {cs.get('cce_gap')}")
        lines.append(f"- **ε bound**: {cs.get('epsilon')}")
        lines.append(f"- **Certified**: {cs.get('is_certified')}")
    lines.append("")

    # Bootstrap & sensitivity
    lines.append("## Bootstrap stability")
    lines.append("")
    bs = certificate.get("bootstrap_stability")
    lines.append("_(no bootstrap result)_" if bs is None else "```json")
    if bs is not None:
        lines.append(json.dumps(bs, indent=2, default=str)[:4000])
        lines.append("```")
    lines.append("")

    lines.append("## Sensitivity tipping points")
    lines.append("")
    ss = certificate.get("sensitivity_tipping_points")
    lines.append("_(no sensitivity result)_" if ss is None else "```json")
    if ss is not None:
        lines.append(json.dumps(ss, indent=2, default=str)[:4000])
        lines.append("```")
    lines.append("")

    # Recommendation
    lines.append("## Recommendation")
    lines.append("")
    rec = certificate.get("recommendation")
    if rec is None:
        lines.append("_(no recommendation supplied)_")
    elif isinstance(rec, dict):
        for k, v in rec.items():
            lines.append(f"- **{k}**: {v}")
    else:
        lines.append(str(rec))
    lines.append("")

    # Limitations
    lines.append("## Limitations")
    lines.append("")
    for item in certificate.get("limitations", []):
        lines.append(f"- {item}")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
