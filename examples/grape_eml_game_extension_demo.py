"""Grape data-induced EML game extension demo for PhenoGame 0.3.0.

End-to-end example exercised by the audit checklist:

1. Load USA-NPN grape data.
2. Construct a transparent, formula-driven payoff target on Ripe fruits.
3. Fit the EML-tree payoff model and compare it to baselines on a held-out split.
4. Compile a finite EML-induced decision game and certify epsilon-CCE.
5. Bootstrap the recommendation to estimate action stability.
6. Run the standard PhenoGame robustness report.
7. Run counterfactual phenology-timing tests (earlier/later observed DOY).
8. Build a machine-readable game audit certificate (JSON-LD compatible).
9. Save three artifacts to ``outputs/``:

   * ``grape_decision_certificate.json`` — full audit certificate.
   * ``grape_robustness_summary.csv`` — bootstrap + sensitivity summary.
   * ``grape_counterfactual_table.csv`` — recommendations vs DOY offset.

Scope boundary: this is a research demo on bundled phenology data with a
transparent surrogate payoff. It is NOT a validated grower recommendation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from phenogame import (
    __version__,
    action_stability,
    build_game_audit_certificate,
    certify_provable,
    compare_baselines,
    compile_eml_payoff_game,
    counterfactual_timing_test,
    load_npn_csv,
    robustness_report,
    run_phenology_game,
)

PHENOPHASE = "Ripe fruits"
TARGET_DOY = 200.0  # transparent target window for Ripe fruits (mid-July)
PAYOFF_FORMULA = "surrogate utility: -|DOY - target_doy| - 0.001*|AGDD - mean(AGDD)|"


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy/pandas/dataclass-like values into JSON-safe types."""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(obj, np.ndarray):
        return [_to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    # Fall back to repr for opaque objects (e.g. nested dataclasses we don't
    # explicitly know how to flatten).  This keeps the JSON file usable but
    # avoids losing information silently.
    return repr(obj)


def _build_surrogate_utility(df: pd.DataFrame, target_doy: float) -> pd.Series:
    """Transparent surrogate utility derived from real NPN observations.

    payoff = -|DOY - target| - 0.001 * |AGDD - mean(AGDD)|

    This is a closed-form mapping so reviewers can read it directly. It is NOT
    a validated agronomic value model. The grape demo uses this surrogate so
    that the EML-tree -> game -> certificate pipeline can run end to end on
    bundled open data without inventing an undocumented ground truth.
    """
    doy = df["Mean_First_Yes_DOY"].astype(float)
    agdd = df["Mean_AGDD"].astype(float)
    payoff = -(doy - target_doy).abs() - 0.001 * (agdd - agdd.mean()).abs()
    return payoff.astype(float)


def _ensure_outputs_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def run(out_dir: Path, seed: int = 42, n_bootstraps: int = 100,
        eml_features: int = 64) -> Dict[str, Path]:
    """Execute the demo and return the paths of the generated output files."""
    out_dir = _ensure_outputs_dir(out_dir)
    print(f"PhenoGame {__version__} — grape data-induced EML game extension demo")
    print("=" * 64)

    # 1. Load and filter
    npn = load_npn_csv("npn_grape.csv")
    sub = npn[npn["Phenophase_Description"] == PHENOPHASE].copy()
    if sub.empty:
        raise RuntimeError(f"No '{PHENOPHASE}' rows after NPN filtering.")
    print(f"Loaded {len(sub)} '{PHENOPHASE}' rows from npn_grape.csv")

    # 2. Surrogate utility — closed-form function of real NPN observations
    #    (no random / fabricated values; see _build_surrogate_utility docstring).
    sub = sub.assign(payoff=_build_surrogate_utility(sub, TARGET_DOY))
    feature_columns = [
        "Mean_First_Yes_DOY", "Mean_AGDD", "Latitude", "Elevation_in_Meters"
    ]

    # 3. Baseline comparison
    print("\n--- Baseline comparison (held-out test split) ---")
    comparison = compare_baselines(
        sub,
        feature_columns=feature_columns,
        target_column="payoff",
        test_fraction=0.3,
        seed=seed,
        n_eml_features=eml_features,
        include_random_forest=True,
    )
    print(comparison.summary())
    comparison_table = comparison.to_dataframe()

    # 4. EML-induced finite game and ε-CCE certification
    print("\n--- Compiling EML-induced finite game ---")
    game = compile_eml_payoff_game(
        sub,
        strategy_column="Mean_First_Yes_DOY",
        scenario_column="Mean_AGDD",
        feature_columns=feature_columns,
        target_column="payoff",
        n_eml_features=eml_features,
        seed=seed,
        n_strategy_bins=3,
        n_scenario_bins=3,
        imputation="row_min",
    )
    cce_cert = certify_provable(game.payoff_matrix, iterations=200, seed=seed)
    print(f"  matrix shape: {game.payoff_matrix.matrix.shape}")
    print(f"  CCE gap={cce_cert.cce_gap:.4f}  epsilon={cce_cert.epsilon:.4f}  "
          f"certified={cce_cert.is_certified}")

    # 5. Action stability (only run if we have enough bins to make it interesting)
    print("\n--- Action stability under bootstrap resampling ---")
    try:
        stability = action_stability(
            sub,
            strategy_column="Mean_First_Yes_DOY",
            scenario_column="Mean_AGDD",
            feature_columns=feature_columns,
            target_column="payoff",
            n_resamples=10,
            seed=seed,
            n_eml_features=max(16, eml_features // 2),
            iterations=50,
        )
        print(stability.summary())
    except (ValueError, RuntimeError) as exc:
        print(f"  action_stability skipped: {exc}")
        stability = None

    # 6. Standard phenology game (uses bundled steering pipeline)
    print("\n--- Standard phenology game (steering pipeline) ---")
    base_result = run_phenology_game(npn, phenophase=PHENOPHASE, seed=seed)
    print(f"  recommended_strategy={base_result.recommended_strategy}  "
          f"cce_gap={base_result.cce_gap:.4f}  certified={base_result.certified}")

    # 7. Robustness
    print("\n--- Robustness report ---")
    try:
        rr = robustness_report(npn, phenophase=PHENOPHASE,
                               n_bootstraps=n_bootstraps, seed=seed)
        print(f"  robustness_score={rr.robustness_score:.3f}  "
              f"modal={rr.bootstrap.modal_action}  "
              f"cert_rate={rr.bootstrap.certification_rate:.3f}")
    except (RuntimeError, ValueError) as exc:
        # Tiny bootstrap counts on sparse phenophases can fail; treat as a
        # graceful skip so the rest of the demo (and the audit certificate)
        # still produces deliverables.
        print(f"  robustness_report skipped: {exc}")
        rr = None

    # 8. Counterfactual timing
    print("\n--- Counterfactual timing test ---")
    cf = counterfactual_timing_test(
        npn, phenophase=PHENOPHASE,
        offsets=[-7.0, -5.0, -3.0, 0.0, 3.0, 5.0, 7.0], seed=seed,
    )
    print(cf.summary())

    # 9. Build the audit certificate
    assumptions: Dict[str, Any] = {
        "phenophase": PHENOPHASE,
        "target_doy": TARGET_DOY,
        "feature_columns": feature_columns,
        "payoff_definition": PAYOFF_FORMULA,
        "n_bootstraps": int(n_bootstraps),
        "n_eml_features": int(eml_features),
        "seed": int(seed),
        "phenogame_version": __version__,
    }
    limitations: List[str] = [
        "The payoff is a transparent surrogate, not a validated economic value.",
        "EML-tree learning is upstream representation, not a new equilibrium proof.",
        "Sample size is small (n=11 'Ripe fruits' rows in bundled data).",
        ("Counterfactual timing perturbs observed DOY only; it does not "
         "simulate weather, labor, or market consequences and therefore "
         "may yield identical recommendations across offsets when the "
         "shift is symmetric across strategies and scenarios."),
        "Action-stability frequencies are descriptive, not coverage probabilities.",
        "Results are conditional on supplied data quality and feature choices.",
    ]
    audit = build_game_audit_certificate(
        base_result,
        crop="grape",
        phenophase=PHENOPHASE,
        assumptions=assumptions,
        limitations=limitations,
        robustness=rr,
        counterfactual=cf,
    )

    cert_dict: Dict[str, Any] = audit.to_dict()
    cert_dict.update({
        "eml_induced_game": {
            "matrix_shape": list(game.payoff_matrix.matrix.shape),
            "strategy_labels": list(game.payoff_matrix.strategy_labels),
            "scenario_labels": list(game.payoff_matrix.scenario_labels),
            "cce_gap": float(cce_cert.cce_gap),
            "epsilon": float(cce_cert.epsilon),
            "certified": bool(cce_cert.is_certified),
            "iterations": int(cce_cert.iterations),
            "training_rmse": float(game.payoff_fit.rmse),
        },
        "baseline_comparison": _to_jsonable(comparison_table.to_dict(orient="records")),
        "action_stability": (
            None if stability is None else {
                "n_resamples": stability.n_resamples,
                "base_action": stability.base_action,
                "stability": stability.stability,
                "most_common_action": stability.most_common_action,
                "most_common_frequency": stability.most_common_frequency,
                "action_counts": stability.action_counts,
            }
        ),
        "counterfactual": {
            "offsets": list(cf.offsets),
            "recommendations": list(cf.recommendations),
            "cce_gaps": _to_jsonable(cf.cce_gaps),
            "certified": [bool(x) for x in cf.certified],
            "baseline_action": cf.baseline_action,
            "best_offset": float(cf.best_offset),
            "best_estimated_payoff": float(cf.best_estimated_payoff),
        },
    })

    # 10. Persist artifacts
    paths = {
        "certificate": out_dir / "grape_decision_certificate.json",
        "robustness": out_dir / "grape_robustness_summary.csv",
        "counterfactual": out_dir / "grape_counterfactual_table.csv",
        "baselines": out_dir / "grape_baseline_comparison.csv",
    }

    paths["certificate"].write_text(
        json.dumps(_to_jsonable(cert_dict), indent=2, sort_keys=True)
    )

    rows: List[Dict[str, Any]]
    if rr is None:
        rows = [{
            "metric": "robustness_score",
            "value": "skipped",
            "note": "robustness_report failed (likely too few valid bootstraps); see stdout",
        }]
    else:
        rows = [{
            "metric": "robustness_score",
            "value": float(rr.robustness_score),
            "note": "0.4*modal_freq + 0.3*cert_rate + 0.3*sensitivity_stability",
        }, {
            "metric": "modal_strategy",
            "value": rr.bootstrap.modal_action,
            "note": "modal recommendation across bootstraps",
        }, {
            "metric": "modal_frequency",
            "value": float(rr.bootstrap.modal_frequency),
            "note": "fraction of bootstraps recommending modal",
        }, {
            "metric": "certification_rate",
            "value": float(rr.bootstrap.certification_rate),
            "note": "fraction of bootstraps where ε-CCE certification passed",
        }]
        for s in rr.sensitivity:
            rows.append({
                "metric": f"sensitivity::{s.parameter}",
                "value": "stable" if s.stable else "unstable",
                "note": f"recommendation stable across perturbations of {s.parameter}",
            })
    pd.DataFrame(rows).to_csv(paths["robustness"], index=False)

    cf_table = pd.DataFrame({
        "offset_days": cf.offsets,
        "recommendation": cf.recommendations,
        "cce_gap": cf.cce_gaps,
        "certified": cf.certified,
    })
    cf_table.to_csv(paths["counterfactual"], index=False)

    comparison_table.to_csv(paths["baselines"], index=False)

    print("\n--- Artifacts written ---")
    for label, p in paths.items():
        print(f"  {label}: {p}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path("outputs"),
        help="Directory to write demo artifacts (default: ./outputs)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstraps", type=int, default=100)
    parser.add_argument("--eml-features", type=int, default=64)
    args = parser.parse_args()
    run(args.out, seed=args.seed, n_bootstraps=args.bootstraps,
        eml_features=args.eml_features)


if __name__ == "__main__":
    main()
