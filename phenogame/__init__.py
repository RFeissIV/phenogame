"""
phenogame — Farmer vs Nature: game-theoretic crop phenology decisions.

Calibrate response functions on field data, build finite decision games over
farmer strategies and Nature scenarios, certify epsilon-CCE policies, and
attribute outcomes to drivers.

    Calibrate: field data -> response functions
    Game:      farmer strategies x Nature scenarios -> DPUU recommendations
    Certify:   no-regret dynamics -> data-induced epsilon-CCE certificate
    Attribute: Banzhaf/Shapley driver attribution -> why was my crop late?

Math: EML operator eml(x,y) = exp(x) - ln(y), Odrzywolek (2026).
Game theory: Dillon (1962), "Applications of game theory in
agricultural economics: Review and requiem."
"""

__version__ = "0.3.0"

__all__ = [
    "PhenoGame", "__version__",
    "RESPONSE_FUNCTIONS", "ResponseFunction",
    "list_functions", "get_function", "functions_for_scale", "functions_for_model",
    "fit_all", "fit_single", "FitReport", "FitResult",
    "export_eml", "EMLExport", "reduction_chain", "CORE_IDENTITIES",
    "build_payoff_matrix", "PayoffMatrix",
    "solve_all_criteria", "DPUUResult",
    "wald_maximin", "laplace", "hurwicz", "savage_regret",
    "nash_equilibrium", "NashResult",
    "zero_sum_minimax", "MinimaxResult",
    "zero_sum_minimax_equilibrium", "zero_sum_minimax_solution",
    "banzhaf_attribution", "GameReport",
    "plot_fit",
    "certify_provable", "certify_empirical", "compute_cce_gap", "CCECertificate",
    "crop_steering_certificate", "SteeringCertificate", "PayoffFormula",
    "build_payoff_matrix_from_data", "DataSummary",
    "load_npn_csv", "run_phenology_game", "PhenologyGameResult",
    "CropGameResult", "run_game_for_crop", "run_multi_crop",
    "bootstrap_policy_interval", "BootstrapResult",
    "sensitivity_analysis", "SensitivityResult",
    "robustness_report", "RobustnessReport",
    "eml", "EMLTreeRegressor", "EMLPayoffFit", "fit_eml_payoff_model",
    "EMLNode", "EMLTree", "fit_eml_tree", "evaluate_eml_tree",
    "compile_eml_payoff_game", "DataInducedGame",
    "counterfactual_timing_test", "CounterfactualResult",
    "decision_to_triples", "decision_to_jsonld", "DecisionTriple",
    "build_game_audit_certificate", "GameAuditCertificate",
    "MultiAgentDecisionGame",
    "train_test_split_indices", "compare_baselines", "BaselineComparison", "ModelScore",
    "action_stability", "ActionStabilityResult",
    # New EML game-compiler pipeline
    "pairwise_win_matrix", "preference_probabilities",
    "check_binary_choice_constraints", "triangle_violations",
    "stochastic_choice_summary",
    "EMLFamilySpec", "FittedEMLFamily",
    "define_13_eml_families", "fit_eml_family", "fit_all_eml_families",
    "predict_eml_family", "rank_eml_families",
    "NatureScenarios", "build_nature_scenarios", "learned_payoff_surface",
    "construct_13_by_s_payoff_matrix", "compile_game_from_data", "CompiledGame",
    "coalition_value_from_pairwise", "coalition_game",
    "marginal_contribution", "max_marginal_contributions",
    "gilboa_monderer_u0_bound", "binary_choice_game_certificate",
    "generate_game_audit_certificate",
    "export_certificate_json", "export_certificate_markdown",
    "DEFAULT_LIMITATIONS",
    "available_species", "load_real_npn", "load_phenology_table",
    "real_score_matrix_from_models", "real_temperature_trace",
    # v4 EML-RML audit response
    "psi_standard", "psi_exponential", "psi_softplus", "psi_eml",
    "TRANSFORM_REGISTRY", "list_transforms",
    "joint_correlation_residual_normalized", "joint_correlation_residual_raw",
    "compare_regret_transforms",
    "RegretComparisonResult", "TransformAggregate", "TransformRun",
]

import numpy as np
from typing import Dict, Optional

from .response import (
    RESPONSE_FUNCTIONS, ResponseFunction,
    list_functions, get_function, functions_for_scale, functions_for_model,
)
from .fit import fit_all, fit_single, FitReport, FitResult
from .eml import export_eml, EMLExport, reduction_chain, CORE_IDENTITIES
from .game import (
    build_payoff_matrix, PayoffMatrix,
    solve_all_criteria, DPUUResult,
    wald_maximin, laplace, hurwicz, savage_regret,
    nash_equilibrium, NashResult,
    zero_sum_minimax, MinimaxResult,
    banzhaf_attribution, GameReport,
)


# Plotting functions are lazy-loaded to avoid importing Matplotlib during package import.
def plot_fit(*args, **kwargs):
    from .viz import plot_fit as _plot_fit
    return _plot_fit(*args, **kwargs)


from .equilibrium import (
    certify_provable, certify_empirical, compute_cce_gap, CCECertificate,
    zero_sum_minimax_equilibrium, zero_sum_minimax_solution,
)
from .steering import (
    crop_steering_certificate, SteeringCertificate, PayoffFormula,
    build_payoff_matrix_from_data, DataSummary,
)
from .pipeline import (
    load_npn_csv, run_phenology_game, PhenologyGameResult,
    CropGameResult, run_game_for_crop, run_multi_crop,
)
from .robustness import (
    bootstrap_policy_interval, BootstrapResult,
    sensitivity_analysis, SensitivityResult,
    robustness_report, RobustnessReport,
)


class PhenoGame:
    """Calibrated phenology response model with game-theoretic decision support.

    Fit response functions on field data, then build finite decision games
    (farmer strategies x Nature scenarios), certify epsilon-CCE policies,
    and attribute outcomes to drivers.
    """

    def __init__(self):
        self.report: Optional[FitReport] = None
        self._func_id: Optional[str] = None
        self._params: Optional[Dict[str, float]] = None

    @property
    def is_fitted(self) -> bool:
        return self._func_id is not None

    @property
    def best_function(self) -> str:
        if not self.is_fitted:
            raise RuntimeError("Not fitted. Call .fit(x, y) first.")
        return self._func_id

    @property
    def best_params(self) -> Dict[str, float]:
        if not self.is_fitted:
            raise RuntimeError("Not fitted. Call .fit(x, y) first.")
        return self._params.copy()

    def fit(self, x, y, func_ids=None, maxfev=5000) -> FitReport:
        """Fit response functions to data. Ranks by AIC."""
        self.report = fit_all(np.asarray(x, float), np.asarray(y, float),
                              func_ids=func_ids, maxfev=maxfev)
        self._func_id = self.report.best.func_id
        self._params = self.report.best.params.copy()
        return self.report

    def use_model(self, func_id: str, params: Dict[str, float]):
        """Manually set model and parameters."""
        if func_id not in RESPONSE_FUNCTIONS:
            raise KeyError(f"Unknown: '{func_id}'. Available: {list_functions()}")
        self._func_id = func_id
        self._params = params.copy()

    def predict(self, x) -> np.ndarray:
        """Predict response values."""
        if not self.is_fitted:
            raise RuntimeError("Not fitted.")
        return RESPONSE_FUNCTIONS[self._func_id].fn(np.asarray(x, float), **self._params)

    def game(self, env_scenarios, planting_dates, target_threshold,
             target_date, date_col="date", driver_col="temperature",
             metric="gdd", alpha=0.5) -> GameReport:
        """Farmer vs Nature: DPUU decision analysis (Dillon 1962)."""
        if not self.is_fitted:
            raise RuntimeError("Not fitted.")
        pm = build_payoff_matrix(env_scenarios, planting_dates,
                                 self._func_id, self._params,
                                 target_threshold, target_date,
                                 date_col, driver_col, metric)
        criteria = solve_all_criteria(pm, alpha)
        nash = nash_equilibrium(pm)
        return GameReport(payoff=pm, criteria=criteria, nash=nash)

    def attribute(self, env, driver_cols, func_ids, params_dict,
                  baseline_values, date_col="date"):
        """Banzhaf/Shapley driver attribution."""
        return banzhaf_attribution(env, driver_cols, func_ids, params_dict,
                                   baseline_values, date_col)

    def eml_export(self) -> EMLExport:
        """Export calibrated model as EML decomposition."""
        if not self.is_fitted:
            raise RuntimeError("Not fitted.")
        return export_eml(self._func_id, self._params)

    def certify(self, env_scenarios, planting_dates, target_threshold,
                target_date, mode="provable", iterations=10000,
                tau=1.0, seed=0, date_col="date",
                driver_col="temperature", metric="gdd") -> CCECertificate:
        """Run no-regret dynamics and certify epsilon-CCE.

        Parameters
        ----------
        mode : str
            "provable" -- Hedge with formal no-regret bound (recommended).
            "empirical" -- EML-derived transform (measured, not guaranteed).
        iterations : int
            Number of rounds of play.
        tau : float
            Temperature for EML transform (empirical mode only).

        Returns
        -------
        CCECertificate with gap, epsilon, and honest mode labeling.
        """
        if not self.is_fitted:
            raise RuntimeError("Not fitted.")
        pm = build_payoff_matrix(env_scenarios, planting_dates,
                                 self._func_id, self._params,
                                 target_threshold, target_date,
                                 date_col, driver_col, metric)
        if mode == "provable":
            return certify_provable(pm, iterations, seed)
        elif mode == "empirical":
            return certify_empirical(pm, iterations, tau, seed)
        else:
            raise ValueError(f"mode must be 'provable' or 'empirical', got '{mode}'")

    def plot_fit(self, top_n=4, **kw):
        if self.report is None:
            raise RuntimeError("Not fitted.")
        return plot_fit(self.report, top_n=top_n, **kw)

    @staticmethod
    def crop_steering_certificate(
        data, action_columns, scenario_column, outcome_column, **kw
    ) -> SteeringCertificate:
        """Produce a Crop Steering epsilon-CCE Certificate from tabular data.

        This is a convenience method that delegates to the
        phenogame.steering.crop_steering_certificate() function.

        See phenogame.steering.crop_steering_certificate for full documentation.
        """
        return crop_steering_certificate(
            data, action_columns, scenario_column, outcome_column, **kw
        )


# Data-induced EML game compiler extension: data-induced games, counterfactuals, KG export.
from .eml_tree import (
    eml, EMLTreeRegressor, EMLPayoffFit, fit_eml_payoff_model,
    EMLNode, EMLTree, fit_eml_tree, evaluate_eml_tree,
)
from .data_compiler import compile_eml_payoff_game, DataInducedGame
from .counterfactual import counterfactual_timing_test, CounterfactualResult
from .ontology import decision_to_triples, decision_to_jsonld, DecisionTriple
from .game_certificate import build_game_audit_certificate, GameAuditCertificate
from .multiagent import MultiAgentDecisionGame
from .validation import (
    train_test_split_indices,
    compare_baselines, BaselineComparison, ModelScore,
    action_stability, ActionStabilityResult,
)

# Forward-only EML game-compiler pipeline:
#   data → 13 EML families → pairwise P → finite game → minimax/CCE
#   → robustness → recommendation + audit certificate
from .binary_choice import (
    pairwise_win_matrix, preference_probabilities,
    check_binary_choice_constraints, triangle_violations,
    stochastic_choice_summary,
)
from .eml_families import (
    EMLFamilySpec, FittedEMLFamily,
    define_13_eml_families, fit_eml_family, fit_all_eml_families,
    predict_eml_family, rank_eml_families,
)
from .game_compiler import (
    NatureScenarios, build_nature_scenarios, learned_payoff_surface,
    construct_13_by_s_payoff_matrix, compile_game_from_data, CompiledGame,
)
from .bsc_game_bridge import (
    coalition_value_from_pairwise, coalition_game,
    marginal_contribution, max_marginal_contributions,
    gilboa_monderer_u0_bound, binary_choice_game_certificate,
)
from .certificate import (
    generate_game_audit_certificate,
    export_certificate_json, export_certificate_markdown,
    DEFAULT_LIMITATIONS,
)
from .fixtures import (
    available_species, load_real_npn, load_phenology_table,
    real_score_matrix_from_models, real_temperature_trace,
)

# EML-RML twenty-pass-audit response (v4): four-transform regret-matching
# comparator, normalized joint correlation residual, and pure-function
# kernels. See EML_RML_AUDIT_RESPONSE.md.
from .regret_transforms import (
    psi_standard, psi_exponential, psi_softplus, psi_eml,
    TRANSFORM_REGISTRY, list_transforms,
)
from .joint_correlation import (
    joint_correlation_residual_normalized,
    joint_correlation_residual_raw,
)
from .regret_panel import (
    compare_regret_transforms,
    RegretComparisonResult, TransformAggregate, TransformRun,
)
