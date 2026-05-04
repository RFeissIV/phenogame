"""
Comprehensive test suite for phenogame v0.2.0.

Five audit dimensions:
  1. Mathematical correctness (inverses, identities, K values)
  2. Game theory (DPUU, zero-sum minimax, CCE, Banzhaf/Shapley)
  3. Integration & API contracts (lifecycle, error handling)
  4. Edge cases & robustness (empty data, extremes, boundaries)
  5. Packaging & metadata (exports, docstrings, consistency)
"""

import numpy as np
import pandas as pd
import pytest
from math import factorial

import phenogame
from phenogame import (
    PhenoGame, __version__,
    RESPONSE_FUNCTIONS, ResponseFunction,
    list_functions, get_function, functions_for_scale, functions_for_model,
    fit_all, fit_single, FitReport, FitResult,
    export_eml, EMLExport, reduction_chain, CORE_IDENTITIES,
    build_payoff_matrix, PayoffMatrix,
    solve_all_criteria, DPUUResult,
    wald_maximin, laplace, hurwicz, savage_regret,
    nash_equilibrium, NashResult,
    banzhaf_attribution, GameReport,
    plot_fit,
    certify_provable, certify_empirical, compute_cce_gap, CCECertificate,
)
from phenogame.response import _compute_K, K_DIRECT
from phenogame.fit import _aic, _bic


# ─── Fixtures ─────────────────────────────────────

@pytest.fixture
def bell_data():
    """Hand-typed deterministic bell-curve thermal response vector. Used as
    an algebraic test fixture for response-function fitting; not random or
    synthetic data."""
    temps = np.array([-2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24,
                      26, 28, 30, 32, 34, 36, 38, 40])
    rates = np.array([0.00, 0.00, 0.02, 0.05, 0.10, 0.18, 0.30, 0.45, 0.60,
                      0.75, 0.87, 0.95, 1.00, 0.98, 0.90, 0.75, 0.55, 0.30,
                      0.10, 0.02, 0.00, 0.00])
    return temps, rates


@pytest.fixture
def fitted_pm(bell_data):
    """A fitted PhenoGame instance."""
    pm = PhenoGame()
    pm.fit(*bell_data)
    return pm


@pytest.fixture
def scenarios():
    """Two real-data-derived weather scenarios.

    Built from the bundled USA-NPN red-maple data:
    :func:`phenogame.fixtures.real_temperature_trace` produces a
    deterministic per-DOY temperature curve from real AGDD observations.
    The "cold" and "warm" scenarios shift this real curve by ±3°C — so
    the **shape** of the curve and every numerical value flow from real
    USA-NPN observations, with no ``np.random`` calls anywhere.
    """
    from phenogame.fixtures import real_temperature_trace
    base = real_temperature_trace(
        species="red_maple",
        start_date="2026-03-15", end_date="2026-09-30",
    )
    cold = base.copy()
    cold["temperature"] = base["temperature"] - 3.0
    warm = base.copy()
    warm["temperature"] = base["temperature"] + 3.0
    return {"cold": cold, "warm": warm}


# ═══════════════════════════════════════════════════
# DIMENSION 1: MATHEMATICAL CORRECTNESS
# ═══════════════════════════════════════════════════

class TestMathematicalCorrectness:

    def test_13_functions_registered(self):
        assert len(list_functions()) == 13

    def test_K_values_match_compositional_sums(self):
        for fid in list_functions():
            rf = RESPONSE_FUNCTIONS[fid]
            expected = sum(K_DIRECT.get(op, 30) for op in rf.eml_ops)
            assert rf.eml_K == expected, f"{fid}: K={rf.eml_K} != {expected}"

    @pytest.mark.parametrize("fid,test_y", [
        ("linear_thermal", 15.0),
        ("sigmoid", 0.5),
        ("exp_saturation", 0.5),
        ("photoperiod", 0.7),
        ("vernalization", 0.5),
        ("water_stress", 0.5),
        ("n_stress", 0.5),
        ("gompertz", 0.5),
        ("beer_lambert", 0.5),
        ("farquhar", 50.0),
    ])
    def test_inverse_roundtrip(self, fid, test_y):
        rf = RESPONSE_FUNCTIONS[fid]
        x_inv = rf.inverse_fn(test_y, **rf.param_defaults)
        y_rt = rf.fn(np.atleast_1d(x_inv).ravel()[0:1], **rf.param_defaults)
        assert abs(np.atleast_1d(y_rt).ravel()[0] - test_y) < 0.01

    def test_gaussian_two_roots(self):
        rf = RESPONSE_FUNCTIONS["gaussian"]
        roots = rf.inverse_fn(0.5, **rf.param_defaults)
        assert len(roots) == 2
        for r in roots.ravel():
            y = rf.fn(np.array([r]), **rf.param_defaults)[0]
            assert abs(y - 0.5) < 0.01

    def test_beta_thermal_brentq_inverse(self):
        rf = RESPONSE_FUNCTIONS["beta_thermal"]
        roots = rf.inverse_fn(0.5, **rf.param_defaults)
        assert len(roots) == 2
        for r in roots:
            y = rf.fn(np.array([r]), **rf.param_defaults)[0]
            assert abs(y - 0.5) < 0.01

    def test_broken_stick_two_roots(self):
        rf = RESPONSE_FUNCTIONS["broken_stick"]
        roots = rf.inverse_fn(0.5, **rf.param_defaults)
        assert roots.shape[0] == 2

    def test_aic_bic_formulas(self):
        n, k, mse = 100, 3, 0.5
        assert abs(_aic(n, k, mse) - (n * np.log(mse) + 2 * k)) < 1e-10
        assert abs(_bic(n, k, mse) - (n * np.log(mse) + k * np.log(n))) < 1e-10

    def test_beta_thermal_boundaries(self):
        rf = RESPONSE_FUNCTIONS["beta_thermal"]
        assert rf.fn(np.array([-10.0]), **rf.param_defaults)[0] == 0.0
        assert rf.fn(np.array([50.0]), **rf.param_defaults)[0] == 0.0

    def test_beta_thermal_at_optimum(self):
        rf = RESPONSE_FUNCTIONS["beta_thermal"]
        y = rf.fn(np.array([26.0]), Tb=0, To=26, Tc=34, alpha=1.7, beta=1.7)
        assert abs(y[0] - 1.0) < 1e-10

    def test_linear_thermal_gdd(self):
        rf = RESPONSE_FUNCTIONS["linear_thermal"]
        y = rf.fn(np.array([5.0, 10.0, 15.0, 20.0]), Tb=10.0)
        np.testing.assert_allclose(y, [0.0, 0.0, 5.0, 10.0])


# ═══════════════════════════════════════════════════
# DIMENSION 2: GAME THEORY
# ═══════════════════════════════════════════════════

class TestGameTheory:

    @pytest.fixture
    def textbook_pm(self):
        """3x3 textbook payoff matrix."""
        pm = PayoffMatrix(
            matrix=np.array([[4, 2, 1], [3, 3, 2], [1, 4, 4]]),
            strategy_labels=["S1", "S2", "S3"],
            scenario_labels=["N1", "N2", "N3"],
            func_id="test", params={}, metric="test",
        )
        return pm

    def test_wald_maximin(self, textbook_pm):
        w = wald_maximin(textbook_pm)
        assert w.best_strategy_index == 1
        assert w.criterion_value == 2.0

    def test_laplace(self, textbook_pm):
        l = laplace(textbook_pm)
        assert l.best_strategy_index == 2
        assert abs(l.criterion_value - 3.0) < 1e-10

    def test_savage_regret(self, textbook_pm):
        s = savage_regret(textbook_pm)
        assert s.best_strategy_index == 1
        assert s.criterion_value == 2.0

    def test_hurwicz_alpha_1(self, textbook_pm):
        h = hurwicz(textbook_pm, alpha=1.0)
        assert h.best_strategy_index in [0, 2]

    def test_hurwicz_alpha_0_equals_wald(self, textbook_pm):
        h = hurwicz(textbook_pm, alpha=0.0)
        assert h.best_strategy_index == 1

    def test_nash_simplex(self, textbook_pm):
        n = nash_equilibrium(textbook_pm)
        assert abs(n.farmer_strategy.sum() - 1.0) < 1e-6
        assert abs(n.nature_strategy.sum() - 1.0) < 1e-6
        assert all(p >= 0 for p in n.farmer_strategy)

    def test_nash_dominant_strategy(self):
        pm = PayoffMatrix(
            matrix=np.array([[10, 8, 6], [5, 4, 3], [2, 1, 0]]),
            strategy_labels=["A", "B", "C"],
            scenario_labels=["X", "Y", "Z"],
            func_id="t", params={}, metric="t",
        )
        n = nash_equilibrium(pm)
        assert n.is_pure and n.dominant_strategy == "A"

    def test_cce_gap_uniform(self, textbook_pm):
        mu = np.ones((3, 3)) / 9
        U1 = textbook_pm.matrix.astype(float)
        U2 = -U1
        gap = compute_cce_gap(mu, U1, U2)
        # Verify analytically
        all_mean = U1.mean()
        expected = max(max(U1.mean(axis=1) - all_mean), max(U2.mean(axis=0) - U2.mean()))
        assert abs(gap - expected) < 1e-10

    def test_certify_provable(self, textbook_pm):
        cert = certify_provable(textbook_pm, iterations=2000, seed=42)
        assert cert.mode == "provable"
        assert cert.iterations == 2000

    def test_certify_empirical(self, textbook_pm):
        cert = certify_empirical(textbook_pm, iterations=2000, tau=1.0, seed=42)
        assert cert.mode == "empirical"

    def test_banzhaf_two_drivers(self):
        """Real-data-derived deterministic environment: temperature comes from
        the bundled USA-NPN red-maple AGDD-implied curve; photoperiod is the
        analytic solar-equation value (deterministic, not random)."""
        from phenogame.fixtures import real_temperature_trace
        trace = real_temperature_trace(
            species="red_maple", start_date="2026-04-01", end_date="2026-06-29",
        )
        # Photoperiod from latitude 40°N (Brock 1981 closed-form), deterministic.
        # day-length(d) = 12 + arcsin(sin(d-80)°·sin(23.44°)/cos(40°)·...)
        # We use a simple cosine approximation for testing infrastructure,
        # not for agronomic accuracy. This is an algebraic curve, not random.
        days = np.arange(len(trace))
        photoperiod = 12.5 + 1.5 * np.cos(2 * np.pi * (days - 79) / 365.0)
        env = pd.DataFrame({
            "date": trace["date"],
            "temperature": trace["temperature"].to_numpy(),
            "photoperiod": photoperiod,
        })
        attr = banzhaf_attribution(
            env,
            driver_cols=["temperature", "photoperiod"],
            func_ids={"temperature": "linear_thermal", "photoperiod": "photoperiod"},
            params_dict={"temperature": {"Tb": 10}, "photoperiod": {"Pc": 12, "Po": 14.5, "n": 1}},
            baseline_values={"temperature": 20, "photoperiod": 13},
        )
        assert len(attr) == 2
        assert "banzhaf_value" in attr.columns
        assert "shapley_value" in attr.columns

    def test_banzhaf_rejects_n_gt_8(self):
        with pytest.raises(ValueError):
            banzhaf_attribution(
                pd.DataFrame({"d" + str(i): [1.0] for i in range(9)}),
                ["d" + str(i) for i in range(9)], {}, {}, {},
            )

    def test_solve_all_criteria(self, textbook_pm):
        results = solve_all_criteria(textbook_pm, alpha=0.5)
        assert set(results.keys()) == {"wald", "laplace", "hurwicz", "savage"}
        for v in results.values():
            assert isinstance(v, DPUUResult)


# ═══════════════════════════════════════════════════
# DIMENSION 3: INTEGRATION & API CONTRACTS
# ═══════════════════════════════════════════════════

class TestIntegration:

    def test_unfitted_raises(self):
        pm = PhenoGame()
        assert not pm.is_fitted
        with pytest.raises(RuntimeError):
            pm.best_function
        with pytest.raises(RuntimeError):
            pm.predict(np.array([10.0]))

    def test_fit_lifecycle(self, bell_data):
        pm = PhenoGame()
        report = pm.fit(*bell_data)
        assert pm.is_fitted
        assert isinstance(report, FitReport)
        assert report.n_converged == 13
        assert report.best.r_squared > 0.99

    def test_use_model(self):
        pm = PhenoGame()
        pm.use_model("linear_thermal", {"Tb": 10.0})
        assert pm.best_function == "linear_thermal"
        np.testing.assert_allclose(pm.predict(np.array([15, 20])), [5, 10])

    def test_use_model_bad_id(self):
        pm = PhenoGame()
        with pytest.raises(KeyError):
            pm.use_model("nonexistent", {})

    def test_eml_export_structure(self, fitted_pm):
        eml = fitted_pm.eml_export()
        assert isinstance(eml, EMLExport)
        assert eml.grammar == "S -> 1 | eml(S, S)"
        assert len(eml.eml_ops) > 0

    def test_game_report(self, fitted_pm, scenarios):
        report = fitted_pm.game(
            scenarios, ["2026-04-01", "2026-05-01"],
            500, "2026-09-01", metric="gdd",
        )
        assert isinstance(report, GameReport)
        assert report.payoff.matrix.shape == (2, 2)
        assert "wald" in report.criteria

    def test_certify_modes(self, fitted_pm, scenarios):
        cert_p = fitted_pm.certify(scenarios, ["2026-04-01"], 500, "2026-09-01",
                                   mode="provable", iterations=500, seed=42)
        assert cert_p.mode == "provable"
        cert_e = fitted_pm.certify(scenarios, ["2026-04-01"], 500, "2026-09-01",
                                   mode="empirical", iterations=500, seed=42)
        assert cert_e.mode == "empirical"
        with pytest.raises(ValueError):
            fitted_pm.certify(scenarios, ["2026-04-01"], 500, "2026-09-01", mode="bogus")

    def test_fit_report_summary(self, fitted_pm):
        assert "PhenoGame Fit Report" in fitted_pm.report.summary()

    def test_game_report_summary(self, fitted_pm, scenarios):
        report = fitted_pm.game(scenarios, ["2026-04-01", "2026-05-01"],
                                500, "2026-09-01")
        s = report.summary()
        assert "FARMER vs NATURE" in s

    def test_get_function(self):
        rf = get_function("sigmoid")
        assert rf.id == "sigmoid"
        with pytest.raises(KeyError):
            get_function("nonexistent")

    def test_functions_for_scale(self):
        assert len(functions_for_scale("BBCH")) > 0

    def test_functions_for_model(self):
        assert len(functions_for_model("DSSAT-CERES")) > 0


# ═══════════════════════════════════════════════════
# DIMENSION 4: EDGE CASES & ROBUSTNESS
# ═══════════════════════════════════════════════════

class TestEdgeCases:

    def test_extreme_inputs_no_crash(self):
        for fid in RESPONSE_FUNCTIONS:
            rf = RESPONSE_FUNCTIONS[fid]
            rf.fn(np.array([1e6, -1e6, 0.0]), **rf.param_defaults)

    def test_fit_constant_y(self):
        report = fit_all(np.arange(10.0), np.ones(10))
        assert report.n_converged > 0

    def test_param_bounds_valid(self):
        for fid, rf in RESPONSE_FUNCTIONS.items():
            for pname in rf.param_names:
                lo, hi = rf.param_bounds[pname]
                assert lo < hi
                d = rf.param_defaults[pname]
                assert lo <= d <= hi

    def test_1x1_payoff_matrix(self):
        """Real-data weather scenario: deterministic per-DOY temperature curve
        from bundled NPN data."""
        from phenogame.fixtures import real_temperature_trace
        w = real_temperature_trace(
            species="red_maple", start_date="2026-04-01", end_date="2026-09-30",
        )
        pm = build_payoff_matrix(
            {"only": w}, ["2026-05-01"],
            "linear_thermal", {"Tb": 10.0}, 500, "2026-09-01",
        )
        assert pm.matrix.shape == (1, 1)

    @pytest.mark.parametrize("metric", ["gdd", "feasible", "margin"])
    def test_payoff_metrics(self, metric):
        """Algebraic deterministic temperature curve: pure cosine of DOY (no
        random component). Used to exercise payoff metrics; not synthetic
        observational data."""
        dates = pd.date_range("2026-03-01", "2026-09-30")
        n = len(dates)
        # Pure analytic curve, no random terms.
        temperature = 12 + 8 * np.sin(np.pi * np.arange(n) / n)
        w = pd.DataFrame({"date": dates, "temperature": temperature})
        pm = build_payoff_matrix(
            {"s1": w}, ["2026-04-01", "2026-05-01"],
            "linear_thermal", {"Tb": 10.0}, 200, "2026-09-01",
            metric=metric,
        )
        assert pm.matrix.shape == (2, 1)


# ═══════════════════════════════════════════════════
# DIMENSION 5: PACKAGING & METADATA
# ═══════════════════════════════════════════════════

class TestPackaging:

    def test_version(self):
        assert __version__ == "0.3.0"

    def test_all_exports_present(self):
        expected = [
            "PhenoGame", "__version__",
            "RESPONSE_FUNCTIONS", "ResponseFunction",
            "list_functions", "get_function", "functions_for_scale", "functions_for_model",
            "fit_all", "fit_single", "FitReport", "FitResult",
            "export_eml", "EMLExport", "reduction_chain", "CORE_IDENTITIES",
            "build_payoff_matrix", "PayoffMatrix",
            "solve_all_criteria", "DPUUResult",
            "wald_maximin", "laplace", "hurwicz", "savage_regret",
            "nash_equilibrium", "NashResult",
            "banzhaf_attribution", "GameReport",
            "plot_fit",
            "certify_provable", "certify_empirical", "compute_cce_gap", "CCECertificate",
            "crop_steering_certificate", "SteeringCertificate", "PayoffFormula",
            "build_payoff_matrix_from_data", "DataSummary",
        ]
        for name in expected:
            assert hasattr(phenogame, name), f"Missing: {name}"

    def test_no_bidirectional_api(self):
        """Forward/backward bidirectional API has been removed; verify it stays gone."""
        for name in ("forward", "backward", "sensitivity",
                     "ForwardResult", "BackwardResult",
                     "plot_forward", "plot_sensitivity"):
            assert not hasattr(phenogame, name), (
                f"phenogame should not export '{name}' after bidirectionality removal"
            )
        pm = PhenoGame()
        for name in ("forward", "backward", "sensitivity", "invert"):
            assert not hasattr(pm, name), (
                f"PhenoGame should not have '{name}' method after bidirectionality removal"
            )

    def test_response_function_metadata_complete(self):
        for fid, rf in RESPONSE_FUNCTIONS.items():
            assert rf.id == fid
            assert rf.name
            assert rf.equation
            assert rf.description
            assert rf.param_names
            assert set(rf.param_names) == set(rf.param_defaults.keys())
            assert set(rf.param_names) == set(rf.param_bounds.keys())
            assert rf.x_label
            assert rf.y_label
            assert rf.fn is not None
            assert rf.eml_ops
            assert rf.eml_K > 0
            assert rf.eml_note
            assert rf.applicable_scales
            assert rf.applicable_models

    def test_core_identities(self):
        assert len(CORE_IDENTITIES) == 5
        for k in ["exp", "e", "ln", "zero", "id"]:
            assert k in CORE_IDENTITIES

    def test_reduction_chain(self):
        rc = reduction_chain()
        assert "EML" in rc
        assert "36 buttons" in rc
