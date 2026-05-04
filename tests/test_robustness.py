"""Tests for phenogame.robustness — bootstrap intervals and sensitivity analysis."""

import numpy as np
import pandas as pd
import pytest
import warnings

from phenogame import (
    load_npn_csv,
    bootstrap_policy_interval, BootstrapResult,
    sensitivity_analysis, SensitivityResult,
    robustness_report, RobustnessReport,
)


@pytest.fixture
def npn():
    return load_npn_csv()


class TestBootstrap:

    def test_basic_bootstrap(self, npn):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = bootstrap_policy_interval(
                npn, phenophase="Ripe fruits",
                n_bootstraps=50, seed=42,
            )
        assert isinstance(result, BootstrapResult)
        assert result.n_bootstraps >= 10
        assert result.modal_action in ["early", "standard", "late"]
        assert 0 <= result.modal_frequency <= 1
        assert result.cce_gap_ci[0] <= result.cce_gap_mean <= result.cce_gap_ci[1]
        assert 0 <= result.certification_rate <= 1
        assert 0 <= result.consensus_rate <= 1

    def test_bootstrap_seed_stable(self, npn):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r1 = bootstrap_policy_interval(npn, n_bootstraps=30, seed=42)
            r2 = bootstrap_policy_interval(npn, n_bootstraps=30, seed=42)
        assert r1.modal_action == r2.modal_action
        assert r1.cce_gap_mean == r2.cce_gap_mean

    def test_bootstrap_summary_renders(self, npn):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = bootstrap_policy_interval(npn, n_bootstraps=30, seed=42)
        s = result.summary()
        assert "Bootstrap Policy Interval" in s
        assert "Modal action" in s

    def test_bootstrap_bad_phenophase(self, npn):
        with pytest.raises(ValueError):
            bootstrap_policy_interval(npn, phenophase="Nonexistent")


class TestSensitivity:

    def test_basic_sensitivity(self, npn):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            results = sensitivity_analysis(npn, seed=42)
        assert len(results) == 2  # doy_offset + strategy_scale
        for r in results:
            assert isinstance(r, SensitivityResult)
            assert len(r.values) > 0
            assert len(r.recommendations) == len(r.values)
            assert isinstance(r.stable, bool)

    def test_doy_offset_result(self, npn):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            results = sensitivity_analysis(
                npn, doy_offsets=[-3, 0, 3], seed=42,
            )
        doy_result = results[0]
        assert doy_result.parameter == "doy_offset"
        assert len(doy_result.values) == 3

    def test_sensitivity_summary_renders(self, npn):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            results = sensitivity_analysis(npn, seed=42)
        for r in results:
            s = r.summary()
            assert "Sensitivity" in s
            assert "Stable" in s


class TestRobustnessReport:

    def test_full_report(self, npn):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            report = robustness_report(
                npn, n_bootstraps=30, seed=42,
            )
        assert isinstance(report, RobustnessReport)
        assert isinstance(report.bootstrap, BootstrapResult)
        assert len(report.sensitivity) == 2
        assert 0 <= report.robustness_score <= 1

    def test_report_summary_renders(self, npn):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            report = robustness_report(npn, n_bootstraps=30, seed=42)
        s = report.summary()
        assert "ROBUSTNESS REPORT" in s
        assert "score" in s.lower()

    def test_report_seed_stable(self, npn):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r1 = robustness_report(npn, n_bootstraps=30, seed=42)
            r2 = robustness_report(npn, n_bootstraps=30, seed=42)
        assert r1.robustness_score == r2.robustness_score
