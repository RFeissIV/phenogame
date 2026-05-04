"""Tests for phenogame.binary_choice — using REAL USA-NPN data.

The ``rng.normal`` synthetic-data fabrications used in the previous version
of this file have been removed. All score inputs come from
:func:`phenogame.fixtures.real_score_matrix_from_models`, which fits the 13
EML families on real USA-NPN bootstrap resamples.

A small number of tests still use hand-constructed algebraic edge cases
(deliberate cyclic / inconsistent preference matrices) for the express
purpose of verifying that the constraint detectors trigger. Those are
labelled in their docstrings; they are not substitute observations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phenogame.binary_choice import (
    pairwise_win_matrix,
    preference_probabilities,
    check_binary_choice_constraints,
    triangle_violations,
    stochastic_choice_summary,
)
from phenogame.fixtures import real_score_matrix_from_models


# ─────────────────────────────────────────────────────────────────────────
# Real-data fixtures
# ─────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def real_scores_peach() -> pd.DataFrame:
    return real_score_matrix_from_models("peach", n_repetitions=12, seed=0)


@pytest.fixture(scope="module")
def real_scores_grape() -> pd.DataFrame:
    return real_score_matrix_from_models("grape", n_repetitions=12, seed=0)


# ─────────────────────────────────────────────────────────────────────────
# pairwise_win_matrix
# ─────────────────────────────────────────────────────────────────────────

def test_pairwise_win_matrix_shape_and_diag(real_scores_peach):
    W = pairwise_win_matrix(real_scores_peach)
    assert W.shape == (13, 13)
    assert np.all(np.diag(W.to_numpy()) == 0.0)


def test_pairwise_win_matrix_total_count(real_scores_peach):
    W = pairwise_win_matrix(real_scores_peach).to_numpy()
    n = real_scores_peach.shape[1]
    n_rows = real_scores_peach.shape[0]
    expected = (np.ones((n, n)) - np.eye(n)) * n_rows
    actual = W + W.T - np.diag(np.diag(W + W.T))
    assert np.allclose(actual, expected)


def test_pairwise_win_matrix_rejects_nan():
    df = pd.DataFrame({"a": [1.0, np.nan], "b": [0.0, 1.0]})
    with pytest.raises(ValueError):
        pairwise_win_matrix(df)


def test_pairwise_win_matrix_requires_dataframe():
    with pytest.raises(TypeError):
        pairwise_win_matrix(np.zeros((3, 3)))


# ─────────────────────────────────────────────────────────────────────────
# preference_probabilities
# ─────────────────────────────────────────────────────────────────────────

def test_preference_probabilities_complementarity(real_scores_peach):
    W = pairwise_win_matrix(real_scores_peach)
    P = preference_probabilities(W)
    n = P.shape[0]
    off = np.where(np.eye(n, dtype=bool), 0.0, P + P.T - 1.0)
    assert np.max(np.abs(off)) < 1e-12


def test_preference_probabilities_no_nan_inf(real_scores_grape):
    W = pairwise_win_matrix(real_scores_grape)
    P = preference_probabilities(W)
    assert np.all(np.isfinite(P))
    assert np.all((P >= 0.0) & (P <= 1.0))


def test_preference_probabilities_handles_empty_pairs():
    """Algebraic edge case (not data): a zero win-matrix should map to all-0.5."""
    W = np.zeros((3, 3))
    P = preference_probabilities(W)
    assert np.allclose(P, 0.5)


def test_preference_probabilities_rejects_nonsquare():
    with pytest.raises(ValueError):
        preference_probabilities(np.zeros((3, 4)))


# ─────────────────────────────────────────────────────────────────────────
# check_binary_choice_constraints — REAL pairwise probabilities
# ─────────────────────────────────────────────────────────────────────────

def test_constraints_on_real_data(real_scores_peach):
    P = preference_probabilities(pairwise_win_matrix(real_scores_peach))
    audit = check_binary_choice_constraints(P)
    assert audit["complementarity_ok"]
    assert audit["nonnegativity_ok"]
    assert isinstance(audit["triangle_ok"], bool)
    assert audit["all_passed"] == (
        audit["complementarity_ok"] and audit["nonnegativity_ok"]
        and audit["triangle_ok"]
    )


def test_constraints_detect_complementarity_failure():
    """Algebraic edge case (not data): a deliberately inconsistent matrix
    used to confirm the detector triggers."""
    P = np.array([
        [0.5, 0.7, 0.7],
        [0.7, 0.5, 0.7],
        [0.7, 0.7, 0.5],
    ])
    audit = check_binary_choice_constraints(P)
    assert not audit["complementarity_ok"]
    assert audit["max_complementarity_error"] > 0.1


def test_triangle_violation_caught_in_constructed_cycle():
    """Algebraic edge case (not data): a deliberate cyclic preference matrix
    used to confirm the triangle detector triggers."""
    P = np.array([
        [0.5, 0.9, 0.1],
        [0.1, 0.5, 0.9],
        [0.9, 0.1, 0.5],
    ])
    audit = check_binary_choice_constraints(P)
    assert audit["complementarity_ok"]
    assert not audit["triangle_ok"]
    assert audit["n_triangle_violations"] > 0
    df = triangle_violations(P)
    assert len(df) > 0
    assert (df["sum"] > 2.0).all()


# ─────────────────────────────────────────────────────────────────────────
# stochastic_choice_summary
# ─────────────────────────────────────────────────────────────────────────

def test_stochastic_choice_summary_structure_real(real_scores_peach):
    P = preference_probabilities(pairwise_win_matrix(real_scores_peach))
    labels = list(real_scores_peach.columns)
    summary = stochastic_choice_summary(P, labels=labels)
    assert summary["n_alternatives"] == 13
    assert len(summary["borda_scores"]) == 13
    assert len(summary["ranking"]) == 13
    assert summary["constraints"]["all_passed"] in (True, False)
    assert sorted(summary["ranking"]) == sorted(labels)


def test_stochastic_choice_summary_labels_mismatch_raises():
    P = np.eye(4)
    with pytest.raises(ValueError):
        stochastic_choice_summary(P, labels=["a", "b"])
