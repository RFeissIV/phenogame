"""Tests for phenogame.bsc_game_bridge — using REAL USA-NPN data.

Replaces the previous ``rng.normal`` / ``_make_clean_P`` synthetic helper
with real pairwise preference matrices computed from
:func:`phenogame.fixtures.real_score_matrix_from_models`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phenogame.binary_choice import (
    pairwise_win_matrix,
    preference_probabilities,
)
from phenogame.bsc_game_bridge import (
    coalition_value_from_pairwise,
    coalition_game,
    marginal_contribution,
    max_marginal_contributions,
    binary_choice_game_certificate,
)
from phenogame.fixtures import real_score_matrix_from_models


# ─────────────────────────────────────────────────────────────────────────
# Real-data fixtures
# ─────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def real_P_4() -> np.ndarray:
    """REAL 4×4 pairwise probabilities: take the first 4 of the 13 EML
    families on real peach bootstrap repetitions."""
    sm = real_score_matrix_from_models("peach", n_repetitions=20, seed=0)
    sm = sm.iloc[:, :4]  # first 4 columns only (n=4 keeps 2^n enumeration cheap)
    return preference_probabilities(pairwise_win_matrix(sm))


@pytest.fixture(scope="module")
def real_P_13() -> np.ndarray:
    """REAL full 13×13 pairwise preferences from real peach data."""
    sm = real_score_matrix_from_models("peach", n_repetitions=30, seed=0)
    return preference_probabilities(pairwise_win_matrix(sm))


# ─────────────────────────────────────────────────────────────────────────
# coalition_value_from_pairwise
# ─────────────────────────────────────────────────────────────────────────

def test_coalition_value_empty_is_zero(real_P_4):
    assert coalition_value_from_pairwise(real_P_4, set()) == 0.0


def test_coalition_value_full_set_equals_zero(real_P_4):
    n = real_P_4.shape[0]
    assert coalition_value_from_pairwise(real_P_4, list(range(n))) == 0.0


def test_coalition_value_singleton_real(real_P_4):
    val = coalition_value_from_pairwise(real_P_4, [0])
    n = real_P_4.shape[0]
    expected = float(sum(real_P_4[0, j] for j in range(n) if j != 0))
    assert abs(val - expected) < 1e-12


def test_coalition_value_invalid_index_raises(real_P_4):
    with pytest.raises(ValueError):
        coalition_value_from_pairwise(real_P_4, [99])


# ─────────────────────────────────────────────────────────────────────────
# coalition_game
# ─────────────────────────────────────────────────────────────────────────

def test_coalition_game_enumerates_2n(real_P_4):
    v = coalition_game(real_P_4)
    assert len(v) == 2 ** 4
    assert v[frozenset()] == 0.0


def test_coalition_game_refuses_too_large_n():
    """Algebraic edge case: a 25×25 zeros matrix is used purely to verify the
    enumeration limiter triggers. Not data, by construction."""
    big = np.full((25, 25), 0.5)
    with pytest.raises(ValueError):
        coalition_game(big)


# ─────────────────────────────────────────────────────────────────────────
# marginal contributions
# ─────────────────────────────────────────────────────────────────────────

def test_marginal_contribution_default_uses_N_minus_i(real_P_4):
    v = coalition_game(real_P_4)
    N = list(range(4))
    m = marginal_contribution(v, 0, N)
    expected = v[frozenset(N)] - v[frozenset(N) - {0}]
    assert abs(m - expected) < 1e-12


def test_max_marginal_contributions_keys_match_N(real_P_4):
    v = coalition_game(real_P_4)
    N = list(range(4))
    m_max = max_marginal_contributions(v, N)
    assert set(m_max.keys()) == set(N)
    for val in m_max.values():
        assert np.isfinite(val)


# ─────────────────────────────────────────────────────────────────────────
# binary_choice_game_certificate
# ─────────────────────────────────────────────────────────────────────────

def test_certificate_fields_present_and_finite_real(real_P_4):
    cert = binary_choice_game_certificate(real_P_4)
    assert cert["n_alternatives"] == 4
    assert isinstance(cert["constraints"], dict)
    assert np.isfinite(cert["v_N"])
    assert np.isfinite(cert["sum_max_marginals"])
    assert np.isfinite(cert["lhs"])
    assert np.isfinite(cert["rhs"])
    assert isinstance(cert["inequality_holds"], bool)
    assert "note" in cert


def test_certificate_uniform_weights_lhs_matches_offdiag_sum(real_P_4):
    cert = binary_choice_game_certificate(real_P_4)
    expected_lhs = float(real_P_4.sum() - np.trace(real_P_4))
    assert abs(cert["lhs"] - expected_lhs) < 1e-9


def test_certificate_refuses_too_large_n():
    """Algebraic edge case: a 15×15 zeros matrix used purely to verify the
    enumeration limiter triggers. Not data, by construction."""
    big = np.full((15, 15), 0.5)
    with pytest.raises(ValueError):
        binary_choice_game_certificate(big, max_n=12)


def test_certificate_holds_on_real_13_family_data(real_P_13):
    """Real 13×13 preference matrix from real peach data — the inequality
    the theorem-valid Gilboa-Monderer u=0 diagnostic inequality holds.
    External-flow coalition summaries are reported separately."""
    cert = binary_choice_game_certificate(real_P_13)
    assert cert["inequality_holds"]
    assert cert["constraints"]["complementarity_ok"]
    assert cert["constraints"]["nonnegativity_ok"]
