"""Tests for phenogame.joint_correlation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phenogame.joint_correlation import (
    joint_correlation_residual_normalized,
    joint_correlation_residual_raw,
)


# ────────────────────────────────────────────────────────────────────────
# Properties of normalized residual
# ────────────────────────────────────────────────────────────────────────

def test_independent_distribution_has_zero_residual():
    """If q = p_row × p_col, normalised residual must be 0 (within float tol)."""
    p_row = np.array([0.4, 0.6]).reshape(-1, 1)
    p_col = np.array([0.7, 0.3]).reshape(1, -1)
    q = p_row @ p_col
    rho = joint_correlation_residual_normalized(q)
    np.testing.assert_allclose(rho, 0.0, atol=1e-12)


def test_perfectly_correlated_has_residual_near_one():
    """Diagonal joint distribution (perfect correlation) achieves max
    residual of 1 under our normalisation."""
    q = np.eye(2) * 0.5
    rho = joint_correlation_residual_normalized(q)
    np.testing.assert_allclose(rho, 1.0, atol=1e-10)


def test_residual_in_unit_interval_random():
    """Random valid joint distributions: residual ∈ [0, 1]."""
    rng = np.random.default_rng(0)
    for _ in range(100):
        m = int(rng.integers(2, 6))
        n = int(rng.integers(2, 6))
        q = rng.uniform(size=(m, n))
        q = q / q.sum()
        rho = joint_correlation_residual_normalized(q)
        assert 0.0 <= rho <= 1.0


def test_normalization_makes_dimensions_comparable():
    """Audit F8: a 'similarly correlated' distribution should give a similar
    normalised value across dimensions."""
    # Two-thirds-on-diagonal distributions of different sizes.
    q22 = np.array([[0.4, 0.1], [0.1, 0.4]])  # mass on diagonal
    q33 = np.diag([0.3, 0.3, 0.3]) + 0.1 / 9 * np.ones((3, 3))
    q33 = q33 - np.diag([0.3 / 9, 0.3 / 9, 0.3 / 9])  # rebalance to sum 1
    q33 = q33 / q33.sum()
    rho22 = joint_correlation_residual_normalized(q22)
    rho33 = joint_correlation_residual_normalized(q33)
    # Both should be moderately high (>0.3) and within similar ballpark
    # despite different dimensions.
    assert rho22 > 0.3
    assert rho33 > 0.3


# ────────────────────────────────────────────────────────────────────────
# Validation
# ────────────────────────────────────────────────────────────────────────

def test_rejects_non_distribution():
    with pytest.raises(ValueError):
        joint_correlation_residual_normalized(np.array([[0.5, 0.5], [0.5, 0.5]]))


def test_rejects_negative_entries():
    with pytest.raises(ValueError):
        joint_correlation_residual_normalized(np.array([[-0.1, 0.6], [0.3, 0.2]]))


def test_rejects_nan_inf():
    q = np.full((2, 2), 0.25)
    q[0, 0] = np.nan
    with pytest.raises(ValueError):
        joint_correlation_residual_normalized(q)


def test_rejects_non_2d():
    with pytest.raises(ValueError):
        joint_correlation_residual_normalized(np.array([1.0]))


# ────────────────────────────────────────────────────────────────────────
# DataFrame input
# ────────────────────────────────────────────────────────────────────────

def test_accepts_dataframe():
    df = pd.DataFrame(np.eye(3) / 3.0)
    rho = joint_correlation_residual_normalized(df)
    np.testing.assert_allclose(rho, 1.0, atol=1e-10)


# ────────────────────────────────────────────────────────────────────────
# Raw residual reference
# ────────────────────────────────────────────────────────────────────────

def test_raw_matches_audit_doc_definition():
    """Audit doc Section 3.4: raw residual = sum(|q − p_row × p_col|)."""
    q = np.array([[0.4, 0.1], [0.1, 0.4]])
    raw = joint_correlation_residual_raw(q)
    p_row = q.sum(axis=1, keepdims=True)
    p_col = q.sum(axis=0, keepdims=True)
    expected = float(np.sum(np.abs(q - p_row @ p_col)))
    np.testing.assert_allclose(raw, expected)


def test_normalized_smaller_than_raw_for_3x3_or_larger():
    """For min(m,n) ≥ 2 the normaliser is ≥ 1, so normalised ≤ raw."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.uniform(size=(3, 3))
        q = q / q.sum()
        raw = joint_correlation_residual_raw(q)
        norm = joint_correlation_residual_normalized(q)
        assert norm <= raw + 1e-12
