"""Tests for phenogame.regret_transforms."""

from __future__ import annotations

import numpy as np
import pytest

from phenogame.regret_transforms import (
    psi_standard, psi_exponential, psi_softplus, psi_eml,
    TRANSFORM_REGISTRY, list_transforms,
)


@pytest.fixture
def regrets():
    """Simple regret vector exercising negative, zero, positive entries."""
    return np.array([-2.0, -0.5, 0.0, 0.5, 2.0])


# ────────────────────────────────────────────────────────────────────────
# psi_standard — Hart-Mas-Colell positive-part regret matching
# ────────────────────────────────────────────────────────────────────────

def test_psi_standard_clamps_at_zero(regrets):
    out = psi_standard(regrets)
    assert out[0] == 0.0
    assert out[1] == 0.0
    assert out[2] == 0.0
    assert out[3] == 0.5
    assert out[4] == 2.0


def test_psi_standard_is_strict_rm():
    """Strict RM ⇒ zero regret ⇒ zero weight."""
    out = psi_standard(np.zeros(4))
    assert np.all(out == 0.0)


# ────────────────────────────────────────────────────────────────────────
# psi_exponential — Hedge kernel
# ────────────────────────────────────────────────────────────────────────

def test_psi_exponential_strictly_positive_finite(regrets):
    out = psi_exponential(regrets)
    assert np.all(out > 0.0)
    assert np.all(np.isfinite(out))


def test_psi_exponential_clipped_at_50(regrets):
    """Large regrets must not overflow."""
    out = psi_exponential(np.array([10000.0, -10000.0]))
    assert np.all(np.isfinite(out))


def test_psi_exponential_temperature_monotone():
    """Larger τ ⇒ flatter distribution."""
    R = np.array([0.0, 1.0, 2.0])
    p_sharp = psi_exponential(R, tau=0.5)
    p_flat = psi_exponential(R, tau=2.0)
    # Higher τ ⇒ ratio closer to 1
    sharp_ratio = p_sharp[2] / p_sharp[0]
    flat_ratio = p_flat[2] / p_flat[0]
    assert flat_ratio < sharp_ratio


# ────────────────────────────────────────────────────────────────────────
# psi_softplus — audit F6: always positive
# ────────────────────────────────────────────────────────────────────────

def test_psi_softplus_always_positive_F6(regrets):
    """Audit finding F6: softplus assigns positive weight to every action,
    including those with arbitrarily-negative regret."""
    out = psi_softplus(regrets)
    assert np.all(out > 0.0)
    out_negative = psi_softplus(np.array([-100.0, -50.0, -10.0]))
    assert np.all(out_negative > 0.0)


def test_psi_softplus_finite_at_extremes():
    out = psi_softplus(np.array([10000.0, -10000.0]))
    assert np.all(np.isfinite(out))


def test_psi_softplus_log_2_at_zero():
    """softplus(0) = log(2) ≈ 0.693 — matches audit doc Section 3.6 Table."""
    out = psi_softplus(np.array([0.0]))
    np.testing.assert_allclose(out[0], np.log(2.0), atol=1e-12)


# ────────────────────────────────────────────────────────────────────────
# psi_eml — the audited EML-RML transform
# ────────────────────────────────────────────────────────────────────────

def test_psi_eml_clamps_at_zero(regrets):
    """EML clamps at zero per audit doc construction."""
    out = psi_eml(regrets)
    assert np.all(out >= 0.0)


def test_psi_eml_value_at_zero():
    """Audit Section 3.6 Table: ψ_EML(0) = 1 − log(2) ≈ 0.307."""
    out = psi_eml(np.array([0.0]))
    np.testing.assert_allclose(out[0], 1.0 - np.log(2.0), atol=1e-12)


def test_psi_eml_positive_for_positive_regret():
    out = psi_eml(np.array([0.5, 1.0, 2.0]))
    assert np.all(out > 0.0)
    # Monotone in regret
    assert out[0] < out[1] < out[2]


def test_psi_eml_zero_for_large_negative_regret():
    """Audit Section 3.6 Table: large negative ⇒ pmax clamp ⇒ 0."""
    out = psi_eml(np.array([-30.0, -50.0]))
    np.testing.assert_array_equal(out, np.array([0.0, 0.0]))


def test_psi_eml_matches_existing_private_transform():
    """ψ_eml(R, τ) must match phenogame.equilibrium._eml_regret_transform
    bit-for-bit (or within float tolerance) for the same inputs — the v4
    public ψ_eml is a refactor, not a new transform."""
    from phenogame.equilibrium import _eml_regret_transform
    rng = np.random.default_rng(0)
    R = rng.normal(size=10) * 5.0
    tau = 1.3
    out_public = psi_eml(R, tau)
    out_private = _eml_regret_transform(R, tau)
    np.testing.assert_allclose(out_public, out_private, atol=1e-9, rtol=1e-9)


def test_psi_eml_finite_at_extremes():
    out = psi_eml(np.array([10000.0, -10000.0]))
    assert np.all(np.isfinite(out))


# ────────────────────────────────────────────────────────────────────────
# Registry
# ────────────────────────────────────────────────────────────────────────

def test_registry_complete():
    for name in ["standard", "exp", "softplus", "eml"]:
        assert name in TRANSFORM_REGISTRY
        info = TRANSFORM_REGISTRY[name]
        for k in ["fn", "uses_tau", "no_regret_proof",
                  "is_strict_rm", "always_positive"]:
            assert k in info


def test_registry_correct_audit_facts():
    """Audit findings F6 and F9 must be encoded in the registry."""
    # F6: softplus is always positive and not strict RM
    assert TRANSFORM_REGISTRY["softplus"]["always_positive"] is True
    assert TRANSFORM_REGISTRY["softplus"]["is_strict_rm"] is False
    # F9: EML has no formal no-regret proof
    assert "empirical" in TRANSFORM_REGISTRY["eml"]["no_regret_proof"].lower()
    # standard has formal proof
    assert "Hart-Mas-Colell" in TRANSFORM_REGISTRY["standard"]["no_regret_proof"]


def test_list_transforms_canonical_order():
    assert list_transforms() == ["standard", "exp", "softplus", "eml"]
