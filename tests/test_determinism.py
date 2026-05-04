"""Determinism contract: real-data pipeline must produce identical outputs.

This test pins the exact numerical outputs of the canonical real-data
pipeline (USA-NPN grape, default seeds) so that any unintended drift in
algorithmic randomness, ordering, library versions, or numerical paths
fails loudly.

The randomness in PhenoGame is fully algorithmic and fully seeded:

  - Hedge action sampling in :func:`phenogame.equilibrium.certify_provable`
    and :func:`phenogame.equilibrium.certify_empirical`
  - Random feature scales in :class:`phenogame.eml_tree.EMLTreeRegressor`
  - Bootstrap row-index sampling in robustness / validation / fixtures

None of these fabricate data. All accept a ``seed`` parameter, and as of
this version every library default is pinned (``seed=0`` instead of
``seed=None``). This test enforces that contract.

Failure modes this catches:

  - Removing ``np.random.default_rng(seed)`` and using a global RNG
  - Switching algorithm internals in a non-equivalent way
  - A scipy / numpy / pandas upgrade that changes deterministic ordering
  - Accidental introduction of a non-seeded random call

If a legitimate change makes these numbers shift, update the pinned
values **with a commit message that explains why**.
"""

from __future__ import annotations

import numpy as np

import phenogame as pg
from phenogame.fixtures import load_phenology_table


# Canonical pipeline parameters — match the bundled demo defaults.
_FEATURES = ["Latitude", "Longitude", "Elevation_in_Meters", "Mean_First_Yes_Year"]
_TARGET = "Mean_First_Yes_DOY"
_SCENARIO = "State"
_SPECIES = "grape"
_BASE_SEED = 42        # demo's default base_seed for family fitting
_CCE_SEED = 1          # demo's default Hedge seed
_CCE_ITERS = 2000

# Pinned values — observed at v0.3.0 with the bundled USA-NPN grape CSV
# (121 clean rows, 4 states: CA, MA, NY, OR).
EXPECTED = {
    "clean_rows": 121,
    "payoff_shape": (13, 4),
    "scenario_labels": ["CA", "MA", "NY", "OR"],
    "top_family_id": "F05",
    "top_family_rmse": 51.0526571438,
    "minimax_value": 126.3543685275,
    "minimax_dominant": "F06",
    "cce_gap": 0.0109880801,
    "cce_epsilon": 0.0506453291,
    "cce_certified": True,
}

# Tolerance for floating-point comparisons. Tight enough to catch real drift,
# loose enough to absorb harmless float-instruction-order differences across
# CPUs that can happen at the LSB.
_RTOL = 1e-9
_ATOL = 1e-9


def _run_canonical_pipeline():
    df = load_phenology_table(_SPECIES, feature_cols=_FEATURES,
                              target_col=_TARGET, scenario_col=_SCENARIO,
                              min_rows=20)
    cg = pg.compile_game_from_data(
        df,
        target_col=_TARGET,
        feature_cols=_FEATURES,
        scenario_cols=[_SCENARIO],
        base_seed=_BASE_SEED,
    )
    mm = pg.zero_sum_minimax_equilibrium(cg.payoff_matrix)
    cce = pg.certify_provable(cg.payoff_matrix, iterations=_CCE_ITERS, seed=_CCE_SEED)
    return df, cg, mm, cce


def test_determinism_clean_row_count():
    df, _, _, _ = _run_canonical_pipeline()
    assert len(df) == EXPECTED["clean_rows"], (
        f"Clean grape row count drifted: got {len(df)}, "
        f"expected {EXPECTED['clean_rows']}. Bundled CSV may have been altered."
    )


def test_determinism_payoff_shape_and_scenarios():
    _, cg, _, _ = _run_canonical_pipeline()
    assert cg.payoff_matrix.matrix.shape == EXPECTED["payoff_shape"]
    assert list(cg.payoff_surface.columns) == EXPECTED["scenario_labels"]


def test_determinism_top_family_pinned():
    _, cg, _, _ = _run_canonical_pipeline()
    top = cg.family_ranking.iloc[0]
    assert top["family_id"] == EXPECTED["top_family_id"], (
        f"Top family drifted: got {top['family_id']}, "
        f"expected {EXPECTED['top_family_id']}."
    )
    np.testing.assert_allclose(
        top["train_rmse"], EXPECTED["top_family_rmse"],
        rtol=_RTOL, atol=_ATOL,
        err_msg="Top-family training RMSE drifted from pinned value.",
    )


def test_determinism_minimax_pinned():
    _, _, mm, _ = _run_canonical_pipeline()
    np.testing.assert_allclose(
        mm.game_value, EXPECTED["minimax_value"],
        rtol=_RTOL, atol=_ATOL,
        err_msg="Minimax game value drifted from pinned value.",
    )
    assert mm.dominant_strategy == EXPECTED["minimax_dominant"]


def test_determinism_cce_pinned():
    _, _, _, cce = _run_canonical_pipeline()
    np.testing.assert_allclose(
        cce.cce_gap, EXPECTED["cce_gap"],
        rtol=_RTOL, atol=_ATOL,
        err_msg="ε-CCE gap drifted from pinned value.",
    )
    np.testing.assert_allclose(
        cce.epsilon, EXPECTED["cce_epsilon"],
        rtol=_RTOL, atol=_ATOL,
        err_msg="ε-CCE bound drifted from pinned value.",
    )
    assert cce.is_certified == EXPECTED["cce_certified"]


def test_determinism_two_runs_byte_identical():
    """Same seeds → byte-identical payoff matrix (no hidden non-determinism)."""
    _, cg1, mm1, cce1 = _run_canonical_pipeline()
    _, cg2, mm2, cce2 = _run_canonical_pipeline()
    np.testing.assert_array_equal(cg1.payoff_matrix.matrix, cg2.payoff_matrix.matrix)
    assert mm1.game_value == mm2.game_value
    assert cce1.cce_gap == cce2.cce_gap
    np.testing.assert_array_equal(cce1.empirical_joint, cce2.empirical_joint)


def test_determinism_default_seed_is_pinned_not_none():
    """Library defaults must be deterministic. ``seed=None`` would be a
    contract violation (would silently draw from system entropy)."""
    import inspect
    from phenogame.equilibrium import certify_provable, certify_empirical

    for fn in [certify_provable, certify_empirical]:
        sig = inspect.signature(fn)
        seed_default = sig.parameters["seed"].default
        assert seed_default is not None, (
            f"{fn.__name__}: seed default must be a fixed integer "
            f"(got None — that would be non-deterministic by default)."
        )
        assert isinstance(seed_default, int), (
            f"{fn.__name__}: seed default should be an int, got {type(seed_default)}."
        )
