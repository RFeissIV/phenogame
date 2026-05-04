"""Tests for the data-induced EML game extension modules — using REAL USA-NPN data.

Replaces the previous ``rng.normal``-fabricated ``toy_df`` with real
USA-NPN observations from :func:`phenogame.fixtures.load_phenology_table`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from phenogame import (
    EMLTreeRegressor,
    MultiAgentDecisionGame,
    build_game_audit_certificate,
    compile_eml_payoff_game,
    counterfactual_timing_test,
    decision_to_jsonld,
    fit_eml_payoff_model,
    load_npn_csv,
)
from phenogame.equilibrium import certify_provable
from phenogame.fixtures import load_phenology_table


_FEATURES = ["Latitude", "Longitude", "Elevation_in_Meters", "Mean_First_Yes_Year"]
_TARGET = "Mean_First_Yes_DOY"


def _real_action_scenario_table(species: str = "peach", n_max: int = 80) -> pd.DataFrame:
    """Real USA-NPN data reshaped so 'action' = state, 'scenario' = year,
    x1 = lat, x2 = lon, payoff = first-yes DOY. Trimmed to ``n_max`` rows so
    the small-game compile path exercises a meaningful number of strategies."""
    df = load_phenology_table(species, feature_cols=_FEATURES,
                              target_col=_TARGET, scenario_col="State",
                              min_rows=20)
    out = pd.DataFrame({
        "action": df["State"].astype(str),
        "scenario": df["Mean_First_Yes_Year"].astype(int).astype(str),
        "x1": df["Latitude"].astype(float),
        "x2": df["Longitude"].astype(float),
        "payoff": df["Mean_First_Yes_DOY"].astype(float),
    })
    return out.head(n_max).reset_index(drop=True)


def test_eml_tree_regressor_predicts_real():
    df = _real_action_scenario_table()
    model = EMLTreeRegressor(n_features=8, seed=2).fit(df[["x1", "x2"]], df["payoff"])
    pred = model.predict(df[["x1", "x2"]])
    assert pred.shape == (len(df),)
    assert np.all(np.isfinite(pred))


def test_fit_eml_payoff_model_summary_real():
    df = _real_action_scenario_table()
    fit = fit_eml_payoff_model(df, ["x1", "x2"], "payoff", n_features=8)
    assert fit.n_rows == len(df)
    assert "EML-tree payoff fit" in fit.summary()


def test_compile_eml_payoff_game_and_certify_real():
    df = _real_action_scenario_table()
    game = compile_eml_payoff_game(
        df, strategy_column="action", scenario_column="scenario",
        feature_columns=["x1", "x2"], target_column="payoff", n_eml_features=8,
    )
    assert game.payoff_matrix.matrix.shape[0] >= 2
    cert = certify_provable(game.payoff_matrix, iterations=20, seed=3)
    assert cert.iterations == 20


def test_counterfactual_timing_test_runs():
    """Uses bundled real USA-NPN grape CSV via load_npn_csv (no fabrication)."""
    npn = load_npn_csv()
    result = counterfactual_timing_test(npn, offsets=[-3, 0, 3], seed=2)
    assert len(result.offsets) == 3
    assert "Counterfactual" in result.summary()


def test_jsonld_and_audit_certificate():
    """Uses bundled real USA-NPN grape CSV via load_npn_csv (no fabrication)."""
    npn = load_npn_csv()
    result = counterfactual_timing_test(npn, offsets=[0], seed=2)
    js = decision_to_jsonld(result, crop="grape", phenophase="Ripe fruits")
    assert js["@type"].endswith("DecisionCertificate")
    cert = build_game_audit_certificate(
        result, crop="grape", phenophase="Ripe fruits", assumptions={"mode": "test"}
    )
    assert "GAME AUDIT CERTIFICATE" in cert.summary()


def test_multiagent_container_validates():
    """Algebraic edge case (not data): validates the multi-agent payoff
    tensor container with deterministic zero / one tensors. No fabricated
    observations are involved."""
    g = MultiAgentDecisionGame(
        agents=["grower", "weather"],
        action_labels={"grower": ["early", "late"], "weather": ["cool", "warm"]},
        payoff_tensors={
            "grower": np.zeros((2, 2)),
            "weather": np.ones((2, 2)),
        },
    )
    g.validate()
    assert "Multi-agent" in g.summary()
