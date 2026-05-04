"""Tests for phenogame.certificate — using REAL USA-NPN data.

Replaces the previous ``rng.normal``-fabricated ``_toy_dataset`` with real
USA-NPN observations from :func:`phenogame.fixtures.load_phenology_table`.
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pandas as pd

from phenogame.binary_choice import (
    pairwise_win_matrix, preference_probabilities,
    check_binary_choice_constraints,
)
from phenogame.certificate import (
    DEFAULT_LIMITATIONS,
    generate_game_audit_certificate,
    export_certificate_json,
    export_certificate_markdown,
)
from phenogame.equilibrium import certify_provable
from phenogame.fixtures import load_phenology_table
from phenogame.game import zero_sum_minimax
from phenogame.game_compiler import compile_game_from_data


REQUIRED_KEYS = {
    "schema",
    "data_summary",
    "eml_family_scores",
    "preference_matrix",
    "binary_stochastic_choice_checks",
    "nature_scenarios",
    "payoff_matrix",
    "minimax_solution",
    "cce_solution",
    "bootstrap_stability",
    "sensitivity_tipping_points",
    "recommendation",
    "limitations",
}


_FEATURES = ["Latitude", "Longitude", "Elevation_in_Meters", "Mean_First_Yes_Year"]
_TARGET = "Mean_First_Yes_DOY"


def _real_grape_table() -> pd.DataFrame:
    return load_phenology_table(
        "grape", feature_cols=_FEATURES, target_col=_TARGET,
        scenario_col="State", min_rows=20,
    )


def test_generate_certificate_with_partial_inputs_returns_required_keys():
    cert = generate_game_audit_certificate({})
    assert REQUIRED_KEYS.issubset(set(cert.keys()))
    assert cert["limitations"] == DEFAULT_LIMITATIONS


def test_generate_certificate_full_pipeline_real():
    df = _real_grape_table()
    cg = compile_game_from_data(df, target_col=_TARGET,
                                feature_cols=_FEATURES,
                                scenario_cols=["State"])

    P = preference_probabilities(pairwise_win_matrix(cg.payoff_surface.T))
    audit = check_binary_choice_constraints(P)
    minimax = zero_sum_minimax(cg.payoff_matrix)
    cce = certify_provable(cg.payoff_matrix, iterations=200, seed=1)

    cert = generate_game_audit_certificate({
        "compiled_game": cg,
        "family_scores": cg.family_ranking,
        "preference_matrix": P,
        "constraint_audit": audit,
        "minimax": minimax,
        "cce": cce,
        "recommendation": {
            "selected_family": cg.family_ranking.iloc[0]["family_id"],
            "minimax_value": float(minimax.game_value),
        },
    })

    assert REQUIRED_KEYS.issubset(set(cert.keys()))
    assert cert["data_summary"]["target_column"] == _TARGET
    assert cert["payoff_matrix"]["shape"] == [13, len(cg.scenarios.labels)]
    assert cert["minimax_solution"]["game_value"] == float(minimax.game_value)
    assert cert["cce_solution"]["mode"] == "provable"
    assert cert["limitations"] == DEFAULT_LIMITATIONS


def test_export_certificate_json_roundtrips_real():
    df = _real_grape_table()
    cg = compile_game_from_data(df, target_col=_TARGET,
                                feature_cols=_FEATURES,
                                scenario_cols=["State"])
    cert = generate_game_audit_certificate({
        "compiled_game": cg,
        "family_scores": cg.family_ranking,
    })
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cert.json")
        export_certificate_json(cert, path)
        with open(path, "r") as f:
            loaded = json.load(f)
    assert REQUIRED_KEYS.issubset(set(loaded.keys()))


def test_export_certificate_markdown_writes_file_real():
    df = _real_grape_table()
    cg = compile_game_from_data(df, target_col=_TARGET,
                                feature_cols=_FEATURES,
                                scenario_cols=["State"])
    cert = generate_game_audit_certificate({
        "compiled_game": cg,
        "family_scores": cg.family_ranking,
    })
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cert.md")
        export_certificate_markdown(cert, path)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    assert "# PhenoGame audit certificate" in text
    assert "## Limitations" in text
    assert "## Recommendation" in text


def test_certificate_handles_inf_values_safely():
    """Algebraic edge case (not data): manually construct a result object
    carrying inf to confirm the JSON sanitiser converts it to null."""
    cert = generate_game_audit_certificate({
        "minimax": type("MX", (), {
            "game_value": float("inf"),
            "is_pure": False,
            "dominant_strategy": None,
            "strategy_labels": ["a"],
            "scenario_labels": ["b"],
            "farmer_strategy": np.array([1.0]),
            "nature_strategy": np.array([1.0]),
        })(),
    })
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cert.json")
        export_certificate_json(cert, path)
        with open(path) as f:
            loaded = json.load(f)
    assert loaded["minimax_solution"]["game_value"] is None
