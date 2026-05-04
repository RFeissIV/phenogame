"""Smoke test: the grape data-induced EML game extension demo runs
end-to-end and writes the four expected artifact files.

The demo is small enough on the bundled NPN grape data that this can run as a
real test.  We use a small bootstrap count to keep the test fast.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "grape_eml_game_extension_demo.py"


def _load_demo_module():
    spec = importlib.util.spec_from_file_location("grape_demo", EXAMPLE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load demo at {EXAMPLE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not EXAMPLE.exists(), reason="demo script missing")
def test_grape_demo_runs_and_writes_artifacts(tmp_path: Path):
    demo = _load_demo_module()
    paths = demo.run(tmp_path / "outputs", seed=0, n_bootstraps=30, eml_features=16)
    for label in ("certificate", "robustness", "counterfactual", "baselines"):
        assert label in paths
        assert paths[label].exists(), f"{label} artifact missing: {paths[label]}"
        assert paths[label].stat().st_size > 0

    cert = json.loads(paths["certificate"].read_text())
    # Certificate must include the audit-required sections.
    for key in ("crop", "phenophase", "assumptions", "limitations",
                "eml_induced_game", "baseline_comparison", "counterfactual"):
        assert key in cert
    assert cert["crop"] == "grape"
    assert cert["phenophase"] == "Ripe fruits"
    # Honest-disclosure check: the surrogate-utility caveat must be in plain text.
    assert "surrogate" in cert["assumptions"]["payoff_definition"].lower()
