# PhenoGame Data-Induced EML Game Compiler Extension

## Purpose

This patch moves PhenoGame from a fixed-payoff decision demo toward a data-induced game compiler:

```text
data -> EML-tree payoff model -> finite decision game -> robust policy certificate
```

## What is new

1. EML-tree payoff learning from tabular biological/agronomic data.
2. Data-induced finite-game construction from learned payoff surfaces.
3. Counterfactual timing tests.
4. Bootstrap and sensitivity compatibility through existing robustness tools.
5. Ontology/KG-compatible JSON-LD exports.
6. Game audit certificates.
7. Multi-agent game data container for future grower/weather/market models.
8. Held-out validation utilities (`compare_baselines`, `train_test_split_indices`)
   that score the EML-tree model against `naive_mean`, `ridge_linear`, and
   (optional) `random_forest` baselines using RMSE, MAE, and R² on a
   deterministic test split.
9. `action_stability` — bootstrap-resampled recommendation tracking that
   reports how often the data-induced game's recommendation is preserved.
10. End-to-end reference demo at `examples/grape_eml_game_extension_demo.py`
    that emits four reproducible artifacts to `outputs/`:
    `grape_decision_certificate.json`, `grape_robustness_summary.csv`,
    `grape_counterfactual_table.csv`, and `grape_baseline_comparison.csv`.

## What is not claimed

- No new Nash theorem.
- No proof that EML-tree transforms satisfy no-regret learning.
- No validated agronomic recommendation engine.
- No enterprise-production deployment guarantee.
- The bundled grape demo uses a transparent **surrogate utility derived from real NPN observations**
  (`-|DOY - target| - 0.001*|AGDD - mean(AGDD)|`) so the pipeline can run on
  open data; it is not a validated value model.
- Action-stability frequencies are empirical, not coverage probabilities.

## Correct claim

PhenoGame is an alpha-stage scientific-computing framework that can learn nonlinear payoff/response surfaces from data, compile them into finite decision games, and produce uncertainty-aware decision certificates.
