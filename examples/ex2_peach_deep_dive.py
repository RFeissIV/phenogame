"""
PhenoGame Example 2: Peach Frost Protection Deep Dive.

Single-species analysis showing the full pipeline:
  1. Load NPN data
  2. Asymmetric loss game (late = 5x early cost)
  3. Bootstrap policy intervals
  4. Sensitivity to cost ratio changes
  5. Full robustness report

Peach open flowers is the best demo because the decision is
real and high-stakes: frost on open peach flowers destroys
the crop. Acting early wastes money. Acting late loses everything.
"""

import warnings
from phenogame import (
    load_npn_csv, run_phenology_game, robustness_report,
)

npn = load_npn_csv("npn_peach.csv")

print("=" * 60)
print("  PEACH FROST PROTECTION: ASYMMETRIC LOSS ANALYSIS")
print("=" * 60)
print()

# 1. Game with high asymmetry
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    game = run_phenology_game(
        npn, phenophase="Open flowers",
        cost_early=1.0, cost_late=5.0,
        seed=42,
    )

print(game.summary())
print()

# 2. Compare: what if we used symmetric loss?
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    symmetric = run_phenology_game(
        npn, phenophase="Open flowers",
        cost_early=1.0, cost_late=1.0,
        seed=42,
    )

print("-" * 60)
print("  SYMMETRIC vs ASYMMETRIC COMPARISON")
print("-" * 60)
print(f"  Symmetric (1:1): rec={symmetric.recommended_strategy}")
print(f"  Asymmetric (1:5): rec={game.recommended_strategy}")
print()
if symmetric.recommended_strategy != game.recommended_strategy:
    print("  The cost structure CHANGES the recommendation.")
    print("  This is the key insight: symmetric loss hides")
    print("  the real economic asymmetry of frost protection.")
else:
    print("  Same recommendation, but payoff structure differs.")
print()

# 3. Robustness
print("-" * 60)
print("  ROBUSTNESS REPORT")
print("-" * 60)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    report = robustness_report(
        npn, phenophase="Open flowers",
        n_bootstraps=100, seed=42,
        cost_early=1.0, cost_late=5.0,
    )
print(report.summary())
print()
print("  Data-induced epsilon-CCE certificate, NOT a new proof.")
