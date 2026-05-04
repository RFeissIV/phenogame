"""
PhenoGame Example 1: Four-Species Asymmetric Loss Comparison.

Each species has different cost asymmetry reflecting real economics:
  Red maple  - monitoring (low asymmetry, early~late cost)
  Peach      - frost protection (extreme: late = crop loss)
  Apple      - spray timing (moderate asymmetry)
  Wine grape - harvest (both directions costly: early=low sugar, late=rot)

The key result: species with different cost structures get DIFFERENT
recommendations. The framework discriminates based on economics.

Uses MAD (median absolute deviation) for robust variability estimation.
Data: USA-NPN site phenometrics. Not a new proof.
"""

import warnings
from phenogame import load_npn_csv, run_phenology_game

# Species-specific cost ratios reflecting actual grower economics
species_config = {
    "Red maple": {
        "file": "npn_red_maple.csv",
        "phenophase": "Breaking leaf buds",
        "cost_early": 1.0,
        "cost_late": 1.5,
        "rationale": "Monitoring/phenology tracking. Low asymmetry.",
    },
    "Peach": {
        "file": "npn_peach.csv",
        "phenophase": "Open flowers",
        "cost_early": 1.0,
        "cost_late": 5.0,
        "rationale": "Frost protection. Late = crop loss. High asymmetry.",
    },
    "Apple": {
        "file": "npn_apple.csv",
        "phenophase": "Open flowers",
        "cost_early": 1.0,
        "cost_late": 2.5,
        "rationale": "Spray/frost timing. Moderate asymmetry.",
    },
    "Wine grape": {
        "file": "npn_grape.csv",
        "phenophase": "Ripe fruits",
        "cost_early": 2.0,
        "cost_late": 3.0,
        "rationale": "Harvest. Early=low sugar, late=rot. Both costly.",
    },
}

print("=" * 80)
print("  FOUR-SPECIES ASYMMETRIC LOSS COMPARISON")
print("  Payoff: U = 1 - loss/max; loss = c_early*max(0,actual-target)")
print("                                     + c_late*max(0,target-actual)")
print("  Variability: MAD (median absolute deviation)")
print("  Data: USA-NPN site phenometrics")
print("=" * 80)
print()

results = {}
for name, cfg in species_config.items():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        npn = load_npn_csv(cfg["file"])
        r = run_phenology_game(
            npn,
            phenophase=cfg["phenophase"],
            cost_early=cfg["cost_early"],
            cost_late=cfg["cost_late"],
            seed=42,
        )
    results[name] = (r, cfg)
    print(f"--- {name}: {cfg['phenophase']} ---")
    print(f"  {cfg['rationale']}")
    print(f"  Cost ratio: early={cfg['cost_early']}, late={cfg['cost_late']}")
    print(f"  n={r.n_obs}, strategies={r.strategies}")
    print(f"  Recommended: {r.recommended_strategy}")
    print(f"  CCE gap: {r.cce_gap:.4f}, Certified: {r.certified}")
    print()

# Comparative table
print("=" * 80)
print("  COMPARATIVE TABLE")
print("=" * 80)
print()
print(f"{'Species':15s} {'Phenophase':25s} {'c_e':>4s} {'c_l':>4s} "
      f"{'n':>5s} {'Rec':>10s} {'Gap':>8s} {'Cert':>5s}")
print("-" * 78)
for name, (r, cfg) in results.items():
    print(f"{name:15s} {cfg['phenophase']:25s} "
          f"{cfg['cost_early']:4.1f} {cfg['cost_late']:4.1f} "
          f"{r.n_obs:5d} {r.recommended_strategy:>10s} "
          f"{r.cce_gap:8.4f} {'Y' if r.certified else 'N':>5s}")

print()
print("  The framework recommends DIFFERENT strategies for species")
print("  with different cost asymmetries. This is not a tautology —")
print("  it reflects the economics: peach growers should act early")
print("  because frost kills the crop, while red maple monitoring")
print("  has no urgency penalty.")
print()
print("  Data-induced epsilon-CCE certificates, NOT a new proof.")
