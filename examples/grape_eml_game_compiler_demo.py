"""
examples/grape_eml_game_compiler_demo.py — REAL-DATA end-to-end demo.

Runs the forward-only EML game pipeline on **real USA-NPN site phenometrics
data** (no synthetic / fabricated rows anywhere in this script):

    Data (USA-NPN bundled CSV)
      → 13 EML candidate payoff families
      → binary EML-tree fitting
      → pairwise stochastic choice matrix P
      → finite game against Nature (13 × S)
      → minimax / ε-CCE
      → bootstrap + sensitivity
      → recommendation + audit certificate

Default species is **wine grape** (Vitis vinifera) using the bundled
``npn_grape.csv``. After USA-NPN ``-9999`` sentinel filtering this gives
121 clean rows. Pass ``--species`` to switch; ``red_maple`` gives the
largest sample (~9,200 clean rows).

Provenance: the bundled CSVs are direct USA-NPN site phenometrics exports
filtered to 2010-2020. They are byte-identical to the rows contained in
the user-supplied ``datasheet_*.zip`` ZIP archives. Both forms are real
data; this script uses the bundled form because it is shipped with the
package.
"""

from __future__ import annotations

import argparse
import os
import tempfile

import numpy as np
import pandas as pd

from phenogame.binary_choice import (
    pairwise_win_matrix,
    preference_probabilities,
    check_binary_choice_constraints,
)
from phenogame.bsc_game_bridge import binary_choice_game_certificate
from phenogame.certificate import (
    generate_game_audit_certificate,
    export_certificate_json,
    export_certificate_markdown,
)
from phenogame.equilibrium import certify_provable
from phenogame.fixtures import available_species, load_phenology_table
from phenogame.game import zero_sum_minimax
from phenogame.game_compiler import compile_game_from_data


def real_bootstrap(df: pd.DataFrame,
                   feature_cols: list, target_col: str, scenario_col: str,
                   n_boot: int = 10, seed: int = 0) -> dict:
    """Bootstrap selection-frequency on REAL data (resamples row indices only,
    never fabricates values)."""
    rng = np.random.default_rng(seed)
    counts: dict = {}
    n = len(df)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        sub = df.iloc[idx].reset_index(drop=True)
        cg = compile_game_from_data(
            sub, target_col=target_col,
            feature_cols=feature_cols, scenario_cols=[scenario_col],
            base_seed=42 + b,
        )
        winner = cg.family_ranking.iloc[0]["family_id"]
        counts[winner] = counts.get(winner, 0) + 1
    n_total = sum(counts.values())
    freq = {k: v / n_total for k, v in counts.items()}
    modal = max(freq, key=freq.get)
    return {
        "n_bootstraps": n_boot,
        "winner_frequency": freq,
        "modal_family": modal,
        "modal_frequency": freq[modal],
    }


def real_sensitivity(df: pd.DataFrame,
                     feature_cols: list, target_col: str,
                     scenario_col: str) -> dict:
    """Re-run pipeline on REAL data with different model-randomness seeds."""
    winners = []
    for s in [1, 7, 13, 21, 42]:
        cg = compile_game_from_data(
            df, target_col=target_col,
            feature_cols=feature_cols, scenario_cols=[scenario_col],
            base_seed=s,
        )
        winners.append(cg.family_ranking.iloc[0]["family_id"])
    distinct = sorted(set(winners))
    return {
        "winners_by_seed": winners,
        "distinct_winners": distinct,
        "stable": len(distinct) == 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", default="grape",
                        choices=available_species(),
                        help=f"Bundled USA-NPN species (default: grape). "
                             f"Choices: {available_species()}")
    parser.add_argument("--out", default=None,
                        help="Output directory for certificate artifacts. "
                             "Defaults to a fresh tempdir.")
    parser.add_argument("--n-boot", type=int, default=10,
                        help="Bootstrap repetitions on real data (default 10).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Master seed controlling EML-family fitting and "
                             "bootstrap resamples. Pipeline is bit-reproducible "
                             "given a fixed (--seed, --n-boot, --species). "
                             "Default: 42.")
    parser.add_argument("--cce-seed", type=int, default=1,
                        help="Seed for Hedge no-regret dynamics in the ε-CCE "
                             "certifier. Default: 1.")
    args = parser.parse_args()

    print("=" * 72)
    print("PhenoGame: forward-only EML game pipeline (REAL USA-NPN data)")
    print("=" * 72)

    feature_cols = ["Latitude", "Longitude",
                    "Elevation_in_Meters", "Mean_First_Yes_Year"]
    target_col = "Mean_First_Yes_DOY"
    scenario_col = "State"

    # 1. Load REAL USA-NPN data.
    df = load_phenology_table(args.species,
                              feature_cols=feature_cols,
                              target_col=target_col,
                              scenario_col=scenario_col,
                              min_rows=20)
    print(f"\nLoaded REAL USA-NPN '{args.species}' data: "
          f"{df.shape[0]} clean rows × {df.shape[1]} columns")
    print(f"  States represented: {sorted(df[scenario_col].unique())}")
    print(f"  Year range: {int(df['Mean_First_Yes_Year'].min())} - "
          f"{int(df['Mean_First_Yes_Year'].max())}")
    print(f"  Predicting {target_col} from {feature_cols}")

    # 2-4. Pipeline → 13 × S payoff matrix.
    cg = compile_game_from_data(
        df, target_col=target_col,
        feature_cols=feature_cols,
        scenario_cols=[scenario_col],
        base_seed=args.seed,
    )
    print("\n--- compiled game (REAL data) ---")
    print(cg.summary())

    # 5. Pairwise stochastic choice matrix from per-scenario predictions.
    P = preference_probabilities(pairwise_win_matrix(cg.payoff_surface.T))
    audit = check_binary_choice_constraints(P)
    print("\n--- binary stochastic choice audit ---")
    print(f"  complementarity_ok={audit['complementarity_ok']}, "
          f"nonnegativity_ok={audit['nonnegativity_ok']}, "
          f"triangle_ok={audit['triangle_ok']}, "
          f"all_passed={audit['all_passed']}")

    # 6. Cooperative-game diagnostic certificate.
    bsc_cert = binary_choice_game_certificate(P)
    print("\n--- diagnostic / necessary-condition audit (cooperative bridge) ---")
    print(f"  v(N) = {bsc_cert['v_N']:.4f}")
    print(f"  sum_i v_i* = {bsc_cert['sum_max_marginals']:.4f}")
    print(f"  inequality holds: {bsc_cert['inequality_holds']}")
    print(f"  slack = {bsc_cert['slack']:.4f}")

    # 7. Minimax (Algorithm vs Nature, zero-sum).
    minimax = zero_sum_minimax(cg.payoff_matrix)
    print("\n--- zero-sum minimax solution ---")
    print(f"  game value = {minimax.game_value:.4f}")
    print(f"  pure? {minimax.is_pure}")
    print(f"  dominant strategy: {minimax.dominant_strategy}")

    # 8. ε-CCE via Hedge.
    cce = certify_provable(cg.payoff_matrix, iterations=2000, seed=args.cce_seed)
    print("\n--- ε-CCE certificate (Hedge / provable) ---")
    print(f"  iterations = {cce.iterations}")
    print(f"  cce_gap = {cce.cce_gap:.6f}")
    print(f"  ε bound  = {cce.epsilon:.6f}")
    print(f"  certified ε-CCE: {cce.is_certified}")

    # 9. Robustness on REAL data.
    bs = real_bootstrap(df, feature_cols, target_col, scenario_col,
                        n_boot=args.n_boot, seed=args.seed)
    print("\n--- bootstrap selection frequency (REAL data resamples) ---")
    print(f"  modal family = {bs['modal_family']} "
          f"(freq={bs['modal_frequency']:.2f}, n_boot={bs['n_bootstraps']})")
    sa = real_sensitivity(df, feature_cols, target_col, scenario_col)
    print("\n--- sensitivity (random-seed sweep, REAL data) ---")
    print(f"  distinct winners across seeds: {sa['distinct_winners']}")
    print(f"  stable: {sa['stable']}")

    # 10. Recommendation.
    rec = {
        "species": args.species,
        "data_source": "USA-NPN site phenometrics (bundled real CSV)",
        "n_clean_rows": int(len(df)),
        "selected_eml_family": cg.family_ranking.iloc[0]["family_id"],
        "minimax_policy": dict(zip(minimax.strategy_labels,
                                    [float(p) for p in minimax.farmer_strategy])),
        "minimax_value": float(minimax.game_value),
        "cce_gap": float(cce.cce_gap),
        "cce_epsilon": float(cce.epsilon),
        "bootstrap_modal_family": bs["modal_family"],
        "bootstrap_modal_frequency": bs["modal_frequency"],
        "sensitivity_stable": bool(sa["stable"]),
        "binary_choice_constraints_passed": bool(audit["all_passed"]),
    }

    # 11. Audit certificate.
    cert = generate_game_audit_certificate({
        "compiled_game": cg,
        "family_scores": cg.family_ranking,
        "preference_matrix": pd.DataFrame(
            P,
            index=list(cg.payoff_surface.index),
            columns=list(cg.payoff_surface.index),
        ),
        "constraint_audit": audit,
        "minimax": minimax,
        "cce": cce,
        "bootstrap": bs,
        "sensitivity": sa,
        "recommendation": rec,
    })
    # Append data provenance to the data summary.
    cert["data_summary"]["data_source"] = "USA-NPN site phenometrics (real)"
    cert["data_summary"]["species"] = args.species
    cert["data_summary"]["bundled_csv"] = f"phenogame/data/npn_{args.species}.csv"
    cert["data_summary"]["n_clean_rows"] = int(len(df))

    # 12. Export.
    out_dir = args.out or tempfile.mkdtemp(prefix="phenogame_demo_")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"{args.species}_eml_certificate.json")
    md_path = os.path.join(out_dir, f"{args.species}_eml_certificate.md")
    export_certificate_json(cert, json_path)
    export_certificate_markdown(cert, md_path)
    print("\n--- audit certificate ---")
    print(f"  JSON: {json_path}")
    print(f"  MD  : {md_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
