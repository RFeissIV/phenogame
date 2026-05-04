# EML-RML twenty-pass audit — response & closure map

This document maps every finding (F1–F10) of the EML-RML twenty-pass
audit (`eml_combined_audit_v2.docx`) to its status in PhenoGame v4.

The audited artifact in the audit doc is an **R sibling pipeline** that
benchmarks four regret-matching transforms (standard, exponential,
softplus, EML) on five canonical games + 100 random 3×3 games.
PhenoGame's `phenogame.equilibrium._eml_regret_transform` is
algebraically identical to the R reference's `psi_eml_regret`. v4 adds
the comparator panel that the audit findings collectively call for, on
top of PhenoGame's already-shipped data-induced game pipeline.

| Finding | Severity | Status in v4 | Implementation |
|---|---|---|---|
| F1: Quadratic DataFrame growth | Low | **Out of scope (R-pipeline-specific)** | Python pipeline does not have rbind-in-loop; pre-allocates lists. |
| F2: Single seed; no variance quantification | Moderate | **Closed** | `compare_regret_transforms(n_seeds=N)` returns mean ± SD across seeds. Default `n_seeds=10`. |
| F3: Random games fixed at 3×3 | Moderate | **Out of scope for the package** | The package operates on user-supplied data-induced games, not random benchmark zoos. The comparator accepts arbitrary ``(m, n)`` payoff matrices, so the user can run the panel on 2×2, 3×3, 5×5 manually if desired. |
| F4: Stage 3 is speed-only | Low | **Closed** | Comparator records final CCE gap, iters-to-ε, and wall-clock seconds per run; aggregates report all three. |
| F5: Limited figure generation | Low | **Out of scope (R-pipeline-specific)** | Python ecosystem; users can plot from the comparator's `to_dataframe()`. |
| F6: Softplus is not strict regret matching | Moderate | **Closed** | `psi_softplus` docstring carries the F6 caveat verbatim; `TRANSFORM_REGISTRY["softplus"]["always_positive"] = True` and `is_strict_rm = False` flow through to the aggregate report. |
| F7: No cross-rule correlation comparison | Moderate | **Closed** | All four transforms run on identical (game, seed) pairs; comparator reports paired aggregates so EML can be compared head-to-head against standard / exp / softplus on the same game. |
| F8: Correlation residual unnormalized | Low (Moderate if F3 addressed) | **Closed** | New module `phenogame.joint_correlation` implements `ρ(q) ∈ [0, 1]` via normaliser `2(1 − 1/min(m, n))`. Both raw and normalised forms exposed. |
| F9: No formal no-regret proof | Advisory | **Acknowledged honestly** | `psi_eml` docstring states "no formal no-regret proof is known"; `TRANSFORM_REGISTRY["eml"]["no_regret_proof"]` reads "empirical only; not formally proven (audit F9)"; cited references Hart & Mas-Colell (2000), Blum & Mansour (2007), Freund & Schapire (1999) appear in the module docstring. |
| F10: No wall-clock timing | Low | **Closed** | Comparator records `wall_clock_seconds` per run via `time.perf_counter()`; aggregate reports mean ± SD. |

**Summary.** v4 closes 6/10 findings (F2, F4, F6, F7, F8, F10) directly,
acknowledges 1 honestly (F9), and marks 3 as out-of-scope for a Python
data-induced-game package (F1, F3, F5 are R-pipeline-specific).

## What the user gets

```python
from phenogame import compile_game_from_data, compare_regret_transforms
from phenogame.fixtures import load_phenology_table

df = load_phenology_table("grape")
cg = compile_game_from_data(
    df, target_col="Mean_First_Yes_DOY",
    feature_cols=["Latitude","Longitude","Elevation_in_Meters","Mean_First_Yes_Year"],
    scenario_cols=["State"],
)
panel = compare_regret_transforms(cg.payoff_matrix, n_seeds=10, iterations=2000)
print(panel.summary())
```

The user's data-induced game is run under all four no-regret dynamics
with n_seeds=10 paired comparisons. Output: per-transform final CCE gap
mean ± SD, iters-to-ε mean ± SD, normalised joint correlation residual
mean ± SD, wall-clock seconds mean ± SD. This is the legitimate
publishable artifact — empirical convergence to ε-CCE on a benchmarked
set of dynamics, applied to real USA-NPN data.

## What v4 does NOT do

- Does not promote `certify_empirical` to the default certifier; Hedge
  remains the default with its formal Freund-Schapire bound.
- Does not claim EML-RML as a new equilibrium concept anywhere.
- Does not delete or rename the existing
  `phenogame.equilibrium._eml_regret_transform`. The new
  `phenogame.regret_transforms.psi_eml` is a public refactor; tests pin
  bit-for-bit equivalence.
- Does not modify the v3 determinism contract; all 7 pinned values still
  hold and are tested as part of the v4 suite.

## References

- Hart, S. and Mas-Colell, A. (2000). A simple adaptive procedure
  leading to correlated equilibrium. *Econometrica* 68(5).
- Blum, A. and Mansour, Y. (2007). From external to internal regret.
  *Journal of Machine Learning Research*.
- Freund, Y. and Schapire, R. E. (1999). Adaptive game playing using
  multiplicative weights. *Games and Economic Behavior* 29.
- Odrzywolek, A. (2026). All elementary functions from a single binary
  operator. *arXiv:2603.21852*.
