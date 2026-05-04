# Changelog

## 0.3.0+eml-rml-release-polish (2026-05-03, public-release polish)

### Changed (cosmetic / packaging only — no behaviour changes)
- Standardized public terminology around the data-induced EML game compiler
  extension. EML trees are described as upstream function representations,
  not as equilibrium concepts.
- Renamed legacy extension files, examples, and tests to neutral
  EML-game-extension names.
- Reworded the grape demo notes from synthetic-payoff language to
  "transparent deterministic surrogate utility derived from real observations".
  The rows are real; the utility function is a transparent modeling assumption.
- Corrected the Odrzywolek (2026) citation title in
  `EML_RML_AUDIT_RESPONSE.md`, `phenogame/game.py`, and the README
  references list to the actual arXiv title:
  *All elementary functions from a single binary operator*
  (arXiv:2603.21852).
- Improved Dillon (1962) citation in the README references list to use
  the canonical full title:
  *Applications of game theory in agricultural economics: Review and
  requiem* — already present in `phenogame/game.py`.
- Replaced speculative internal-audit wording in `phenogame/eml.py` with
  a neutral descriptive comment about the EML identities used for metadata.

### Removed (release-hygiene only)
- Removed cache/build debris from the source tree and source distribution.
- Added top-level ignore and manifest exclusions covering Python and
  packaging caches (`__pycache__/`, `.pytest_cache/`, `.mypy_cache/`,
  `.ruff_cache/`, `*.egg-info/`, `build/`, `dist/`).

### Notes
- No model output or numeric determinism pin was intentionally changed in
  this pass. The pinned determinism contract remains covered by
  `tests/test_determinism.py`.
- The Repository URL in `pyproject.toml`, `CITATION.cff`, and
  `CONTRIBUTING.md` still requires maintainer confirmation before public
  release.

## 0.3.0+eml-game-postpatch (2026-05-03, second patch)

### Removed (BREAKING)
- The forward/backward bidirectional query API has been removed in its
  entirety. PhenoGame is now positioned as a game-theoretic decision
  package, not a bidirectional phenology mapper.
- Removed modules: `phenogame/forward.py`, `phenogame/backward.py`.
- Removed top-level exports: `forward`, `ForwardResult`, `backward`,
  `sensitivity` (the per-temperature inverse-query helper),
  `BackwardResult`, `plot_forward`, `plot_sensitivity`.
- Removed `PhenoGame` methods: `forward()`, `backward()`,
  `sensitivity()`, `invert()`.
- Removed visualization helpers `plot_forward()` and `plot_sensitivity()`
  from `phenogame.viz` (`plot_fit` is preserved).
- Removed corresponding unit tests in `tests/test_phenogame.py`.

### Kept (intentional)
- `ResponseFunction.inverse_fn` attribute and the per-function numerical
  inverses in `phenogame/response.py`. These are mathematical
  properties of the response-function class (used to verify EML
  decompositions and parameter-recovery identities); they are not part
  of a bidirectional query pipeline.
- `phenogame.robustness.sensitivity_analysis` and
  `phenogame.robustness.SensitivityResult`. These are robustness-report
  utilities for the game pipeline; they are unrelated to the removed
  backward-query "sensitivity" helper and are explicitly kept.
- All game-theoretic and data-induced EML game extension modules.

### Added
- `tests/test_phenogame.py::TestPackaging::test_no_bidirectional_api` —
  negative regression test that asserts the bidirectional API stays
  removed.

## 0.3.0+eml-game-extension (2026-05-03)

### Added
- `eml_tree.py`: `eml`, `EMLTreeFeature`, `EMLTreeRegressor`, `EMLPayoffFit`,
  `fit_eml_payoff_model` — random EML-feature ridge model used as upstream
  payoff-surface representation.
- `data_compiler.py`: `compile_eml_payoff_game`, `DataInducedGame` —
  compiles a learned payoff surface into a finite strategy×scenario matrix.
- `counterfactual.py`: `counterfactual_timing_test`, `CounterfactualResult`.
- `ontology.py`: `decision_to_triples`, `decision_to_jsonld`, `DecisionTriple`.
- `game_certificate.py`: `build_game_audit_certificate`,
  `GameAuditCertificate`.
- `multiagent.py`: `MultiAgentDecisionGame` data container (no general
  N-player solver claimed).
- `validation.py`: `train_test_split_indices`, `compare_baselines`,
  `BaselineComparison`, `ModelScore` — held-out comparison of EML-tree
  against naive-mean, ridge-linear, and optional random-forest baselines.
- `validation.py`: `action_stability`, `ActionStabilityResult` — bootstrap
  recommendation tracking for the data-induced game.
- `examples/grape_eml_game_extension_demo.py` — end-to-end reference
  demo emitting `grape_decision_certificate.json`,
  `grape_robustness_summary.csv`, `grape_counterfactual_table.csv`, and
  `grape_baseline_comparison.csv`.
- `tests/test_eml_game_extensions.py` (6 tests).
- `tests/test_validation.py` (9 tests).
- `tests/test_grape_demo.py` (1 smoke test).
- CI workflow upgraded with ruff/mypy lint job, coverage reporting, and
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` for reproducible test runs.

### Notes
- The grape demo uses a transparent deterministic surrogate utility
  derived from real observations
  (`-|DOY - target| - 0.001*|AGDD - mean(AGDD)|`) so the pipeline can run
  on bundled open data without inventing an undocumented ground truth.
  The rows are real USA-NPN observations; only the utility function is a
  transparent closed-form expression.
- No new equilibrium theorem is claimed. EML-tree learning is upstream
  representation only.

## 0.3.0 (2026-05-02)

### Added
- `pipeline.py`: `load_npn_csv`, `run_phenology_game`, `PhenologyGameResult`
- `pipeline.py`: `CropGameResult`, `run_game_for_crop`, `run_multi_crop`
- Bundled USA-NPN grape data (`phenogame/data/npn_grape.csv`)
- Explicit -9999 sentinel filtering for NPN data
- Honest claim language throughout: "data-induced ε-CCE certificate, not a new proof"

### Changed
- Renamed package from `phenomap` to `phenogame`
- Renamed `PhenoMap` class to `PhenoGame`
- Renamed `nash_equilibrium()` to `zero_sum_minimax()` (alias preserved)
- Renamed `NashResult` to `MinimaxResult` (alias preserved)
- Examples use only real data (USA-NPN), no synthetic values
- Version bumped to 0.3.0

### Fixed
- Forward query handles empty DataFrames gracefully
- Overflow warnings suppressed in sigmoid/exp/gompertz/beer_lambert
- Beta-thermal parameters validated for f(T) ≤ 1

## 0.2.0 (2026-04-30)

### Added
- `steering.py`: `crop_steering_certificate`, `SteeringCertificate`, `PayoffFormula`
- `equilibrium.py`: Hedge (provable) and EML (empirical) ε-CCE certification
- 13 EML-reducible response functions with inverses
- Bidirectional phenology queries (forward/backward)
- DPUU criteria (Wald, Laplace, Hurwicz, Savage)
- Banzhaf/Shapley driver attribution
