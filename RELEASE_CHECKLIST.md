# Release Checklist

Use this checklist before publishing a public GitHub release, PyPI release, Zenodo archive, or funding packet.

## Code quality

- [ ] `python -m pytest` passes locally.
- [ ] CI passes on Linux, macOS, and Windows.
- [ ] CI passes on Python 3.9, 3.10, 3.11, 3.12, and 3.13.
- [ ] Package imports without importing Matplotlib unless plotting is requested.
- [ ] Public functions have docstrings.
- [ ] New functionality has tests.
- [ ] `python -m build` produces a clean sdist + wheel and `twine check` passes.
- [ ] `ruff check phenogame tests examples` passes.
- [ ] Coverage on `phenogame.validation`, `phenogame.eml_tree`,
      `phenogame.data_compiler`, `phenogame.counterfactual`,
      `phenogame.ontology`, `phenogame.game_certificate` is at least 80%.
- [ ] `examples/grape_eml_game_extension_demo.py` runs end-to-end and
      writes the four `outputs/grape_*` artifacts.

## Claim discipline

- [ ] README says `zero_sum_minimax()` is a two-player zero-sum Farmer-vs-Nature solution, not general Nash equilibrium.
- [ ] README says Hedge is the provable CCE mode.
- [ ] README says EML mode is empirical/representational, not a formal proof.
- [ ] README says output is decision support, not validated agronomic prescription.
- [ ] `EML_GAME_EXTENSIONS.md` clearly states EML-tree learning is upstream
      representation, not a new equilibrium theorem.
- [ ] Grape demo `assumptions.payoff_definition` field marks the demo payoff
      as a transparent surrogate utility, not validated economic value.

## Authorship and protection

- [ ] `LICENSE` is present.
- [ ] `CITATION.cff` is present.
- [ ] Copyright notice is present.
- [ ] GitHub release is dated and tagged.
- [ ] Optional: Zenodo DOI minted from the GitHub release.

## Documentation

- [ ] README quick start runs as written.
- [ ] `DATASETS.md` describes bundled data and limitations.
- [ ] `CHANGELOG.md` reflects the release.
- [ ] `SECURITY.md` explains vulnerability reporting.
- [ ] `CONTRIBUTING.md` explains development workflow.

## Company/product separation

- [ ] Open-source package is limited to PhenoGame Core.
- [ ] Commercial dashboard is not included.
- [ ] Validated grower datasets are not included.
- [ ] Crop-specific calibration services/workflows are not included.
- [ ] Hosted SaaS or enterprise deployment code is not included.
