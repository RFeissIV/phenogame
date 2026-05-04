# Contributing to PhenoGame

## Development Setup

```bash
git clone https://github.com/RFeissIV/phenogame.git
cd phenogame
pip install -e ".[dev]"
pytest tests/
```

## Code Standards

- All code must pass `pytest` with 0 failures.
- All public functions must have docstrings.
- No hardcoded data values in examples — load from bundled CSVs.
- No `np.random` in examples — no synthetic noise.
- Claim language must be honest: "data-induced ε-CCE certificate,
  not a new equilibrium proof."

## Pull Request Process

1. Fork and create a feature branch.
2. Add tests for new functionality.
3. Run `pytest tests/` — all must pass.
4. Update CHANGELOG.md.
5. Submit PR with a clear description.

## Adding New Crops

To add a new crop dataset:
1. Bundle the CSV in `phenogame/data/`.
2. Write a loader function in `pipeline.py`.
3. Write an example in `examples/`.
4. Every value must trace to a cited data source.
5. Add explicit sentinel/missing-value filtering.
