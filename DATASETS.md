# Dataset and Model Card

## Bundled data

PhenoGame 0.3.0 bundles small USA-NPN phenometrics CSVs for demonstration and reproducibility:

| File | Raw rows | Rows after explicit `-9999` filtering | Common name |
|---|---:|---:|---|
| `npn_grape.csv` | 130 | 121 | wine grape |
| `npn_apple.csv` | 843 | 713 | apple |
| `npn_peach.csv` | 329 | 301 | peach |
| `npn_red_maple.csv` | 10,913 | 9,273 | red maple |

Source: USA-NPN phenometrics; DOI: `10.5066/F78S4N1V`.

## Intended use

These files are bundled to demonstrate package mechanics:

- phenophase filtering;
- observed DOY and AGDD summaries;
- finite game construction;
- DPUU criteria;
- zero-sum minimax solution by linear programming for a Farmer-vs-Nature model;
- Hedge-based epsilon-CCE certification;
- bootstrap and sensitivity robustness diagnostics.

## Non-intended use

The bundled datasets are not validated commercial recommendation datasets. They should not be used as the sole basis for grower, crop-insurance, pesticide, irrigation, harvest, or financial decisions.

## Preprocessing

`load_npn_csv()` removes rows where:

- `Mean_First_Yes_DOY == -9999`;
- `Mean_First_Yes_DOY <= 0`;
- `Mean_AGDD == -9999`;
- `Mean_AGDD <= 0`.

No hidden imputation is performed by the loader.

## Known limitations

- Observation density differs by crop and phenophase.
- Small phenophase subsets can produce wide intervals and unstable recommendations.
- Payoff matrices are user/model constructed, not directly observed economic profit.
- Epsilon-CCE certificates are conditional on the finite game, payoff formula, and scenario definitions.
- EML reduction metadata is representational; it is not biological validation.

## Recommended reporting language

Use:

> "PhenoGame produces data-induced finite-game decision certificates for phenology decision support under uncertainty."

Avoid:

> "PhenoGame proves a new Nash theorem" or "PhenoGame provides validated agronomic prescriptions."
