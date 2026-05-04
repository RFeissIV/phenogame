# PhenoGame data and claim audit card

## Source package data

The bundled CSV files were checked against the uploaded `datasheet_1777767104274.zip` export. Each bundled file is an exact row-order subset by `Common_Name` from `site_phenometrics_data.csv` in that upload.

| Bundled file | Common name | Raw rows | `-9999` sentinel cells in raw file | Rows after `load_npn_csv()` filtering | Filtered year range |
|---|---:|---:|---:|---:|---:|
| `npn_grape.csv` | wine grape | 130 | 447 | 121 | 2013-2020 |
| `npn_apple.csv` | apple | 843 | 4,504 | 713 | 2014-2020 |
| `npn_peach.csv` | peach | 329 | 1,406 | 301 | 2010-2020 |
| `npn_red_maple.csv` | red maple | 10,913 | 53,184 | 9,273 | 2010-2020 |

## Loader filtering rule

`load_npn_csv()` removes records where either:

- `Mean_First_Yes_DOY == -9999`
- `Mean_First_Yes_DOY <= 0`
- `Mean_AGDD == -9999`
- `Mean_AGDD <= 0`

It does not filter by state unless the caller does so after loading.

## Verified quick-start result

The README quick-start was re-run on `npn_grape.csv` with `phenophase="Ripe fruits"`, `seed=42`, and the default `iterations=10000`.

Verified values:

| Field | Value |
|---|---:|
| `n_obs` | 11 |
| mean DOY | 198.55 |
| DOY SD | 26.46 |
| mean AGDD | 2803.96 |
| recommended strategy | `early` |
| DPUU consensus | `YES` |
| zero-sum minimax pure | `NO (mixed)` |
| CCE gap | 0.002485 |
| epsilon | 0.014823 |
| certified | `YES` |

## Claim boundary

The package supports a data-induced, finite-game, payoff-dependent epsilon-CCE certificate under the implemented Hedge mode. It does not prove agronomic truth, biological causality, commercial prescription validity, or a new equilibrium theorem.
