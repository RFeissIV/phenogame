# USA-NPN datasheet provenance

The five ZIP archives in this directory are the *original* USA-NPN site
phenometrics exports as supplied to the package author. They are
preserved here verbatim, **not** re-derived, so that any future user can
independently verify the bundled CSVs in the parent directory
(`phenogame/data/npn_*.csv`) match the upstream USA-NPN data.

## Files

| ZIP | Species | Rows | Notes |
|---|---|---:|---|
| `datasheet_1777766690685.zip` | wine grape (Vitis vinifera) | 127 (data + header) | Wine grape only |
| `datasheet_1777766770140.zip` | red maple, wine grape | 11,043 | First red-maple+grape pull |
| `datasheet_1777766839978.zip` | red maple, wine grape | 11,043 | Identical-shape duplicate of `…770140` |
| `datasheet_1777766957944.zip` | + peach | 11,372 | Adds peach |
| `datasheet_1777767104274.zip` | + apple | 12,215 | All four species (canonical superset) |

Search parameters, identical across all five exports:

- Data Type: Site Phenometrics
- Date range: 2010-01-01 → 2020-12-31
- Data Precision Filter: 30
- Phenophase Categories: Flowers, Leaves, Fruits
- Output Fields: Min/Max/Median First Yes DOY, Mean AGDD
- Quality Flags: ignored

## Equivalence with bundled CSVs

The bundled `phenogame/data/npn_*.csv` files contain the same rows, split
by species. We have verified by exact join on
`(Site_ID, Phenophase_ID, Mean_First_Yes_Year, Mean_First_Yes_DOY,
Mean_AGDD)` that the bundled CSVs are byte-equivalent subsets of
`datasheet_1777767104274.zip`'s `site_phenometrics_data.csv`:

| Species | Bundled rows | Datasheet rows | Match |
|---|---:|---:|---|
| apple | 843 | 843 | ✓ |
| wine grape | 130 | 130 | ✓ |
| peach | 329 | 329 | ✓ |
| red maple | 10,913 | 10,913 | ✓ |
| **Total** | **12,215** | **12,215** | **✓** |

(After the package's `-9999` sentinel filter, the cleaned counts are
713 / 121 / 301 / 9,273 = **10,408 clean rows** total.)

## Reproducing the equivalence check

```python
import pandas as pd
import zipfile

# 1. Read the largest superset
with zipfile.ZipFile("datasheet_1777767104274.zip") as z:
    with z.open("datasheet_1777767104274/site_phenometrics_data.csv") as f:
        sup = pd.read_csv(f)

# 2. Read each bundled CSV and confirm row equality
key_cols = ["Site_ID", "Phenophase_ID", "Mean_First_Yes_Year",
            "Mean_First_Yes_DOY", "Mean_AGDD"]
for sp_key, sp_name, fname in [
    ("apple", "apple", "../npn_apple.csv"),
    ("grape", "wine grape", "../npn_grape.csv"),
    ("peach", "peach", "../npn_peach.csv"),
    ("red_maple", "red maple", "../npn_red_maple.csv"),
]:
    bundled = pd.read_csv(fname)
    upl = sup[sup["Common_Name"] == sp_name]
    b = bundled[key_cols].sort_values(key_cols).reset_index(drop=True)
    u = upl[key_cols].sort_values(key_cols).reset_index(drop=True)
    assert b.equals(u), f"{sp_key} mismatch"
    print(f"{sp_key}: {len(bundled)} rows match")
```

## Source

USA-NPN Phenometrics: <https://www.usanpn.org/data/observational>

The USA National Phenology Network is a public program run jointly by
USGS, NOAA, USDA, and partners. Their data are released for public
research use.
