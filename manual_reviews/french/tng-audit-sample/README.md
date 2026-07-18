# TNG CEFR audit samples — french (p20)

Language code: `fr`. Source: `{level}.proposed.csv` vs committed `{level}.csv`.

## Sample design

- Stratified random sample **per level** by change type
  (`edited` / `pos_or_key_changed` / `new_or_renamed` / `unchanged`)
- **95% confidence, ±20% margin**, p=0.5, finite-population corrected
- Seed: `20260718`

## Inventory

| Level | Proposed N | n (p20) | Change mix |
|-------|----------:|--------:|------------|
| A1 | 522 | 23 | unchanged=23 |
| A2 | 520 | 23 | unchanged=23 |
| B1 | 878 | 24 | unchanged=24 |
| B2 | 1707 | 24 | unchanged=24 |
| C1 | 3499 | 24 | unchanged=24 |
| **Σ** | | **118** | |

## Files

| File | |
|------|--|
| `ALL.sample-p20.csv` | combined |
| `{level}.sample-p20.csv` + `.md` | per-level checklist |

## Score (filled)

Verdicts: keep=115 fix=3 drop=0. Defect rate (fix+drop)/n = 2.5%.
Status: **REVIEWED** (agent audit, 95%±20% sample).

