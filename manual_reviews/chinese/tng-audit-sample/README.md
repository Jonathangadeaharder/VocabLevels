# TNG CEFR audit samples — chinese (p20)

Language code: `zh`. Source: `{level}.proposed.csv` vs committed `{level}.csv`.

## Sample design

- Stratified random sample **per level** by change type
  (`edited` / `pos_or_key_changed` / `new_or_renamed` / `unchanged`)
- **95% confidence, ±20% margin**, p=0.5, finite-population corrected
- Seed: `20260718`

## Inventory

| Level | Proposed N | n (p20) | Change mix |
|-------|----------:|--------:|------------|
| A1 | 596 | 24 | unchanged=10, pos_or_key_changed=11, new_or_renamed=1, edited=2 |
| **Σ** | | **24** | |

## Files

| File | |
|------|--|
| `ALL.sample-p20.csv` | combined |
| `{level}.sample-p20.csv` + `.md` | per-level checklist |

## Score (filled)

Verdicts: keep=24 fix=0 drop=0. Defect rate (fix+drop)/n = 0.0%.
Status: **REVIEWED** (agent audit, 95%±20% sample).

