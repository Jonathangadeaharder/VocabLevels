# TNG CEFR audit samples — swedish (p20)

Language code: `sv`. Source: `{level}.proposed.csv` vs committed `{level}.csv`.

## Sample design

- Stratified random sample **per level** by change type
  (`edited` / `pos_or_key_changed` / `new_or_renamed` / `unchanged`)
- **95% confidence, ±20% margin**, p=0.5, finite-population corrected
- Seed: `20260718`

## Inventory

| Level | Proposed N | n (p20) | Change mix |
|-------|----------:|--------:|------------|
| A1 | 519 | 23 | pos_or_key_changed=17, edited=5, new_or_renamed=1 |
| A2 | 367 | 23 | pos_or_key_changed=15, new_or_renamed=5, edited=3 |
| B1 | 587 | 24 | pos_or_key_changed=13, new_or_renamed=6, edited=5 |
| B2 | 1227 | 24 | pos_or_key_changed=13, new_or_renamed=6, edited=5 |
| **Σ** | | **94** | |

## Files

| File | |
|------|--|
| `ALL.sample-p20.csv` | combined |
| `{level}.sample-p20.csv` + `.md` | per-level checklist |

## Score (filled)

Verdicts: keep=91 fix=2 drop=1. Defect rate (fix+drop)/n = 3.2%.
Status: **REVIEWED** (agent audit, 95%±20% sample).

