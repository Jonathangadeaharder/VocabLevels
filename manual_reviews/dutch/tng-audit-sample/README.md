# TNG CEFR audit samples — dutch (p20)

Language code: `nl`. Source: `{level}.proposed.csv` vs committed `{level}.csv`.

## Sample design

- Stratified random sample **per level** by change type
  (`edited` / `pos_or_key_changed` / `new_or_renamed` / `unchanged`)
- **95% confidence, ±20% margin**, p=0.5, finite-population corrected
- Seed: `20260718`

## Inventory

| Level | Proposed N | n (p20) | Change mix |
|-------|----------:|--------:|------------|
| A1 | 393 | 23 | edited=20, pos_or_key_changed=2, new_or_renamed=1 |
| A2 | 357 | 23 | edited=21, new_or_renamed=1, pos_or_key_changed=1 |
| B1 | 536 | 24 | edited=21, pos_or_key_changed=2, new_or_renamed=1 |
| B2 | 1056 | 24 | edited=20, pos_or_key_changed=2, new_or_renamed=2 |
| C1 | 2078 | 24 | edited=19, new_or_renamed=4, pos_or_key_changed=1 |
| **Σ** | | **118** | |

## Files

| File | |
|------|--|
| `ALL.sample-p20.csv` | combined |
| `{level}.sample-p20.csv` + `.md` | per-level checklist |

## Score (filled)

Verdicts: keep=113 fix=5 drop=0. Defect rate (fix+drop)/n = 4.2%.
Status: **REVIEWED** (agent audit, 95%±20% sample).

