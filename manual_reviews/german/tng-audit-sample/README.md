# TNG CEFR audit samples — german (p20)

Language code: `de`. Source: `{level}.proposed.csv` vs committed `{level}.csv`.

## Sample design

- Stratified random sample **per level** by change type
  (`edited` / `pos_or_key_changed` / `new_or_renamed` / `unchanged`)
- **95% confidence, ±20% margin**, p=0.5, finite-population corrected
- Seed: `20260718`

## Inventory

| Level | Proposed N | n (p20) | Change mix |
|-------|----------:|--------:|------------|
| A1 | 581 | 24 | unchanged=24 |
| A2 | 366 | 23 | edited=21, pos_or_key_changed=1, new_or_renamed=1 |
| B1 | 622 | 24 | edited=22, pos_or_key_changed=1, new_or_renamed=1 |
| C1 | 1980 | 24 | edited=20, pos_or_key_changed=2, new_or_renamed=2 |
| **Σ** | | **95** | |

## Files

| File | |
|------|--|
| `ALL.sample-p20.csv` | combined |
| `{level}.sample-p20.csv` + `.md` | per-level checklist |

## Score

Verdict: `keep` / `drop` / `fix`. Defect rate = (drop+fix)/n.
Packs are **UNREVIEWED** until a human/agent fills verdicts.
