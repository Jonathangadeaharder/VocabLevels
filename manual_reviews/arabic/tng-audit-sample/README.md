# TNG CEFR audit samples — arabic (p20)

Language code: `ar`. Source: `{level}.proposed.csv` vs committed `{level}.csv`.

## Sample design

- Stratified random sample **per level** by change type
  (`edited` / `pos_or_key_changed` / `new_or_renamed` / `unchanged`)
- **95% confidence, ±20% margin**, p=0.5, finite-population corrected
- Seed: `20260718`

## Inventory

| Level | Proposed N | n (p20) | Change mix |
|-------|----------:|--------:|------------|
| A1 | 530 | 24 | edited=10, new_or_renamed=10, pos_or_key_changed=4 |
| A2 | 524 | 24 | edited=14, new_or_renamed=5, pos_or_key_changed=5 |
| B1 | 805 | 24 | edited=17, new_or_renamed=5, pos_or_key_changed=2 |
| **Σ** | | **72** | |

## Files

| File | |
|------|--|
| `ALL.sample-p20.csv` | combined |
| `{level}.sample-p20.csv` + `.md` | per-level checklist |

## Score

Verdict: `keep` / `drop` / `fix`. Defect rate = (drop+fix)/n.
Packs are **UNREVIEWED** until a human/agent fills verdicts.
