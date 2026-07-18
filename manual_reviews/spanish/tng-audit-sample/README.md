# TNG CEFR audit samples — spanish (p20)

Language code: `es`. Source: `{level}.proposed.csv` vs committed `{level}.csv`.

## Sample design

- Stratified random sample **per level** by change type
  (`edited` / `pos_or_key_changed` / `new_or_renamed` / `unchanged`)
- **95% confidence, ±20% margin**, p=0.5, finite-population corrected
- Seed: `20260718`

## Inventory

| Level | Proposed N | n (p20) | Change mix |
|-------|----------:|--------:|------------|
| A1 | 559 | 24 | pos_or_key_changed=19, edited=5 |
| A2 | 578 | 24 | pos_or_key_changed=17, edited=7 |
| B1 | 944 | 24 | pos_or_key_changed=15, edited=9 |
| C1 | 3828 | 24 | edited=12, new_or_renamed=1, pos_or_key_changed=11 |
| **Σ** | | **96** | |

## Files

| File | |
|------|--|
| `ALL.sample-p20.csv` | combined |
| `{level}.sample-p20.csv` + `.md` | per-level checklist |

## Score

Verdict: `keep` / `drop` / `fix`. Defect rate = (drop+fix)/n.
Packs are **UNREVIEWED** until a human/agent fills verdicts.
