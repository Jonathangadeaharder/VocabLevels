# Full-list model audit — en / fr / nl

Source: `ALL-fully-done-proposed-for-audit.csv` (~19.5k rows).
Findings CSV: `model-audit-en-fr-nl-full.csv`.

## Result

| Lang | Flagged | Notes |
|------|--------:|-------|
| **fr** | **0** | No defects reported — list accepted as-is on this pass |
| **en** | 5 review | Sensitive B1 headwords only (policy, not form errors) |
| **nl** | 16 | 1 drop, 13 fix, 2 review |

## Applied automatically (NL proposed only)

- Drop: `°` (B1)
- Fix: finite/participle verbs → infinitives; `minimaal` ZH; EN glosses that were Dutch
- Backups: `dutch/{level}.proposed.csv.pre-model-audit.bak`

## Not applied (need human policy)

- EN B1: assault, damn, naked, rape, sexy (review)
- NL C1: `d'r` dialect (review)
- Optional: `leve` → `leven` applied as fix but sense is interjectional “long live”; consider INTJ/drop later

## FR

Completely fine on this full-pass audit (no rows in findings CSV).

## Applied 2026-07-18 (all suggestions)

1. **FR**: proposed → committed all levels; seed-validated A1–C1.
2. **EN**: removed B1 sensitive (assault, damn, naked, rape, sexy); added to C1; seed-validated B1+C1; proposed synced.
3. **NL**: model fixes kept; dropped `d'r`; `leve` as INTJ “long live”; proposed → committed all levels; seed-validated A1–C1.
4. Backups: `dutch/*.proposed.csv.pre-model-audit.bak` where present.
