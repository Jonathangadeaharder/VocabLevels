# Cross-level lemma+upos consolidation (en/fr/nl)

## Policy
For each `(lang, lemma, upos)` appearing on multiple CEFR levels:
- **Keep** the row on the **lowest** level (A1 < A2 < B1 < B2 < C1)
- **Remove** copies on higher levels
- Conflicting glosses: lowest-level English/Chinese wins

## Counts
- Conflicting-gloss groups: 58 (model report)
- Same-gloss multi-level groups also consolidated: 222
- Total rows removed: 294
- Remaining multi-level groups: **0**

## Files
- Report: `cross-level-dedupe-en-fr-nl.csv`
- Updated committed lists: `english|french|dutch/{A1–C1}.csv`
- Re-export for model: `ALL-fully-done-proposed-for-audit.csv`

## Next
Send `ALL-fully-done-proposed-for-audit.csv` for final zero-defect verification sample.
