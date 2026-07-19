# Post-fix p20 (en / fr / nl) — residual check

After full-list model audit + applies. New stratified sample (by UPOS), seed `20260718:postfix`.
95% ±20% FPC. **Manual review of all 356 rows.**

## Results

| Lang | n | keep | fix | drop | review | residual non-keep |
|------|--:|-----:|----:|-----:|-------:|------------------:|
| en | 120 | 119 | 0 | 0 | 1 | 1 (0.8%) |
| fr | 118 | 115 | 3 | 0 | 0 | 3 (2.5%) |
| nl | 118 | 114 | 3 | 0 | 1 | 4 (3.4%) |

## Residual defects found (model did **not** catch all)

- **en A1** `talk` → **review**: Listed as NOUN only; core verb sense talk/VERB also CEFR-relevant (may already exist).
- **fr A2** `dès` → **fix**: Gloss 'as soon as' is dès que; dès alone ≈ from/as of (ADP).
- **fr B2** `revendiquer` → **fix**: english_lemma should be infinitive gloss 'to claim'.
- **fr C1** `travers` → **fix**: Bare travers is NOUN; preposition is à travers. Still open from earlier p20.
- **nl A1** `zitten` → **fix**: english_lemma 'sit' → 'to sit' for VERB consistency.
- **nl B1** `eenheid` → **fix**: UPOS AUX is wrong for "unit"; must be NOUN.
- **nl C1** `talenten` → **fix**: Plural form; citation lemma talent.
- **nl C1** `schamen` → **review**: Usually reflexive zich schamen; bare schamen OK as citation stem for some lists.

## Interpretation

- Prior full-list model pass was **strong but incomplete** (missed at least the residuals above).
- Residual non-keep rate in this p20 is low (~1–2% of sample); consistent with lists being mostly clean after fixes.
- ±20% still allows true residual defect rate up to ~15–20%; this is a spot-check, not a proof of zero defects.

Combined file: `manual_reviews/post-fix-p20/ALL-en-fr-nl.post-fix-p20.csv`
