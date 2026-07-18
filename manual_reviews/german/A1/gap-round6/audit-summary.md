# German A1 Gap Refill Audit Round 6 — 2026-07-16

| Metric | Count |
|--------|------:|
| Total audited | 24 |
| Keep (implicit) | 8 |
| Fix | 1 |
| Drop | 15 |

Reviewer: `manual-gap-r6-2026-07-16`

Baseline: committed `german/A1.csv` = **568** validated rows.

Decisions written to `review.jsonl` (drops and fixes only; keeps implicit per prior rounds).

## Fix

| Line | Lemma | Issue | Replacement |
|-----:|-------|-------|-------------|
| 6 | Beeren | Plural inflection | Beere NOUN — berry / 莓果 |

## Keep (no JSONL entry)

| Line | Lemma | English | POS |
|-----:|-------|---------|-----|
| 7 | Lamm | lamb | NOUN |
| 8 | farbig | colorful | ADJ |
| 16 | naschen | to snack/eat sweets | VERB |
| 18 | rund | round | ADJ |
| 19 | Essig | vinegar | NOUN |
| 20 | lächeln | smile | VERB |
| 22 | endlos | endless | ADJ |
| 25 | Deutschunterricht | German class | NOUN |

## Notable drop categories

1. **Garbage / metadata rows** — `lemma-error` (sagen VERB in A1).
2. **Inflections / wrong POS** — `würden` (werden AUX A1), `falsches` (falsch ADJ A1), `rechtzeitig` ADV (rechtzeitig ADJ B1).
3. **Wrong gloss / wrong lemma** — `Wendung` (phrase not turn; Drehung A1), `wachen` (guard/awake not wake; aufwachen C1).
4. **Deverbal abstracts** — `Einfachheit` (einfach ADJ A1).
5. **Cross-level / above-A1** — `zwischenzeitlich` (inzwischen B1), `vermieten` (mieten C1), `Mahl` (Mahlzeit B2), `dürsten` (Durst B2).
6. **Specialized / compound color** — `altrosa` (r5 precedent for compound colors).
7. **Specialized vocabulary** — `durchsichtig`, `Dunst`, `Espe`.
8. **User-flagged junk confirmed** — all six flagged rows dropped or fixed (`lemma-error` drop, `würden` drop, `Beeren` fix→Beere, `falsches` drop, `Einfachheit` drop, `zwischenzeitlich` drop).

## Outcome

If applied: **+9 rows** to A1 (568 → 577). No internal lemma+POS duplicates within the batch. `lächeln` VERB kept despite `Lächeln` NOUN in C1 (different POS, per round 5 precedent). `rund` ADJ kept despite `rund` ADV in B2 (different POS).

Survivor lemmas: **Beere, Lamm, farbig, naschen, rund, Essig, lächeln, endlos, Deutschunterricht**
