# German A1 Gap Refill Audit Round 7 — 2026-07-16

| Metric | Count |
|--------|------:|
| Total audited | 19 |
| Keep (implicit) | 2 |
| Fix | 0 |
| Drop | 17 |

Reviewer: `manual-gap-r7-2026-07-16`

Baseline: committed `german/A1.csv` = **577** validated rows.

Decisions written to `review.jsonl` (drops only; keeps implicit per prior rounds).

## Keep (no JSONL entry)

| Line | Lemma | English | POS |
|-----:|-------|---------|-----|
| 4 | Schulhof | schoolyard | NOUN |
| 17 | Vegetarier | vegetarian | NOUN |

## Notable drop categories

1. **Garbage / metadata rows** — `sag` (imperative fragment of sagen VERB A1, not INTJ).
2. **Wrong POS** — `dass` PRON (dass SCONJ A1), `regelmäßig` ADV (regelmäßig ADJ C1).
3. **Misspellings** — `Stuehl`, `Stuh` (Stuhl B1 / Stühle C1).
4. **Plural inflections** — `Themen` (Thema NOUN A1).
5. **Near-duplicates / A1 coverage** — `Suche` NOUN (suchen VERB A1, suche VERB A2), `besorgt` ADJ (besorgt VERB A2, sorgen VERB A1).
6. **Wrong gloss** — `hitzig` (heated/feverish, not hot; heiß ADJ A2).
7. **Cross-level / above-A1** — `regelmäßig` (C1), `besorgt` (A2), `Stuhl` family (B1/C1).
8. **Specialized vocabulary** — `U-boot` (submarine; Boot A2), `sankt` (proper-name prefix), `Kapital`, `Rippe`, `herzhaft`, `ultimativ`, `Ultraschall`, `unverbindlich`.
9. **User-flagged junk confirmed** — all nine flagged rows dropped (`sag`, `dass`, `Stuehl`, `Stuh`, `Themen`, `U-boot`, `sankt`, `regelmäßig`); no safe fixes attempted.

## Outcome

If applied: **+2 rows** to A1 (577 → 579). No internal lemma+POS duplicates within the batch. `Vegetarier` NOUN kept despite `vegetarisch` ADJ in A1 (different POS, per round 5/6 precedent). `Schulhof` NOUN kept as Schul- compound alongside Schulbank/Schulranzen/Schulbuch in A1.

Survivor lemmas: **Schulhof, Vegetarier**
