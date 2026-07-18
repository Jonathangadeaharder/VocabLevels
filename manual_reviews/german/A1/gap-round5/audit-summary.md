# German A1 Gap Refill Audit Round 5 — 2026-07-16

| Metric | Count |
|--------|------:|
| Total audited | 35 |
| Keep (implicit) | 6 |
| Fix | 0 |
| Drop | 29 |

Reviewer: `manual-gap-r5-2026-07-16`

Baseline: committed `german/A1.csv` = **562** validated rows.

Decisions written to `review.jsonl` (drops only; keeps implicit per prior rounds).

## Keep (no JSONL entry)

| Line | Lemma | English | POS |
|-----:|-------|---------|-----|
| 7 | schütteln | shake | VERB |
| 13 | Delfin | dolphin | NOUN |
| 20 | himmelblau | sky blue | ADJ |
| 22 | husten | cough | VERB |
| 26 | Pfeffer | pepper | NOUN |
| 31 | frieren | freeze | VERB |

## Notable drop categories

1. **Garbage / metadata rows** — `lemma-wünschen` (B1 wünschen), `Train` (English lemma, English gloss "sal").
2. **Misspellings** — `fliegan` (fliegen A2), `Minut` (Minute A2), `Uropa` (Urgroßvater), `Inde` (not German for index finger).
3. **Inflections / wrong POS** — `dir` (dative du), `viele` (inflected viel DET in A1).
4. **Deverbal abstracts** — `Fremdheit`, `Waschung`, `Milde`.
5. **Wrong gloss / wrong lemma** — `löhnen` (to pay → bezahlen A2; lohnen C1), `jauchen` (not to water plants), `eckig` (angular, not cornered).
6. **Cross-level duplicates** — `Zeh` (Zehe C1), `Diele` (Flur B2), `pusten` (blasen C1).
7. **Above-A1 / specialized** — `Lack`, `Unterhemd`, `Unterrock`, `Daunen`, `Nashorn`, `nachtblau`, `fuchsrot`, `kindlich`, `Durchschnitt`, `prallen`, `Umlage`, `Montage`.
8. **User-flagged junk confirmed** — all nine flagged rows dropped (`lemma-wünschen`, `Fremdheit`, `Waschung`, `dir`, `Train`, `fliegan`, `Minut`, `viele`, `Uropa`).

## Outcome

If applied: **+6 rows** to A1 (562 → 568). No internal lemma+POS duplicates within the batch. `husten` VERB kept despite `Husten` NOUN in B2 (different POS, per round 3 precedent).

Survivor lemmas: **schütteln, Delfin, himmelblau, husten, Pfeffer, frieren**
