# German A1 Gap Refill Audit Round 4 — 2026-07-16

| Metric | Count |
|--------|------:|
| Total audited | 42 |
| Keep (implicit) | 11 |
| Fix | 0 |
| Drop | 31 |

Reviewer: `manual-gap-r4-2026-07-16`

Decisions written to `review.jsonl` (drops only; keeps implicit per prior rounds).

## Keep (no JSONL entry)

| Line | Lemma | English | POS |
|-----:|-------|---------|-----|
| 11 | Pulli | sweater | NOUN |
| 14 | Reiseführer | travel guide | NOUN |
| 19 | Omelett | omelet | NOUN |
| 23 | Teig | dough | NOUN |
| 25 | Kraut | herb | NOUN |
| 28 | hellblau | light blue | ADJ |
| 32 | Eintritt | entrance | NOUN |
| 35 | puzzeln | puzzle | VERB |
| 36 | Joghurt | yogurt | NOUN |
| 38 | vegetarisch | vegetarian | ADJ |
| 39 | Gurke | cucumber | NOUN |

## Notable drop categories

1. **Wrong POS / inflections** — `Sitzen` NOUN (sitzen VERB C1), `trotzdem` DET (ADV A1), `laut` ADJ (ADP A2), `Rate` NOUN (rate VERB C1), `Hände`/`Augen`/`Uhren` plurals, `Verrückter` inflected noun.
2. **Deverbal abstracts** — `Fertigstellung`, `Erhalten`, `Ratschlag`, `Zahlung`.
3. **Garbage / English rows** — `Rockt` (metadata in fields), `Diet` (English for dress), `Ante` (not German for aunt).
4. **Cross-level collisions** — `Sau` C1, `Becher` B2, `offenbar` B1, `umher` (herum A2).
5. **Near-duplicates / A1 coverage** — `trotzdem`, plurals vs `Hand`/`Uhr`, `Wollpulli` vs kept `Pulli`.
6. **Above-A1 / specialized** — `minütlich`, `Eiweiß`, `Nachtigall`, `Oberkörper`, `Nachthemd`, `Lappen`, `Teich`, `Jasmin`, `walken` (textile verb), `uuh`, `ordern` (Denglish).
7. **User-flagged junk confirmed** — all eight flagged rows dropped (`Sitzen`, `trotzdem`, `Verrückter`, `Fertigstellung`, `Erhalten`, `Rate`, `Rockt`, `Hände`/`Augen`).

## Outcome

If applied: **+11 rows** to A1 (551 → 562). No internal lemma+POS survivors beyond the 11 kept. One internal dup resolved (`Wollpulli` dropped, `Pulli` kept).

Survivor lemmas: **Pulli, Reiseführer, Omelett, Teig, Kraut, hellblau, Eintritt, puzzeln, Joghurt, vegetarisch, Gurke**
