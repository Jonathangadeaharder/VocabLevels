# German A1 Gap Refill Audit Round 3 — 2026-07-16

| Metric | Count |
|--------|------:|
| Total audited | 80 |
| Keep (implicit) | 35 |
| Fix | 1 |
| Drop | 44 |

Reviewer: `manual-gap-r3-2026-07-16`

Decisions written to `review.jsonl` (fresh isolated directory; not mixed with prior round `gap/review.jsonl`).

## Fix

| Line | Lemma | Issue | Replacement |
|-----:|-------|-------|-------------|
| 20 | angebeben | Misspelling | angeben VERB — to state / 说明 |

## Keep (no JSONL entry)

| Line | Lemma | English | POS |
|-----:|-------|---------|-----|
| 10 | Vormittag | forenoon | NOUN |
| 11 | Helfer | helper | NOUN |
| 12 | munter | awake | ADJ |
| 14 | Schulbank | school bench | NOUN |
| 17 | Schulranzen | schoolbag | NOUN |
| 18 | Hähnchen | chicken | NOUN |
| 19 | joggen | jog | VERB |
| 21 | hüpfen | hop | VERB |
| 24 | umsteigen | to change (trains) | VERB |
| 27 | Rind | beef | NOUN |
| 29 | Zimt | cinnamon | NOUN |
| 37 | Ameise | ant | NOUN |
| 41 | Feder | feather | NOUN |
| 42 | Pinguin | penguin | NOUN |
| 43 | Hahn | rooster | NOUN |
| 44 | Fliege | fly | NOUN |
| 45 | Dackel | dachshund | NOUN |
| 47 | Igel | hedgehog | NOUN |
| 48 | Erde | earth | NOUN |
| 49 | lila | purple | ADJ |
| 50 | Umhang | cloak | NOUN |
| 51 | Leder | leather | NOUN |
| 52 | Leinen | linen | NOUN |
| 53 | silber | silver | ADJ |
| 54 | Pantoffel | slipper | NOUN |
| 56 | dunkelblau | dark blue | ADJ |
| 61 | zornig | angry | ADJ |
| 63 | Fett | fat | NOUN |
| 66 | Menü | menu | NOUN |
| 68 | pfeifen | to whistle | VERB |
| 70 | ordnen | to organize | VERB |
| 72 | Marmelade | jam | NOUN |
| 73 | zahm | tame | ADJ |
| 76 | Zweig | branch | NOUN |
| 81 | teils | partly | ADV |

## Notable drop categories

1. **Inflected / non-citation forms** — `konnte` (Konjunktiv II), `dich` (accusative du), `Minuten`, `Nasen`, `reife`, `fester`, `umritten`, `dicke`, `ideale`, `wenigst` (truncated).
2. **Wrong POS + A1 duplicates** — `wir` INTJ (PRON in A1), `viel` ADV (DET in A1), `Wissen` NOUN (wissen VERB in A1).
3. **Deverbal / abstract junk** — `Schlechtigkeit`, `Waschgang`, `Zeitraum`, `Wandlung`, `Tätigkeit`, `Läufer`, `leuchtend`, `Impuls`.
4. **Misspellings / non-words** — `angebeben`, `Jahrzeit`, `Oel` (Öl in B2 blocks fix).
5. **Near-duplicates / wrong citation** — `nahe` (nah ADV in A1), `rasch` (schnell in A1), `Bagage` (Gepäck B2), `Eck` (Ecke B1).
6. **Above-A1 / specialized** — wildlife batch (`Dachs`, `Uhu`, `Hirsch`, `Falke`, `Hecht`), `ultramarin`, `Flanell`, `Saum`, `Oase`, `Schlauberger`, `reichhaltig`, `Anbau`, `Durchfall`.
7. **Cross-level sense already covered** — `verliebt` (VERB A2), `necken` (colloquial teasing above core A1).

## Outcome

If applied: **+36 rows** to A1 (515 → 551). No internal duplicates within the 80-row batch. No lemma+POS collisions with committed A1; higher-level homographs retained only where POS differs and entry is A1-appropriate (e.g. Erde NOUN vs Erde PROPN in A2).
