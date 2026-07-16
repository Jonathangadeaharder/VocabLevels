# German A1 Gap Refill Audit — 2026-07-16

| Metric | Count |
|--------|------:|
| Total audited | 101 |
| Keep (implicit) | 11 |
| Fix | 5 |
| Drop | 85 |

## Keeps (no JSONL entry)

Lines 28, 65, 73, 79, 88, 89, 90, 93, 94, 96, 102 — `Heim`, `total` (ADV), `wert`, `tatsächlich`, `Montag`, `drucken`, `Zugticket`, `Pate`, `mager`, `Zahnpasta`, `kosten`.

## Fixes

| Line | Lemma | Issue |
|-----:|-------|-------|
| 25 | ach | English "hear" → "oh" |
| 52 | besonders | ADJ "special" → ADV "especially" |
| 78 | sorgen | English "care" → "worry" (matches Chinese) |
| 86 | buchstabieren | English "book" → "spell" |
| 87 | arm → Arm | Body-part sense needs NOUN `Arm`, not ADJ `arm` (poor) |

## Notable issues

1. **Foreign/non-German lemmas** — `best`, `Get`, `Mind`, `always`, `完全` (lines 9, 23, 42, 80, 81).
2. **Inflected / non-citation forms** — comparatives/superlatives (`besser`, `weniger`, `wenigsten`, `letzte`), Konjunktiv auxiliaries (`könnte`, `würde`, `sollte`), imperatives (`Beeile`, `Nimm`), pronominal inflections (`ihm`, `uns`, `diese`, `dies`, `jene`).
3. **Wrong POS + duplicate** — large batch already covered in committed A1 set under correct UPOS (`weil`, `beide`, `anders`, `viel`, `mein`, `nein`, `dort`, `zusammen`, `verstehen`, `oben`, `wollen`, `noch`, `dein`, etc.).
4. **Noun–verb mistranslations** — deverbal nouns glossed as verbs (`Bringung`, `Verschwinden`, `Erwartung`, `Treffen`, `Waschen`, `Anschein`, `Ersparnis`).
5. **Above-A1 / specialized** — abstract or technical nouns (`Verrücktheit`, `Materie`, `Freisetzung`, `Fremdartigkeit`, `Million`, `vielseitig`, `Kiesel`, `eifersuchtig`, `Jammer`).
6. **Cross-level duplicates** — `sowieso` (A2 ADV), `hinein` (B1 ADV); dropped rather than relocated.
7. **No internal lemma+POS duplicates** within the 101-row batch.

Decisions written to `gap-review.jsonl` (reviewer: `manual-gap-2026-07-16`).
