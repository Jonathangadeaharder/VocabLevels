# German A1 Gap Refill Audit Round 2 — 2026-07-16

| Metric | Count |
|--------|------:|
| Total audited | 87 |
| Keep (implicit) | 1 |
| Fix | 1 |
| Drop | 85 |

Reviewer: `manual-gap-r2-2026-07-16`

Decisions written to `review-round2.jsonl` (fresh file; not appended to round-1 `review.jsonl`).

## Keep (no JSONL entry)

| Line | Lemma | English | POS |
|-----:|-------|---------|-----|
| 88 | Gutschein | voucher | NOUN |

Core A1 shopping vocabulary; citation form, glosses, and POS are correct. No cross-level collision found.

## Fix

| Line | Lemma | Issue |
|-----:|-------|-------|
| 59 | achso | English "we" → "oh I see"; Chinese refined to 啊，原来 |

## Notable issues

1. **Foreign/non-German lemmas** — `best`, `Get`, `again`, `now`, `old`, `on`, `one`, `only`, `or`, `Work`, `Pen` (lines 9, 23, 67, 72–77, 81–82). German equivalents already in committed A1 (`wieder`, `jetzt`, `alt`, `eins`, `nur`, `oder`, `Arbeit`, `Kugelschreiber`).
2. **Inflected / non-citation forms** — comparatives/superlatives (`besser`, `weniger`, `wenigsten`, `letzte`, `Schlechter`), Konjunktiv auxiliaries (`könnte`, `würde`, `sollte`), imperatives (`Hör`, `Nimm`), pronominal inflections (`ihm`, `uns`, `diese`, `dies`, `jene`, `welche`).
3. **Wrong POS + duplicate** — large batch already covered in committed A1 under correct UPOS (`weil`, `beide`, `anders`, `viel`, `mein`, `nein`, `dort`, `zusammen`, `verstehen`, `oben`, `noch`, `dein`, `jeder`, etc.).
4. **Noun–verb mistranslations** — deverbal nouns glossed as verbs (`Bringung`, `Verschwinden`, `Erwartung`, `Treffen`, `Ausgabe`, `Ersparnis`, `Schnelligkeit`, `Veröffentlichung`).
5. **Above-A1 / specialized** — abstract or technical nouns (`Verrücktheit`, `Materie`, `Million`, `Kiesel`, `Vene`, `Natter`, `Weibchen`, `Feinheit`).
6. **Cross-level duplicates** — `sowieso` (A2 ADV), `entlang` (B1 ADP), `besonders` (A2 ADV), `irgendjemand` (B2 PRON), `verschwinden` (A2 VERB), `willkommen` (A2 ADJ), `komplett`/`definitiv`/`Mond`/`Zunge`/`schlechter` (B1), `innen`/`jene`/`solch` (C1). Dropped rather than relocated.
7. **Capitalization errors** — `Schlecht`, `Heute`, `Gestern`, `Verschwinden`, `Willkommen` (should be lowercase unless sentence-initial).
8. **Misspellings** — `Rätel` (→ raten), `Zungue` (→ Zunge).

## Outcome vs round 1

Round 2 is a residual batch overlapping heavily with round-1 rejects, plus English leakage and a handful of new noun garbage (`Schnelligkeit`, `Veröffentlichung`, `Ausgabe`, `Natter`, `Zungue`, `Vene`). Only **Gutschein** survives as a clean add; **achso** needs a gloss fix.

If both are applied: **+2 rows** to A1 (513 → 515).
