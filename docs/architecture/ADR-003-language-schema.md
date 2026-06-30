---
id: ADR-003
kind: adr
title: Language Schema Design
status: accepted
authors: [Jonathan Gadea Harder]
reviewers: [Jonathan Gadea Harder]
tags: []
supersedes: []
superseded_by: []
depends_on: []
blocks: []
implements: []
related: []
external: []
project: VocabLevels
checksum: 63ac5eb025286045a82fd7dedc448c1efa58c501bdad81f206954b37fea39a45
---

**Context:** The database started with English, German, and Spanish, and now also tracks Arabic, French, and Swedish. Each language has a lemma column and two configured translation columns. The schema must handle asymmetric column naming and partial language rollouts without duplicating language metadata across scripts.

**Decision:** Each language directory uses self-describing column names such as `English_Lemma` and `German_Translation`. The `LANGS` mapping in `vocab_schema.py` is the single source of truth for lemma and translation columns; CLI management, quality checks, lemmatization audits, cleanup scripts, and tests import it. Translations remain directional, so a German entry can point to English and Spanish without requiring an inverse row.

**Consequences:**
- Positive: Columns are self-describing (visible without schema lookup)
- Positive: Each file has exactly 3 columns — simple to parse and validate
- Positive: New languages require one schema edit plus data files, not duplicated script edits
- Negative: Translation reciprocity is not enforced (if English "dog" says German "Hund", German might not have "Hund" → English "dog")
- Negative: Column names differ between languages, adding complexity to generic code

**Alternatives:**
- Unified schema (lemma, lang, translation1, translation2): Loses language-specific column names
- Separate translation mapping table: Over-engineered for CSV
