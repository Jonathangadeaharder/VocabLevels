---
id: ADR-004
kind: adr
title: Quality Validation Tooling
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
checksum: 9a5131494b6697a82c0999f0d8fbac16ca9d89393fb6f2d45321c6beb10a38ee
---

**Context:** With 24,000 manually curated vocabulary entries, data quality issues are inevitable. Duplicate lemmas, missing translations, special characters, and inflected forms (plurals, verb conjugations) degrade database quality. Automated validation is needed.

**Decision:** `check_quality.py` validates all CSV files checking: header correctness, row counts vs CEFR targets, empty lemmas, multi-word lemmas, digits/special chars in lemmas, intra-level and cross-level duplicates, missing translations, whitespace issues, and shared translations. Additional scripts (`audit_lemmatization.py`, `cleanup_inflections.py`) handle data-specific quality tasks such as inflected form detection and cleanup.

**Consequences:**
- Positive: Catches structural issues before they compound
- Positive: Clear error reporting with file and line numbers
- Negative: Does not validate translation plausibility (e.g., correct word in wrong language)
- Negative: Target row counts (±5% tolerance) may mask level mismatches

**Alternatives:**
- Manual review: Impractical at 24,000 entries
- Machine translation verification: Would add ML dependency
