# ADR-004: Quality Validation Tooling

**Status:** Accepted

**Context:** With 24,000 manually curated vocabulary entries, data quality issues are inevitable. Duplicate lemmas, missing translations, special characters, and inflected forms (plurals, verb conjugations) degrade database quality. Automated validation is needed.

**Decision:** `check_quality.py` validates all CSV files checking: header correctness, row counts vs CEFR targets, empty lemmas, multi-word lemmas, digits/special chars in lemmas, intra-level and cross-level duplicates, missing translations, whitespace issues, and shared translations. Additional scripts (`audit_lemmatization.py`, `cleanup_inflections.py`, `check_quality.py`) handle data-specific quality tasks.

**Consequences:**
- Positive: Catches structural issues before they compound
- Positive: Clear error reporting with file and line numbers
- Negative: Does not validate translation plausibility (e.g., correct word in wrong language)
- Negative: Target row counts (±5% tolerance) may mask level mismatches

**Alternatives:**
- Manual review: Impractical at 24,000 entries
- Machine translation verification: Would add ML dependency
