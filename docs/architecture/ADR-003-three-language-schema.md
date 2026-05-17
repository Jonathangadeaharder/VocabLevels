# ADR-003: Three-Language Schema Design

**Status:** Accepted

**Context:** The database covers English, German, and Spanish. Each language has a lemma column and two translation columns pointing to the other two languages. The schema must handle asymmetric column naming (German nouns are capitalized, English/Spanish are not).

**Decision:** Each language directory uses a different column naming convention: `English_Lemma`, `German_Translation`, etc. The `LANGS` dict in both `vocab_manager.py` and `check_quality.py` maps each language to its lemma column and translation column names. Translations are directional (German entry has English_Translation and Spanish_Translation).

**Consequences:**
- Positive: Columns are self-describing (visible without schema lookup)
- Positive: Each file has exactly 3 columns — simple to parse and validate
- Negative: Translation reciprocity is not enforced (if English "dog" says German "Hund", German might not have "Hund" → English "dog")
- Negative: Column names differ between languages, adding complexity to generic code

**Alternatives:**
- Unified schema (lemma, lang, translation1, translation2): Loses language-specific column names
- Separate translation mapping table: Over-engineered for CSV
