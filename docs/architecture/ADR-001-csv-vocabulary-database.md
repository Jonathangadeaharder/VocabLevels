# ADR-001: CSV-Based Vocabulary Database

**Status:** Accepted

**Context:** The project manages trilingual vocabulary data organized by CEFR levels (A1-C1). The data must be human-readable, version-controllable, and editable without specialized tooling. A full database engine would be over-engineered for 15 files with ~24,000 entries.

**Decision:** Store vocabulary data as CSV files — one file per language per CEFR level (15 files total: 3 languages × 5 levels). Each CSV has a lemma column and two translation columns. Files are sorted alphabetically by lemma. No database engine or ORM.

**Consequences:**
- Positive: Fully human-readable with any text editor or spreadsheet
- Positive: Git-tracked with readable diffs per row
- Positive: Zero infrastructure (no DB server, no schema migrations)
- Negative: No referential integrity (orphaned translations, inconsistent casing)
- Negative: No atomic writes (CSV corruption on failed write)
- Negative: Concurrent edits unsafe (no transaction isolation)

**Alternatives:**
- SQLite: Schema enforcement but requires SQL knowledge to edit
- JSON: Less human-editable in spreadsheets
- YAML: Verbose for 24,000 entries
