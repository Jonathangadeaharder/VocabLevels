---
id: ADR-001
kind: adr
title: CSV-Based Vocabulary Database
status: draft
authors: []
reviewers: []
tags: []
supersedes: []
superseded_by: []
depends_on: []
blocks: []
implements: []
related: []
external: []
project: VocabLevels
checksum: ebc20161d95e94af9e6c27744cddd1ff11d20b21453fc1b6034add5e29900244
---

> Imported legacy ADR artifact from `docs/architecture/ADR-001-csv-vocabulary-database.md`. Keep future lifecycle work in OpenSpec.

**Context:** The project manages multilingual vocabulary data organized by CEFR levels (A1-C1). The data must be human-readable, version-controllable, and editable without specialized tooling. A full database engine would be over-engineered for this CSV-sized dataset.

**Decision:** Store vocabulary data as CSV files — one file per language per CEFR level. Each CSV has a lemma column and two translation columns. Files are sorted alphabetically by lemma. No database engine or ORM.

**Consequences:**
- Positive: Fully human-readable with any text editor or spreadsheet
- Positive: Git-tracked with readable diffs per row
- Positive: Zero infrastructure (no DB server, no schema migrations)
- Positive: CLI-owned writes use temp-file replacement instead of partial in-place rewrites
- Negative: No referential integrity (orphaned translations, inconsistent casing)
- Negative: Concurrent edits unsafe (no transaction isolation)

**Alternatives:**
- SQLite: Schema enforcement but requires SQL knowledge to edit
- JSON: Less human-editable in spreadsheets
- YAML: Verbose for a large vocabulary dataset
