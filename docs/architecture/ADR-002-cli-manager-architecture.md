---
id: ADR-002
kind: adr
title: CLI Manager Architecture
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
checksum: dab1ee0de4fc1eda56ce7dd51b7f4d025d6d25d00eca9dc2c730eb3dc183ecf9
---

**Context:** Users need to add, remove, move, update, find, and lookup vocabulary entries across all language files. The CLI must be scriptable and support batch operations.

**Decision:** Implement `vocab_manager.py` with `argparse` subcommands: `add`, `remove`, `move`, `update`, `find`, `lookup`, `lint`. Each subcommand is a standalone function returning an exit code. The main function dispatches to the correct handler. File operations use `csv.DictReader`/`csv.DictWriter` for schema-aware I/O.

**Consequences:**
- Positive: Each operation is a pure function of its inputs
- Positive: Easy to add new subcommands without restructuring
- Positive: Standard argparse means built-in --help for all commands
- Negative: No transaction safety (mid-write failure corrupts CSV)
- Negative: Input validation is manual and incomplete

**Alternatives:**
- GUI application: Better UX but harder to script, more dependencies
- Database migration pattern: Over-engineered for CSV management
