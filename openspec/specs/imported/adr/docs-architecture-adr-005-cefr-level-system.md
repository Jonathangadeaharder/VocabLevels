---
id: ADR-005
kind: adr
title: CEFR Level System
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
checksum: b25e7f782043aa7c4f70c604984c85888477cb6faabcd0bb01935d0e2f6d28f6
---

> Imported legacy ADR artifact from `docs/architecture/ADR-005-cefr-level-system.md`. Keep future lifecycle work in OpenSpec.

**Context:** Vocabulary is organized by CEFR levels (A1, A2, B1, B2, C1). Each level has a target word count based on CEFR guidelines: A1=600, A2=600, B1=1000, B2=2000, C1=4000. Levels must be consistent across all three languages.

**Decision:** Five fixed levels with per-language CSV files and target counts. The validation tool (`check_quality.py`) checks each level's row count against targets with 5% tolerance. The `move` subcommand transfers entries between levels. No sub-levels or custom levels.

**Consequences:**
- Positive: Simple, standardized level system matches CEFR specifications
- Positive: Level targets provide clear completion criteria
- Negative: 5% tolerance may drift over time
- Negative: Entries at wrong level are hard to detect automatically (requires semantic understanding)

**Alternatives:**
- Frequency-based levels: More accurate but harder to define and validate
- Dynamic level assignment: Requires ML or corpus analysis
