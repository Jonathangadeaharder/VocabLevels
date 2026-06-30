---
id: SPEC-VOCAB-LEVELS
kind: spec
title: VocabLevels — Trilingual CEFR Vocabulary Manager
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
checksum: 4670916a3185a182c40d7b2391f49f0d44b3fb3b8fa9e0e01a951f7df3c517a0
---

> Imported legacy SPEC artifact from `docs/superpowers/specs/vocab-levels-design.md`. Keep future lifecycle work in OpenSpec.

---
checksum: c0af21f75e26a3bb61b67ca9820dbafb2be05c35025b935b3c04c11777d64a2f
---

## Overview

A CLI tool and CSV database for managing multilingual vocabulary organized by CEFR proficiency levels (A1-C1). Provides CRUD operations, quality validation, and cross-language lookup.

## Key Decisions

| Decision | Choice |
|----------|--------|
| Data format | CSV (one file per language/level) |
| Schema | Lemma + 2 configured translations per language |
| Levels | A1(600), A2(600), B1(1000), B2(2000), C1(4000) |
| CLI framework | argparse subcommands |
| Validation | Standalone check_quality.py |
| Package manager | uv |

## Directory Structure

```
VocabLevels/
├── english/               # A1.csv, A2.csv, B1.csv, B2.csv, C1.csv
│   └── *.csv              # Columns: English_Lemma, German_Translation, Spanish_Translation
├── german/                # A1.csv, A2.csv, B1.csv, B2.csv, C1.csv
│   └── *.csv              # Columns: German_Lemma, English_Translation, Spanish_Translation
├── spanish/               # A1.csv, A2.csv, B1.csv, B2.csv, C1.csv
│   └── *.csv              # Columns: Spanish_Lemma, English_Translation, German_Translation
├── vocab_manager.py       # CLI: add, remove, move, update, find, lookup, lint
├── check_quality.py       # Structural validation
├── audit_lemmatization.py # Inflected form detection
└── cleanup_inflections.py # Remove/merge inflected forms
```

## CLI Commands

| Command | Description | Example |
|---------|-------------|---------|
| add | Add lemma with translations | `add german A1 Haus house casa` |
| remove | Remove lemma from all levels | `remove german Haus` |
| move | Move lemma to different level | `move german A2 Haus` |
| update | Update translations or rename | `update german Haus --t1 house` |
| find | Find lemma in a language | `find german Haus` |
| lookup | Search across all languages | `lookup house` |
| lint | Run quality checks | `lint` |

## Data Flow

```
User Input → vocab_manager.py → CSV Read/Write → File System
                  │
                  ├── add: read level → append → sort → write
                  ├── remove: read all levels → filter → write
                  ├── move: find → remove from source → add to target
                  ├── update: find in levels → modify fields → write
                  └── lint: check_quality.py per language
```
