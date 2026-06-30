---
id: TDD-VOCB
kind: tdd
title: VocabLevels
description: >-
  Multilingual CEFR vocabulary database with 26K+ entries across English,
  German, Spanish, Arabic, French, and Swedish
status: draft
date: 2026-05-17T00:00:00.000Z
authors: []
reviewers: []
risk_level: low
scope_type: project
tags:
  - python
  - vocabulary
  - cefr
  - csv-data
related: []
checksum: ac3a6189804fbd794d8edabcde24874ff957bea696fd3a0cd390ecee936ccc58
---

> Imported legacy TDD artifact from `docs/technical-due-diligence.md`. Keep future lifecycle work in OpenSpec.

## Executive Summary

VocabLevels is a multilingual CEFR vocabulary database containing 26,818 entries across 18 CSV files. English, German, and Spanish have complete A1-C1 sets; Arabic, French, and Swedish are A1-only in-progress languages. The project now has a `pyproject.toml`, locked dev tools, pytest coverage, and a shared `vocab_schema.py` source of truth for language columns. Risk remains low because the project has no network, auth, or user-facing server.

## Scope

Assessed the Python scripts, all 18 CSV data files, the CodeRabbit configuration, AGENTS.md quality gate documentation, and the overall project structure. Excluded the upstream CEFR standard itself and the accuracy of individual vocabulary entries.

## Architecture

Multilingual vocabulary data is stored in 18 CSV files with a consistent 3-column structure per file (lemma, translation 1, translation 2). The CLI manager (`vocab_manager.py`) provides 7 commands: lint, find, add, remove, move, update, lookup. Quality scripts include `check_quality.py`, `audit_lemmatization.py`, and `cleanup_inflections.py`. CodeRabbit provides AI PR review with path-specific instructions for each language family.

## Tech Stack

- **Language:** Python 3.11+ (via uv)
- **Package Manager:** uv
- **Linter:** ruff (via `uv run`)
- **Type Checker:** pyright (via `uv run`)
- **PR Review:** CodeRabbit (configured)
- **Data Format:** CSV (UTF-8)
- **Runtime Dependencies:** None (stdlib only)

## Code Quality

The test suite covers CLI commands, quality validation, lemmatization audits, and cleanup behavior with a coverage gate. The CLI tool has clean argument parsing, proper exit codes, and modular command handlers. `check_quality.py` validates 10+ criteria including structure, row counts within 5% of target, duplicates, whitespace, special characters, and translation collisions. The `LANGS` configuration now lives in `vocab_schema.py` and is imported by the production scripts and tests.

## Security

Zero external dependencies eliminates supply chain risk entirely. The CLI uses stdlib only (csv, re, sys, pathlib, argparse). No network requests, no authentication, no user-facing server. CSV injection (formulas starting with `=` or `+`) is not validated. CLI-owned writes use temp-file replacement, but concurrent CSV edits still have no transaction isolation.

## Scalability & Performance

The data set is 26,818 CSV rows across 18 files. The CLI's O(n) search over CSV files is adequate at this scale. No performance bottlenecks exist. Row count targets (600/600/1000/2000/4000 per complete language) are validated within 5% tolerance by check_quality.py.

## Operations & DevOps

A SonarCloud GitHub Actions workflow (`.github/workflows/sonarcloud.yml`) runs on push/PR to main, providing static analysis. CodeRabbit is configured for AI PR review with language-specific instructions for each tracked CSV language. Local quality gates (`ruff`, `pyright`, pytest) are pinned in the `uv` dev group and documented in AGENTS.md. No Dependabot.

## Dependencies & Third-Party Risk

Zero external runtime dependencies -- stdlib only. Runtime supply chain risk is minimal. ruff, pyright, pytest, and pytest-cov are pinned through the `uv` dev dependency group. CodeRabbit is a third-party service for PR review but is non-critical.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Data-quality failures can accumulate in CSV files | Medium | Medium | Keep `check_quality.py` in the local/CI gate and review failed rows directly |
| No CI/CD -- quality gates are manual-only | High | Medium | Add GitHub Actions workflow for ruff + pyright + smoke tests |
| Concurrent CSV edits can overwrite each other | Medium | Medium | Keep edits serialized; CLI-owned writes use temp-file replacement |
| CSV injection -- formulas starting with = or + not validated | Low | Low | Add validation for formula-like cell content |

## Recommendations

1. **Add a CI quality workflow** that runs `ruff format --check`, `ruff check`, `pyright`, and pytest on every PR.
2. **Add CSV injection validation** for formula-like cell content.
3. **Promote in-progress language levels intentionally** by adding data files and one `vocab_schema.py` update per language.
