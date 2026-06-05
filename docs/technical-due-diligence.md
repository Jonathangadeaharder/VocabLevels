---
id: TDD-VOCB
kind: tdd
title: VocabLevels
description: >-
  Trilingual CEFR vocabulary database with ~24K entries across English, German,
  Spanish
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

## Executive Summary

VocabLevels is a trilingual CEFR vocabulary database containing approximately 24K entries across 15 CSV files (English, German, Spanish, levels A1-C1) managed by a 279-line Python CLI. Zero external dependencies, CodeRabbit configured for CSV-aware PR review, and ruff/pyright quality scripts are strengths. But the project has zero test files, no CI/CD pipeline for quality gates (SonarCloud static analysis configured), no pyproject.toml despite using uv, and duplicated configuration across four scripts. Data integrity relies entirely on `check_quality.py` rather than a formal test suite. Risk is low because the project has no network, auth, or user-input attack surface.

## Scope

Assessed all 4 Python scripts, all 15 CSV data files, the CodeRabbit configuration, AGENTS.md quality gate documentation, and the overall project structure. Excluded the upstream CEFR standard itself and the accuracy of individual vocabulary entries.

## Architecture

Trilingual vocabulary data stored in 15 CSV files (3 languages x 5 levels) with consistent 3-column structure per file (lemma, translation 1, translation 2). A 279-line CLI manager (`vocab_manager.py`) provides 7 commands: lint, find, add, remove, move, update, lookup. Quality scripts include `check_quality.py` (149 lines, 10+ validation criteria), `audit_lemmatization.py` (63 lines), and `cleanup_inflections.py` (74 lines). CodeRabbit provides AI PR review with path-specific instructions for each language.

## Tech Stack

- **Language:** Python 3.11+ (via uv)
- **Package Manager:** uv
- **Linter:** ruff (via uvx)
- **Type Checker:** pyright (via uvx)
- **PR Review:** CodeRabbit (configured)
- **Data Format:** CSV (UTF-8)
- **Runtime Dependencies:** None (stdlib only)

## Code Quality

No formal test suite exists. Code quality relies on ruff linting and pyright type checking, documented in AGENTS.md but not automated in CI. The CLI tool has clean argument parsing, proper exit codes, and modular command handlers. `check_quality.py` validates 10+ criteria including structure, row counts within 5% of target, duplicates, whitespace, special characters, and translation collisions. The `LANGS` configuration dictionary is duplicated across 4 files (vocab_manager.py, check_quality.py, audit_lemmatization.py, cleanup_inflections.py) in varying forms -- any schema change requires updating all copies. No pyproject.toml exists despite using uv.

## Security

Zero external dependencies eliminates supply chain risk entirely. The CLI uses stdlib only (csv, re, sys, pathlib, argparse). No network requests, no authentication, no user-facing server. CSV injection (formulas starting with `=` or `+`) is not validated. No backup or transaction mechanism protects CSV file integrity during write operations.

## Scalability & Performance

The data set is ~24K CSV rows across 15 files. The CLI's O(n) search over CSV files is adequate at this scale. No performance bottlenecks exist. Row count targets (600/600/1000/2000/4000 per language per level) are validated within 5% tolerance by check_quality.py.

## Operations & DevOps

A SonarCloud GitHub Actions workflow (`.github/workflows/sonarcloud.yml`) runs on push/PR to main, providing static analysis. CodeRabbit is configured for AI PR review with excellent language-specific instructions for each CSV file. Quality gates (ruff check, ruff format, pyright) are documented in AGENTS.md but must be run manually — no CI workflow enforces them. No Dependabot.

## Dependencies & Third-Party Risk

Zero external runtime dependencies -- stdlib only. No supply chain risk. ruff and pyright are used ephemerally via `uvx` with no pinned versions. CodeRabbit is a third-party service for PR review but is non-critical.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Zero tests -- no regression protection for CLI commands | High | Medium | Add pytest smoke tests for all 7 CLI commands |
| No CI/CD -- quality gates are manual-only | High | Medium | Add GitHub Actions workflow for ruff + pyright + smoke tests |
| Duplicate LANGS config in 4 files -- schema changes require 4 edits | High | Low | Extract to shared module (vocab_config.py) |
| No pyproject.toml -- uv lacks project metadata and tool config | High | Low | Create pyproject.toml with ruff/pyright config |
| CSV write corruption -- no backup or transaction mechanism | Medium | Medium | Add atomic write pattern (write to temp, rename) |
| CSV injection -- formulas starting with = or + not validated | Low | Low | Add validation for formula-like cell content |

## Recommendations

1. **Create pyproject.toml** with project metadata, ruff and pyright tool configuration, and pytest test configuration to enable `uv sync` with proper dependency management.
2. **Add pytest smoke tests** for all 7 CLI commands (find, add, remove, move, update, lookup, lint) using temporary CSV fixtures.
3. **Add GitHub Actions CI pipeline** that runs `ruff check`, `ruff format --check`, `pyright`, and smoke tests on every PR.
4. **Extract shared LANGS configuration** to a `vocab_config.py` module imported by all 4 scripts.
5. **Add atomic write pattern** to CSV write operations -- write to a temporary file and rename -- to prevent half-written CSV state on failure.
