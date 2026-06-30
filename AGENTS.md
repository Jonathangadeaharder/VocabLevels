# AGENTS.md — VocabLevels

## Build & Test Commands

```bash
uv sync
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run python -m pytest
```

## PR Instructions

- Branch: feature/*, fix/*, chore/*
- Title: `<type>(<scope>): <description>`
- Types: feat, fix, docs, style, refactor, perf, test, build, ci, chore
- Run the quality gates before committing
- One logical change per commit
