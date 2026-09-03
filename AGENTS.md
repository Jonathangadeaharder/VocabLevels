# AGENTS.md — VocabLevels

## Build & Test Commands

```bash
uv sync
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run python -m pytest
```

Local pytest runs need a clean `PYTHONPATH`: `tests/conftest.py` fails fast
listing foreign entries. If an interactive shell pollutes it, run
`env -u PYTHONPATH uv run python -m pytest`. CI pins a clean env.

## PR Instructions

- Branch: feature/*, fix/*, chore/*
- Title: `<type>(<scope>): <description>`
- Types: feat, fix, docs, style, refactor, perf, test, build, ci, chore
- Run the quality gates before committing
- One logical change per commit
