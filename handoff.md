# Handoff — Gemma CEFR + handcraft QA (2026-07-17)

## Goal

Finish Gemma-backed CEFR + handcraft QA (plan: Gemma CEFR/CoNLL pipeline).

- **CEFR:** fix/refill VocabLevels CSVs; quality-first; target 600/level when refill requested.
- **Handcraft:** generate clean CoNLL-U under lemmatizer `data/handcraft/{lang}/train|val/` (lowercase filenames).
- **Gold UD** (`data/gold/`) stays untouched.
- Auth: `GEMINI_API_KEY` from env / `~/.zshenv` only — never commit keys.


## Provider (updated 2026-07-17)

Google Gemini retired for this pipeline. Use **TNG** `chat.model.tngtech.com`:

| Role | Model id |
|------|----------|
| Primary dual-review A | `Qwen/Qwen3.5-397B-A17B-FP8` |
| Independent dual-review B | `Qwen/Qwen3.6-35B-A3B-FP8` |
| Adjudication | `google/gemma-4-31B-it` |

Auth: `export API_KEY=...` (Bearer). Legacy `GEMINI_API_KEY` still accepted.

## Current status (German A1)

| Item | Value |
|------|------:|
| Committed `german/A1.csv` | **581** rows (+ header → 582 lines) |
| Target | 600 |
| Gap remaining | **19** |
| Validated freeze | resynced on last manual-review apply |
| Quality stop | **YES** (2026-07-17) — two consecutive zero-keep gap rounds (r9, r10) |
| Issue | https://github.com/Jonathangadeaharder/VocabLevels/issues/33 |

Handcraft DE: `../german-spanish-english-eurobert-lemmatizer/data/handcraft/de/train/a1.conllu` — 20 sentences, validators green.

## Yield trend

| Round | Gap generated | Kept after audit | A1 after |
|------:|--------------:|-----------------:|---------:|
| 7 | 19 | 2 | 579 |
| 8 | 19 | 2 | 581 |
| 9 | 19 | **0** | 581 |
| 10 | 13 | **0** | 581 |

## Done this session

1. Round-8 audit/apply: kept `zart`, `mittags` → 581.
2. Round-9/10 audits: all drop; quality stop short of 600.
3. Gap refill bugfix: `protect_accepted=True` so frozen loanword cognates (Hotel, Hand, Name, …) do not fail `_assert_language_clean` during gap refill.
4. `manual-review --lang` generalized to all `LANGUAGE_DIRECTORIES`; language gates via `cefr_row_issues` for all profiles.
5. Docs: `docs/gemma-qa.md` updated for multi-lang manual-review.
6. Tests: multi-lang Spanish smoke + protect_accepted loanword test.
7. **Runtime tracing** (`scripts/gemma_qa/trace.py`): JSONL + stderr for model I/O, thoughts, quota waits, novel rejects, scale/refill/cefr lifecycle. See `docs/gemma-qa.md` Runtime tracing. Tail: `tail -f .gemma_qa/events.jsonl` or `status --events 40`.

## Not done

- Scale remaining langs × levels (CEFR quality-first + agent audit apply).
- Handcraft all codes × levels (20 sentences).
- Commit large worktree; open/update PR for issue #33.
- Forced exact TARGET counts for non-A1 DE (out of scope quality-first).

## Immediate next steps

1. Commit code + DE A1 + manual_reviews + docs on `fix/cefr-vocab-clean`.
2. Run scale CEFR quality-first (prefer `--single-model gemma-4-26b-a4b-it`), skip full re-review of DE A1.
3. Per proposal: agent audit → `manual-review --apply` with collisions.
4. Handcraft phase after CEFR milestones.
5. Prefer 26B single-model — 31B often HTTP 500.

## Critical design rules

### Validated rows are frozen

Once a row is clean (manual `--apply`/`--append`, or `seed-validated` / gap-refill seed), fingerprint lives in `.gemma_qa/validated.sqlite3`.

- Later `cefr` runs **must not** model-review or German-repair validated rows.
- Gap refill uses `protect_accepted=True` so committed rows skip final language-clean assert.

### Models & quotas

| Resource | ID | RPM | TPM | RPD |
|----------|-----|----:|----:|----:|
| Gemma 31B | `gemma-4-31b-it` | 30 | 16k | 14.4k |
| Gemma 26B | `gemma-4-26b-a4b-it` | 30 | 16k | 14.4k |
| Antigravity | `antigravity-preview-05-2026` | 60 | 100k | **100** |

## Key commands

```bash
cd /Users/jonathangadeaharder/projects/vidiomtm/VocabLevels
source ~/.zshenv

uv run python -m scripts.gemma_qa scale \
  --root . \
  --lemmatizer-root ../german-spanish-english-eurobert-lemmatizer \
  --phase cefr \
  --single-model gemma-4-26b-a4b-it

uv run python -m scripts.gemma_qa manual-review \
  --root . --lang spanish --level A1 \
  --input spanish/A1.proposed.csv \
  --decisions manual_reviews/spanish/A1/scale-r1 \
  --check-other-level-collisions --apply

uv run pytest tests/test_gemma_qa_refill.py tests/test_gemma_qa_manual_review.py -q --no-cov
```

Docs: `docs/gemma-qa.md`.
