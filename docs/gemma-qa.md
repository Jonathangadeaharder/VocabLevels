# Gemma QA pipeline

The Gemma QA commands review CEFR CSVs and propose handcrafted CoNLL-U training
sentences for English, German, Spanish, Arabic, French, Swedish, Chinese, and
Dutch. Model output is not assumed correct. Deterministic validation, independent
model review, checkpointing, and manual review remain required quality gates.

## Configuration

Set the API key only in the process environment (TNG gateway):

```bash
export API_KEY=...   # chat.model.tngtech.com
# optional legacy alias: GEMINI_API_KEY still accepted
```

Provider: **TNG OpenAI-compatible** `https://chat.model.tngtech.com/v1/chat/completions`
(not Google Gemini `generateContent`).

| Model id | Role | AA Intelligence (approx) |
| --- | --- | ---: |
| `Qwen/Qwen3.5-397B-A17B-FP8` | primary dual-review A / generation | ~34 |
| `Qwen/Qwen3.6-35B-A3B-FP8` | independent dual-review B | 32 |
| `google/gemma-4-31B-it` | adjudication on disagreement | 29 |

Requests use `reasoning_effort=none` and `response_format=json_schema` by default.
Soft local packing cap: **28,000** estimated input tokens per request (prompt
batching only). **No client-side RPM/TPM/RPD gate** — TNG load-balances.
Checkpoints live under `.gemma_qa/`; API keys are never written there.

## Individual tasks

CEFR commands use language directory names:

```bash
uv run python -m scripts.gemma_qa cefr \
  --root . --lang spanish --level B1
```

The default writes `spanish/B1.proposed.csv` and may remain below the nominal
CEFR row target after invalid rows and duplicates are removed. This
quality-first behavior avoids manufacturing low-quality filler. Exact refill is
explicit:

```bash
uv run python -m scripts.gemma_qa cefr \
  --root . --lang spanish --level B1 --refill-to-target
```

Refill first uses trusted English same-level concepts. Novel target-language
generation is used only when exact refill is requested and the trusted pivot is
insufficient. English reuses reviewed English concepts locally instead of
asking the model to translate English back into English.

Handcraft commands use ISO language codes:

```bash
uv run python -m scripts.gemma_qa handcraft \
  --vocab-root . \
  --lemmatizer-root ../german-spanish-english-eurobert-lemmatizer \
  --lang es --level B1 --count 20
```

The default output is
`data/handcraft/{code}/train/{level}.proposed.conllu` below the lemmatizer root,
with the level filename case-folded (for example, `a1.proposed.conllu`).
Use `--apply` only after reviewing the proposal; it writes the corresponding
`.csv` or `.conllu` destination.

Before generating, check the committed CEFR cell (structural citation gate;
loanword English-echo rows stay allowed):

```bash
uv run python -m scripts.gemma_qa handcraft-ready \
  --vocab-root . --lang de --level A1 --count 20
```

Exit 0 means enough clean rows for the requested sentence count. Prefer
`--phase handcraft` only after CEFR for that cell is applied; scale runs CEFR
before handcraft when `--phase both`, and blocks handcraft if a CEFR sibling
task exists and is not `succeeded`.

Training-data path (perfect lemma+UPOS → CoNLL-U): generate proposed → dual-model
validators (target lemma **and** UPOS on a token) + lemma_checker → audit →
`--apply` into `data/handcraft/{code}/train/`. Gold UD is never written.

## Resumable scale runs

```bash
uv run python -m scripts.gemma_qa scale \
  --root . \
  --lemmatizer-root ../german-spanish-english-eurobert-lemmatizer
```

Scale defaults to all eight languages, all five levels, both phases, 20
handcrafted sentences per task, proposal-only output, and quality-first CEFR
review. CEFR batches are independent. Active pool uses a **strategy pattern**
(`model_strategies.py`): every active model can run dual review *or*
adjudication (no role silos). Strategies encode endpoint, wire id, and
thinking strip subtleties.

**Active pool (fast):** Qwen 397B, Qwen 35B, GLM-5.2 internal, GLM-5.2
external, optional GLM-5.2-TEE. **Out of rotation (bottlenecks):** Gemma-4,
GLM-5.1-FP8.

Default batch concurrency ~4 (env `GEMMA_QA_BATCH_CONCURRENCY`). Per-model
HTTP cap 2 (`GEMMA_QA_PER_MODEL_INFLIGHT`). Selection prefers free slots.
`reasoning_effort=none` via GLM/Qwen strategies. Ledger-checkpointed per
`batch_id`.

### Hang recovery (auto)

Idle HTTP read alone can hang forever on slow-drip streams. Recovery layers:

| Layer | Default | Env |
| --- | --- | --- |
| Request wall-clock (hard POST cap) | 180s | `GEMMA_QA_REQUEST_WALL_S` |
| Idle read / write / pool | 120 / 120 / 60 | fixed in client |
| Per-model slot acquire | 120s | `GEMMA_QA_SLOT_ACQUIRE_TIMEOUT_S` |
| Dual-wait ceiling then pair rotate | 2×wall+60s | derived |
| Dual pair retries (timeout/422/transport) | 4 attempts | fixed in cefr |

Wall-clock failures retry at most twice per generate, then raise. CEFR rotates
the dual model pair. Slots release in `finally` so other batches can progress.
Resume is ledger-safe after kill or auto-fail.

Narrow with `--languages`, `--levels`, or `--phase`:

```bash
uv run python -m scripts.gemma_qa scale \
  --root . \
  --lemmatizer-root ../german-spanish-english-eurobert-lemmatizer \
  --languages de es --levels A1 A2 --phase cefr
```

Use `--refill-to-target` for exact CEFR refill and `--apply` for reviewed
destinations. Both are explicit because generated completeness is weaker than
reviewed lexical quality.

## Gap refill

When a level is below target after manual review, generate only the missing rows
into `{level}.gap.proposed.csv` without rewriting the committed CSV:

```bash
uv run python -m scripts.gemma_qa refill \
  --root . --lang german --level A1
```

Pass prior gap-audit decisions so rejected lemmas and English pivot concepts are
not retried:

```bash
uv run python -m scripts.gemma_qa refill \
  --root . --lang german --level A1 \
  --reject-decisions manual_reviews/german/A1/gap
```

After reviewing the gap proposal, merge accepted rows into the committed CSV
with `manual-review --append`:

```bash
uv run python -m scripts.gemma_qa manual-review \
  --root . --lang german --level A1 \
  --input german/A1.gap.proposed.csv \
  --decisions manual_reviews/german/A1/gap \
  --append --apply
```

`manual-review --lang` accepts any CEFR language directory
(`english`, `german`, `spanish`, `arabic`, `french`, `swedish`,
`chinese`, `dutch`). Language gates run via `cefr_row_issues` for all
profiles (German capitalization/infinitive rules, Arabic/Chinese script,
English-echo rejection, etc.).

Task status, attempts, output paths, and sanitized errors are stored in
`.gemma_qa/scale.sqlite3`. Successful tasks are skipped on resume. Failed tasks
remain failed unless `--retry-failed` is supplied. Independent tasks continue
after a failure, and the command exits nonzero when any selected task remains
failed. Phases run sequentially (CEFR, then handcraft when both are selected).
Each phase owns its client and SQLite connections.

```bash
uv run python -m scripts.gemma_qa status --root .
uv run python -m scripts.gemma_qa status --root . --events 40
```

## Runtime tracing

Every CLI command configures structured tracing:

- **stderr** human lines: `[INFO] generate.ok model=... duration_ms=... rows=...`
- **JSONL** append-only log: `.gemma_qa/events.jsonl` (gitignored via `.gemma_qa/`)

| Env | Effect |
| --- | --- |
| `GEMMA_QA_LOG_LEVEL` | `DEBUG` / `INFO` (default) / `WARN` / `ERROR` |
| `GEMMA_QA_LOG_BODIES=1` | Include prompt/response previews in events |
| `GEMMA_QA_LOG_PATH` | Override JSONL path |
| `GEMMA_QA_REQUEST_WALL_S` | Hard max seconds per HTTP POST (default 180) |
| `GEMMA_QA_SLOT_ACQUIRE_TIMEOUT_S` | Max wait for free model slot (default 120) |
| `GEMMA_QA_PER_MODEL_INFLIGHT` | Concurrent HTTP per model (default 2) |
| `GEMMA_QA_BATCH_CONCURRENCY` | Parallel CEFR batches (default ~4) |

Event kinds worth grepping for bottlenecks:

| Kind | Meaning |
| --- | --- |
| `generate.retry` | HTTP 429/5xx with backoff |
| `generate.wall_clock_timeout` | Hard POST deadline; stream closed |
| `generate.transport_error` | Timeout/connect flake (includes wall) |
| `cefr.dual_wait_ceiling` | Dual legs exceeded wait; pair rotate |
| `cefr.dual_retry` | Batch retried with different dual pair |
| `generate.parse_error` | Model JSON failed schema; repair loop |
| `semantic.validation_error` | Semantic identity/cardinality fail + row summary |
| `semantic.checkpoint_hit` | Skipped API via ledger |
| `novel.*` | Gap novel rounds: accepts, reject reason histogram |
| `refill.*` / `cefr.*` / `scale.*` | Phase lifecycle |

Thought/reasoning parts (`thought: true` in candidates) are extracted when present and logged as `thoughts`.

## Manual review

Proposal files are not training data or authoritative vocabulary until reviewed.
Inspect lexical sense, CEFR fit, citation lemma, UPOS, translations, sentence
naturalness, tokenization, and language-specific annotation. Apply only accepted
proposals. Existing manual decision files remain separate inputs and are never
changed by scale runs.

## Validated rows

Once a lang/level row is marked validated in `.gemma_qa/validated.sqlite3`, later
`cefr` runs skip model review and German language repair for that exact row
(fingerprint: NFC lang, level, lemma, english_lemma, chinese_lemma, upos). Only
unvalidated rows and gap fills are sent to the model. `manual-review --apply` and
`manual-review --append` resync validation from the committed CSV. Gap refill
seeds committed rows at start. Seed existing clean levels explicitly:

```bash
uv run python -m scripts.gemma_qa seed-validated \
  --root . --lang german --level A1
```
