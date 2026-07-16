# Gemma QA pipeline

The Gemma QA commands review CEFR CSVs and propose handcrafted CoNLL-U training
sentences for English, German, Spanish, Arabic, French, Swedish, Chinese, and
Dutch. Model output is not assumed correct. Deterministic validation, independent
model review, checkpointing, and manual review remain required quality gates.

## Configuration

Set the API key only in the process environment:

```bash
export GEMINI_API_KEY=...
```

The pipeline uses three independent resources:

| Resource | RPM | input TPM | RPD | Notes |
| --- | ---: | ---: | ---: | --- |
| `gemma-4-31b-it` | 30 | 16,000 | 14,400 | `generateContent` + JSON schema |
| `gemma-4-26b-a4b-it` | 30 | 16,000 | 14,400 | parallel Gemma worker |
| `antigravity-preview-05-2026` | 60 | 100,000 | **100** | Interactions API; **no structured output**; RPD hard-fails at 100 |

TPM is input tokens per minute only (not input+output). Gemma prompt packing
targets up to **14,500** estimated input tokens per request. Antigravity CEFR
prompts are capped tighter (**8,000**) because each `interactions.create` costs
**one of 100 daily requests** — keep prompts short, `tools=[]`, `background=false`.
Antigravity is scheduled as a **tertiary adjudication** worker when RPD remains;
Gemma stays primary for generation/review. Quota reservations and checkpoints
live under `.gemma_qa/`; API keys are never written there.

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

## Resumable scale runs

```bash
uv run python -m scripts.gemma_qa scale \
  --root . \
  --lemmatizer-root ../german-spanish-english-eurobert-lemmatizer
```

Scale defaults to all eight languages, all five levels, both phases, 20
handcrafted sentences per task, proposal-only output, and quality-first CEFR
review. Narrow a run with `--languages`, `--levels`, or `--phase`:

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

Task status, attempts, output paths, and sanitized errors are stored in
`.gemma_qa/scale.sqlite3`. Successful tasks are skipped on resume. Failed tasks
remain failed unless `--retry-failed` is supplied. Independent tasks continue
after a failure, and the command exits nonzero when any selected task remains
failed. One CEFR task and one handcraft task may run concurrently; each task
owns its client and SQLite connections.

```bash
uv run python -m scripts.gemma_qa status --root .
```

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
