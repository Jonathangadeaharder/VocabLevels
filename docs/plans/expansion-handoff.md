# Handoff: VocabLevels coverage expansion (Issue #50)

## Read this first: the previous handoff overstated the state

This document replaces a version that reported source work as completed which
is not in the worktree. Everything below was re-measured on 2026-08-20 against
the actual repository. Where the earlier text and the repository disagreed, the
repository won and the deviation is named, so the next worker does not plan
against fiction.

Re-verify with the commands in section 9 before you trust any number here,
including mine.

## 1. Objective, and the number that is actually enforced

Make every ordered pair of the eight supported languages pass criterion 4 in
`check_data_contract.py <delivery>`, exit 0.

**The gate enforces 60 %, not 80 %.** `check_data_contract.py:42` reads
`MIN_COVERAGE = 0.60`, line 269 prints ">= 60 %", and `main()` returns 1 on any
LOW pair. The previous handoff said 80 % in its title, its objective and its
Phase C. Nothing in this repository requires 80 %.

Decide which is authoritative before you start, because the difference is
roughly 30 % more concepts per language:

- If 60 % is the target, the gate is already the specification and you change
  no code.
- If 80 % is the target, raise `MIN_COVERAGE` in the same commit that starts
  the expansion, so the gate and the goal cannot drift apart again.

The contract file the old handoff cites,
`docs/specs/2026-08-19-vocab-data-contract.md`, **does not exist in this
repository**. It lives in `Jonathangadeaharder/Vidiom`. The only spec here is
`docs/specs/vocab-levels-design.md`.

## 2. Measured baseline, 2026-08-20

Rebuilt from the current worktree, untracked expansion files included:

```
build:  ar 9484  de 7352  en 7948  es 8634  fr 7973  nl 6398  sv 5575  zh 11863
gate:   criterion 1: 0   criterion 2: 0   criterion 3: 0
        criterion 4: 0 of 56 pairs pass, median 29 %
        criterion 5: 6175 blank or non-ASCII glosses
        RESULT: FAIL, exit 1
```

Best pairs: sv->en 59 %, nl->en 58 %, es->en 57 %, fr->en 48 %, de->en 47 %.
Worst: zh->ar 18 %, ar->zh 18 %, zh->sv 19 %, zh->de 19 %, ar->sv 19 %.

Every pair is below even the 60 % line. The best pair is one point short.

## 3. What the previous worker did, and where it deviated

Each item is a fact from the worktree, not a summary of a summary.

### 3.1 The reported source completions do not exist

The old handoff listed manual completions: Arabic C1 250 rows, Arabic media 230
rows, Arabic B2 834 rows, Chinese "remaining blanks completed", Swedish B2 282
rows, Dutch 44 B2 glosses.

`git diff` and `git diff --cached` are **empty**. No tracked source CSV differs
from `6205575`. The blanks are still there:

```
blank English glosses in tracked sources: 5586
  chinese 3709   arabic 1545   swedish 288   dutch 44
```

The delivery counts 6175 including expansion rows. So Phase A is not partly
done, it is **not started**. Plan for the full 6175, not for a remainder.

### 3.2 Unsupported language artifact still present

`portuguese/expansion-manual-20260820.csv`, 279 rows, created after the
constraint that forbids Portuguese was written down. The old handoff itself
demanded its removal in step 2 and it is still there. Delete the directory, do
not merge any of it.

### 3.3 Overlap rules were not enforced at merge time

The constraint was: coordinator merges only valid, non-overlapping rows. In the
merged `{lang}/expansion.csv` files, **725 of 2065 rows duplicate a
`(lemma, POS)` that already exists in that language's source CSVs**:

```
chinese 168/273 (62 %)   french 234/408 (57 %)   spanish 174/626
dutch    58/138          arabic  54/196          german    26/190
swedish  11/60           english   0/174
```

Those rows cannot raise coverage; they only inflate the file. There are also
**46 duplicate keys inside single expansion files** (arabic 18, spanish 12,
french 8, dutch 5, swedish 2, chinese 1).

### 3.4 Batch size exceeded

`spanish/expansion-worker-002.csv` holds 398 rows against a stated maximum of
200. `spanish/expansion.csv` at 626 rows and `french/expansion.csv` at 408 are
merges of several batches, which is fine, but the worker artifact is not.

### 3.5 Line endings broken in one artifact

`spanish/expansion-worker-manual-20260820-1522.csv` is entirely CRLF (201 CRLF
sequences, 200 rows). Every other expansion file is LF. Merging it as-is flips
line endings for its rows and produces exactly the oversized diff section 7
warns about.

### 3.6 Artifacts written into `__pycache__`

`__pycache__/expansion.csv` exists, header-only. Something resolved an output
path relative to a cache directory. Delete it and check the writer before
running any generator again.

### 3.7 Naming and provenance are unusable

Four conventions in one tree: `expansion-batch-0001.csv`,
`expansion-worker-00N.csv`, `expansion-worker-manual-<date>.csv`,
`expansion-worker-manual-<date>-<time>.csv`, plus QA notes as both
`*.qa.md` and `*-qa.md`. Only four of the 39 untracked artifacts carry a QA
note at all, so for most rows there is no record of who reviewed them.

A coordinator cannot decide what to merge from this. Pick one pattern before
dispatching anything, and require the QA note as part of the artifact.

### 3.8 A stale delivery directory sits in the repo

`delivery-contract/` is untracked and contains `ar.tsv`, `de.tsv`, `en.tsv`,
`es.tsv` and siblings. The gate is meant to run against a freshly built
directory, by convention under `/tmp`. Do not read this one, and do not commit
it. Add it to `.gitignore` or delete it.

## 4. One rule from the old handoff that is wrong

> "a generated gloss == the foreign lemma is junk, reject it"

Applied literally this rejects legitimate cognates. Real rows in the current
files: `abrupt`, `abstract`, `accent` (nl), `abject`, `adjacent`,
`compendium` (fr), `arena`, `audible`, `audio` (es), `Alarm`, `Chaos`,
`Definition`, `Million` (de), `abbot`, `absolution` (sv). All 174 English
expansion rows have gloss == lemma by construction.

The real defect the rule was aiming at is a gloss that is the foreign lemma
**without being English**, for example `词` or `abfassung` in the English
column. Check that the gloss is a real English word, not that it differs from
the lemma.

## 5. Scope, unchanged

Supported: `en`, `de`, `es`, `fr`, `sv`, `ar`, `nl`, `zh`. Do not create,
retain, merge or review artifacts for any other language.

## 6. Execution constraints, each with the failure it exists to prevent

- Manual authoring by parallel subagents, not `scripts/expand_concepts.py`
  (user directive, 2026-08-20: serial ~2k concepts/h, plus CSV-rewrite and
  line-ending damage). Use the script only as a prompt and data-model
  reference.
- Maximum 200 concepts per batch. Violated once, see 3.4.
- Independent QA per batch, and the QA note ships with the artifact. Missing
  for most current artifacts, see 3.7.
- One artifact naming pattern, decided before dispatch, see 3.7.
- Coordinator merges only rows that are new against the source CSVs **and**
  unique within the merge. Violated 725 and 46 times, see 3.3.
- Source CSV edits keep four columns and the original mixed LF/CRLF endings.
  Expansion files are LF. Violated once, see 3.5.
- A gap beats a wrong entry. Drop when uncertain.

## 7. Required next steps

1. Freeze: no worker writes while you audit.
2. Delete `portuguese/`, `__pycache__/expansion.csv`, `delivery-contract/`.
3. Decide 60 % or 80 % (section 1) and record the decision in the issue.
4. Audit every untracked artifact against section 6. Reject rather than repair
   anything whose provenance or QA note is missing; re-authoring 200 concepts
   is cheaper than trusting an unreviewed batch.
5. Rebuild the merged `{lang}/expansion.csv` files from the artifacts that
   survive step 4. Expect them to shrink: at least 725 rows are dead weight.
6. Phase A first, all 6175 blanks. Blank rows cannot count for any pair, and
   zh alone holds 3709 of them while every zh pair sits at 18 to 22 %.
   `/tmp/expand_state/.blankfill.jsonl` holds 2531 approved fills from an
   aborted run; it is unverified, contains known junk (`deadly` for `"dödlig`
   with an empty POS), and its writes were rolled back. Re-QA before use.
7. Phase B: expand only pairs still below the decided threshold, in batches of
   at most 200, prefer the minimal set per target over the full universe.
8. Re-gate after every wave. Criteria 1, 2, 3 and 5 must stay 0.
9. Quality gates and PR, section 10.

## 8. Concept universe

- Universe: 23,619 `(gloss_norm, POS)` concepts, measured on the delivery.
- Missing to the full universe: ar 16,793; de 18,055; en 15,830; es 16,648;
  fr 16,530; nl 18,290; sv 19,119; zh 17,680.
- Minimal set for the 80 % reading, row weighted and target only, about 46 %
  smaller: ar 9,451; de 10,266; en 7,711; es 8,756; fr 8,813; nl 10,493;
  sv 11,099; zh 10,277, roughly 76,866 rows. For 60 % it is smaller again;
  measure after Phase A rather than reusing these numbers.
- Simulated full-universe coverage still caps at 56 to 63 % for zh->X while
  zh's blanks remain. That is why Phase A comes first.

## 9. Verify this document

```bash
cd ~/projects/vidiomtm/VocabLevels
git status --short                     # 39 untracked artifacts at handoff
git diff --stat && git diff --cached --stat   # both empty: sources untouched
env -u PYTHONPATH PYTHONPATH=. .venv/bin/python build_contract_delivery.py /tmp/vocab_audit
env -u PYTHONPATH PYTHONPATH=. .venv/bin/python check_data_contract.py /tmp/vocab_audit
echo $?                                # 1 at handoff
grep -n MIN_COVERAGE check_data_contract.py
```

Blank count per language, overlap against sources, duplicate keys inside a
merge and CRLF damage are all one short Python pass over the CSVs; the numbers
in sections 2 and 3 came from exactly that, not from a worker summary.

## 10. Quality gates and environment

- `env -u PYTHONPATH .venv/bin/python -m pytest tests/test_build_contract_delivery.py --no-cov`
  (8 tests, must stay green). `--no-cov` because the global threshold is 80 %.
- `uv run ruff format --check .`, `uv run ruff check .`, `uv run pyright`.
- `env -u PYTHONPATH` is required everywhere: the global `PYTHONPATH` from
  `~/.zshenv` points at a broken pydantic and breaks imports.
- Full pytest suite: never `-n auto`, cap xdist at 4 workers. 18 workers OOM
  this machine.
- Branch `feat/data-harmonization-pipeline`, conventional commits, one logical
  change per commit.

## 11. Meta-speed endpoint, if you use a model at all

- `POST http://localhost:4000/v1/chat/completions`, local router, no auth.
- `model: "meta-speed"`, `temperature: 0`, `reasoning_effort: "none"`,
  `response_format: {"type":"json_object"}`.
- The prompt must begin with "Output JSON only." and end with the JSON shape,
  otherwise the content is empty and the answer lands in `reasoning_content`.
- Roughly 10 s per request on any alias; eight parallel workers reached about
  9,700 approved concepts per hour.
- Do not restart the router without checking `~/projects/agent-utils/meta-router/`.
- Alternate: `https://chat.model.tngtech.com/v1` with
  `safe run -s SKAINET_API_KEY -- ...`.

## 12. Files

- `build_contract_delivery.py`: pipeline (`load_csv_records`,
  `load_expansion_records`, `build_delivery_rows`, `clean_gloss`, `GLOSS_ASCII`).
- `check_data_contract.py`: the gate. `MIN_COVERAGE = 0.60` at line 42.
- `scripts/expand_concepts.py`: reference only, not the execution path.
- `tests/test_build_contract_delivery.py`: 8 tests.
- `vocab_schema.py`: `LEVELS` (A1..C1), `LANG_DIRS` (`english/en` ... `chinese/zh`).
- Level CSVs: `{lang}/{A1..C1}.csv`, header
  `{Lang}_Lemma,English_Lemma,Chinese_Lemma,POS`.
- Expansion: `{lang}/expansion.csv`, header
  `{Lang}_Lemma,English_Lemma,Chinese_Lemma,POS,CEFR`, LF endings.

## 13. What the earlier handoff got right

Worth keeping, because re-deriving it costs hours: the router details in
section 11, the line-ending hazard in section 6, the `env -u PYTHONPATH` trap,
the xdist cap, the universe measurements in section 8, and the judgement that a
gap beats a wrong entry. The junk-row cleanup in `852b709` (4,147 machine-fill
rows, 302 German glosses embedding their lemma, the `GLOSS_ASCII` fix) is
committed and real; criteria 1, 2 and 3 are 0 because of it.
