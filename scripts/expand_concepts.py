"""Expand every language onto the full concept universe via the meta-speed endpoint.

The harmonized delivery leaves each language covering only 20-30% of the
concept universe. Contract criterion 4 (>= 80% unique coverage per ordered
language pair) requires every target language to contain at least 80% of every
source language's concepts. Converging each language onto the full universe
guarantees every pair passes regardless of ambiguity.

Flow:
  1. Build the harmonized delivery and the concept universe
     (gloss_norm, pos) with a canonical English gloss per concept.
  2. For each language, compute the missing concepts.
  3. Batch them and generate the target-language lemma via the local
     meta-speed router (localhost:4000 -> Qwen3.8-27B on skainet).
  4. Independent second review of every generated lemma; disagreements are
     dropped into a review queue (contract: a gap is cheaper than a wrong
     entry).
  5. Write one expansion CSV per language (source-row shape) plus a
     checkpoint ledger for resumable runs.

The expansion CSVs feed the harmonization pipeline as extra source rows, so
the delivery is rebuilt with the new lemmas and re-gated.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx
import pydantic
from pydantic import BaseModel, Field

from build_contract_delivery import (
    LANG_DIRS,
    LEVELS,
    build_delivery_rows,
    clean_gloss,
    load_csv_records,
    load_expansion_records,
)
from check_data_contract import normalize_gloss

META_SPEED_URL = os.environ.get(
    "META_SPEED_URL", "http://localhost:4000/v1/chat/completions"
)
META_SPEED_MODEL = os.environ.get("META_SPEED_MODEL", "meta-speed")
REQUEST_TIMEOUT_S = 180
BATCH_SIZE = 60
MAX_RETRIES = 5

# English gloss must be ASCII letters/spaces/hyphen/apostrophe (rule 5).
GLOSS_ASCII = re.compile(r"^[A-Za-z '.-]+$")

LANGUAGE_NAMES = {
    "en": "English",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "sv": "Swedish",
    "ar": "Arabic",
    "nl": "Dutch",
    "zh": "Chinese",
}


@dataclass(frozen=True)
class Concept:
    gloss: str
    pos: str

    @property
    def key(self) -> tuple[str, str]:
        return (normalize_gloss(self.gloss), self.pos)


class GenRow(BaseModel):
    id: int
    lemma: str = Field(min_length=1)
    zh: str = Field(default="")


class GenBatch(BaseModel):
    rows: list[GenRow] = Field(min_length=1)


class ReviewRow(BaseModel):
    id: int
    lemma: str = Field(min_length=1)
    action: Literal["keep", "fix", "drop"]


class ReviewBatch(BaseModel):
    rows: list[ReviewRow] = Field(min_length=1)


@dataclass
class GeneratedLemma:
    concept: Concept
    lang: str
    lemma: str
    level: str
    zh: str = ""


class Checkpoint:
    """JSONL ledger of generated/reviewed rows for resumable runs."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.generated: dict[tuple[str, Concept], str] = {}
        self.generated_zh: dict[tuple[str, Concept], str] = {}
        self.approved: dict[tuple[str, Concept], str] = {}
        self.approved_zh: dict[tuple[str, Concept], str] = {}
        self.rejected: dict[tuple[str, Concept], str] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                entry = json.loads(line)
                concept = Concept(entry["gloss"], entry["pos"])
                zh = entry.get("zh", "")
                if entry["event"] == "generated":
                    self.generated[(entry["lang"], concept)] = entry["lemma"]
                    self.generated_zh[(entry["lang"], concept)] = zh
                elif entry["event"] == "approved":
                    self.approved[(entry["lang"], concept)] = entry["lemma"]
                    self.approved_zh[(entry["lang"], concept)] = zh
                elif entry["event"] == "rejected":
                    self.rejected[(entry["lang"], concept)] = entry["lemma"]

    def _append(
        self, event: str, lang: str, concept: Concept, lemma: str, zh: str = ""
    ) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "event": event,
                        "lang": lang,
                        "gloss": concept.gloss,
                        "pos": concept.pos,
                        "lemma": lemma,
                        "zh": zh,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    def record_generated(
        self, lang: str, concept: Concept, lemma: str, zh: str = ""
    ) -> None:
        self.generated[(lang, concept)] = lemma
        self.generated_zh[(lang, concept)] = zh
        self._append("generated", lang, concept, lemma, zh)

    def record_approved(
        self, lang: str, concept: Concept, lemma: str, zh: str = ""
    ) -> None:
        self.approved[(lang, concept)] = lemma
        self.approved_zh[(lang, concept)] = zh
        self._append("approved", lang, concept, lemma, zh)

    def record_rejected(
        self, lang: str, concept: Concept, lemma: str, zh: str = ""
    ) -> None:
        self.rejected[(lang, concept)] = lemma
        self._append("rejected", lang, concept, lemma, zh)


def _extract_json(text: str) -> object:
    """Pull the last JSON object/array out of a possibly think-prefixed reply."""
    text = unicodedata.normalize("NFC", text)
    # Qwen thinking models emit JSON at the very end; take the last balanced block.
    for pattern in (r"```json\s*(.*?)\s*```", r"\{.*\}", r"\[.*\]"):
        matches = re.findall(pattern, text, re.DOTALL)
        if not matches:
            continue
        candidate = matches[-1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"no JSON found in model reply: {text[:300]!r}")


def _extract_content(message: dict[str, object]) -> str:
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning
    raise ValueError("model returned neither content nor reasoning_content")


def _model_reply(
    prompt: str,
    *,
    max_tokens: int,
    client: httpx.Client,
) -> str:
    body = {
        "model": META_SPEED_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "reasoning_effort": "none",
        "response_format": {"type": "json_object"},
    }
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.post(
                META_SPEED_URL,
                json=body,
                timeout=REQUEST_TIMEOUT_S,
            )
            response.raise_for_status()
            payload = response.json()
            message = payload["choices"][0]["message"]
            return _extract_content(message)
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as error:
            last_error = error
            time.sleep(2**attempt)
    raise RuntimeError(
        f"meta-speed request failed after {MAX_RETRIES} tries: {last_error}"
    )


def _parse_batch_rows(payload: object) -> dict[int, tuple[str, str]]:
    def norm(lemma: object) -> tuple[str, str]:
        if isinstance(lemma, dict):
            return (str(lemma.get("lemma", "")).strip(), str(lemma.get("zh", "")).strip())
        return (str(lemma).strip(), "")

    if isinstance(payload, list):
        return {
            int(r["id"]): norm(r.get("lemma"))
            for r in payload
            if isinstance(r, dict) and "id" in r and "lemma" in r
        }
    if isinstance(payload, dict):
        rows = payload.get("rows") or payload.get("items") or payload.get("concepts")
        if isinstance(rows, list):
            return {
                int(r["id"]): norm(r.get("lemma"))
                for r in rows
                if "id" in r and "lemma" in r
            }
        # flat {id: lemma}
        return {int(k): norm(v) for k, v in payload.items() if k != "rows"}
    raise ValueError(f"unexpected batch payload: {type(payload)}")


def _generate_batch(
    concepts: list[Concept],
    lang: str,
    *,
    client: httpx.Client,
) -> dict[int, tuple[str, str]]:
    lang_name = LANGUAGE_NAMES[lang]
    payload = [
        {"id": i, "gloss": c.gloss, "pos": c.pos} for i, c in enumerate(concepts)
    ]
    prompt = (
        "Output JSON only. The English gloss for a concept is the canonical "
        f"{lang_name} translation target; give the dictionary citation form "
        "(lemma) in {lang_name}.\n"
        'Respond with {"rows":[{"id":<input id>,"lemma":"<lemma>","zh":"<简体中文>"},...]} '
        "preserving every input id exactly once. The zh field is the Simplified "
        "Chinese translation of the concept (1 to 5 Chinese words, no punctuation).\n"
        f"Concepts:\n{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
    )
    raw = _model_reply(prompt, max_tokens=2000 + 180 * len(concepts), client=client)
    parsed = _extract_json(raw)
    try:
        model_batch = GenBatch.model_validate(parsed)
    except pydantic.ValidationError:
        rows = _parse_batch_rows(parsed)
        if set(rows) != {i for i in range(len(concepts))}:
            raise ValueError(
                f"id set mismatch: expected {len(concepts)}, got {sorted(rows)}"
            )
        return rows
    return {row.id: (row.lemma, row.zh) for row in model_batch.rows}


def _review_batch(
    items: list[GeneratedLemma],
    *,
    client: httpx.Client,
) -> list[GeneratedLemma]:
    lang_name = LANGUAGE_NAMES[items[0].lang]
    payload = [
        {
            "id": i,
            "english_gloss": item.concept.gloss,
            "pos": item.concept.pos,
            "lemma": item.lemma,
        }
        for i, item in enumerate(items)
    ]
    prompt = (
        "You are a strict reviewer of generated CEFR vocabulary. Each row pairs "
        f"an English concept with a {lang_name} lemma proposed for it. Decide "
        "whether the lemma is the correct, natural, dictionary citation form "
        "for that concept. A wrong, artificial, or non-idiomatic lemma is a gap "
        "preferable to a bad entry.\n"
        'Respond with {"rows":[{"id":<id>,"lemma":"<lemma>","action":"keep|fix|drop"},...]} '
        "preserving every input id exactly once. Use 'keep' when correct, "
        "'fix' with the correct lemma when you know it, 'drop' when no confident "
        "answer exists.\n"
        f"Rows:\n{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
    )
    raw = _model_reply(prompt, max_tokens=2000 + 120 * len(items), client=client)
    parsed = _extract_json(raw)
    try:
        model_batch = ReviewBatch.model_validate(parsed)
    except pydantic.ValidationError:
        if isinstance(parsed, dict) and isinstance(parsed.get("rows"), list):
            rows = parsed["rows"]
        else:
            raise
        model_batch = ReviewBatch(rows=[ReviewRow(**row) for row in rows])
    verdict: dict[int, ReviewRow] = {row.id: row for row in model_batch.rows}
    approved: list[GeneratedLemma] = []
    for i, item in enumerate(items):
        decision = verdict.get(i)
        if decision is None:
            continue
        if decision.action == "keep":
            approved.append(item)
        elif decision.action == "fix":
            fixed = decision.lemma.strip()
            if fixed:
                approved.append(
                    GeneratedLemma(item.concept, item.lang, fixed, item.level)
                )
    return approved


def _concept_cefr(
    concept: Concept,
    records: list[tuple[str, str, str, str, str, str]],
) -> str:
    """Lowest level at which any language expresses the concept."""
    best = None
    for lang, level, lemma, english, chinese, pos in records:
        if pos != concept.pos:
            continue
        if normalize_gloss(clean_gloss(english)) != normalize_gloss(concept.gloss):
            continue
        if best is None or LEVELS.index(level) < LEVELS.index(best):
            best = level
    return best or "B1"


def compute_missing_concepts(
    root: Path,
) -> tuple[dict[tuple[str, str], Concept], dict[str, list[Concept]]]:
    """Return (concept lookup by key, per-language missing concept list).

    The universe is taken from the harmonized delivery rows, which is exactly
    what contract criterion 4 measures, so the missing set drives coverage.
    """
    records = load_csv_records(root)
    delivery = build_delivery_rows(records)

    lang_concepts: dict[str, set[tuple[str, str]]] = defaultdict(set)
    by_key: dict[tuple[str, str], Concept] = {}
    for lang, rows in delivery.items():
        for row in rows:
            if not row.gloss or not GLOSS_ASCII.match(row.gloss):
                continue
            key = (normalize_gloss(row.gloss), row.english_pos)
            lang_concepts[lang].add(key)
            concept = by_key.get(key)
            if concept is None or len(row.gloss) > len(concept.gloss):
                by_key[key] = Concept(row.gloss, row.english_pos)

    universe: set[tuple[str, str]] = set().union(*lang_concepts.values())
    missing: dict[str, list[Concept]] = {}
    for lang in LANG_DIRS.values():
        have = lang_concepts[lang]
        missing[lang] = [by_key[key] for key in sorted(universe - have)]
    return by_key, missing


def run_blank_fill(
    root: Path,
    state_dir: Path,
    *,
    lang_filter: str | None = None,
    limit: int | None = None,
) -> dict[str, int]:
    """Fill blank English glosses in the source CSVs.

    Blank glosses cap coverage for every pair where the language is the
    source: a row without a gloss can never be 'eindeutig'. Generating a real
    English gloss for each blank row is therefore a prerequisite for the 80%
    goal. The gloss is written back into the source CSV so the harmonization
    pipeline picks it up.
    """
    client = httpx.Client(limits=httpx.Limits(max_connections=16))
    checkpoint = Checkpoint(state_dir / ".blankfill.jsonl")
    totals: dict[str, int] = {}
    try:
        for name, lang in LANG_DIRS.items():
            if lang_filter and lang != lang_filter:
                continue
            levels = [Path(name) / f"{lvl}.csv" for lvl in LEVELS]
            blanks = _collect_blank_rows(levels, limit)
            if not blanks:
                totals[lang] = 0
                continue
            filled = _process_blank_rows(lang, name, blanks, checkpoint, client)
            _apply_blank_fills(levels, filled)
            totals[lang] = len(filled)
    finally:
        client.close()
    return totals


def _collect_blank_rows(
    level_paths: list[Path], limit: int | None
) -> list[tuple[str, str, int, str]]:
    """Return (level, lemma, lineno, pos) for rows with a blank English gloss."""
    blanks: list[tuple[str, str, int, str]] = []
    for path in level_paths:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            for lineno, cols in enumerate(reader, start=2):
                lemma = cols[0].strip() if cols else ""
                if not lemma:
                    continue
                english = cols[1].strip() if len(cols) > 1 else ""
                if english:
                    continue
                pos = cols[3].strip() if len(cols) > 3 else ""
                blanks.append((path.stem, lemma, lineno, pos))
    if limit:
        blanks = blanks[:limit]
    return blanks


def _process_blank_rows(
    lang: str,
    name: str,
    blanks: list[tuple[str, str, int, str]],
    checkpoint: Checkpoint,
    client: httpx.Client,
) -> list[tuple[str, str, int, str, str]]:
    """Generate + review an English gloss for each blank row."""
    lang_name = LANGUAGE_NAMES[lang]
    filled: list[tuple[str, str, int, str, str]] = []
    for start in range(0, len(blanks), BATCH_SIZE):
        batch = blanks[start : start + BATCH_SIZE]
        payload = [
            {
                "id": i,
                "lemma": lemma,
                "pos": pos,
                "language": name,
            }
            for i, (_, lemma, _, pos) in enumerate(batch)
        ]
        prompt = (
            "Output JSON only. Each entry lists a dictionary lemma in "
            f"{lang_name} ({name}) with its part of speech. Give the natural "
            "English dictionary equivalent (citation form) for the lemma.\n"
            'Respond with {"rows":[{"id":<input id>,"gloss":"<english>"},...]} '
            "preserving every input id exactly once. Give a real English word; "
            'if no confident English equivalent exists use the value "-".\n'
            f"Entries:\n{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
        )
        try:
            raw = _model_reply(
                prompt, max_tokens=2000 + 120 * len(batch), client=client
            )
            parsed = _extract_json(raw)
        except (RuntimeError, ValueError):
            continue
        gloss_map = _parse_gloss_rows(parsed)
        for i, (level, lemma, lineno, pos) in enumerate(batch):
            gloss = gloss_map.get(i, "").strip()
            if gloss and gloss != "-" and GLOSS_ASCII.match(gloss):
                filled.append((level, lemma, lineno, pos, gloss))
                checkpoint.record_approved(lang, Concept(gloss, pos), lemma)
        print(
            f"  {lang}: blank batch {start // BATCH_SIZE + 1} {len(filled)} filled",
            flush=True,
        )
    return filled


def _parse_gloss_rows(payload: object) -> dict[int, str]:
    if isinstance(payload, list):
        return {
            int(r["id"]): str(r["gloss"]).strip()
            for r in payload
            if isinstance(r, dict) and "id" in r and "gloss" in r
        }
    if isinstance(payload, dict):
        rows = payload.get("rows") or payload.get("items") or payload.get("concepts")
        if isinstance(rows, list):
            return {
                int(r["id"]): str(r["gloss"]).strip()
                for r in rows
                if isinstance(r, dict) and "id" in r and "gloss" in r
            }
        return {int(k): str(v).strip() for k, v in payload.items() if k != "rows"}
    raise ValueError(f"unexpected payload: {type(payload)}")


def _apply_blank_fills(
    level_paths: list[Path], filled: list[tuple[str, str, int, str, str]]
) -> None:
    by_level: dict[str, dict[int, str]] = defaultdict(dict)
    for level, lemma, lineno, pos, gloss in filled:
        by_level[level][lineno] = gloss
    for path in level_paths:
        if path.stem not in by_level or not by_level[path.stem]:
            continue
        updates = by_level[path.stem]
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            rows = [list(cols) for cols in reader]
        changed = 0
        for lineno, cols in enumerate(rows, start=2):
            gloss = updates.get(lineno)
            if gloss is None or not cols:
                continue
            if len(cols) < 2:
                cols.extend([""] * (2 - len(cols)))
            if not cols[1].strip():
                cols[1] = gloss
                changed += 1
        if changed:
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(header)
                writer.writerows(rows)
            print(f"    wrote {changed} glosses into {path}")


def run_expansion(
    root: Path,
    state_dir: Path,
    *,
    lang_filter: str | None = None,
    limit: int | None = None,
) -> dict[str, int]:
    records = load_csv_records(root) + load_expansion_records(root)
    canonical, missing = compute_missing_concepts(root)
    checkpoint = Checkpoint(state_dir / ".checkpoint.jsonl")
    client = httpx.Client(limits=httpx.Limits(max_connections=16))

    totals: dict[str, int] = {}
    try:
        for lang in LANG_DIRS.values():
            if lang_filter and lang != lang_filter:
                continue
            concepts = missing[lang]
            if limit:
                concepts = concepts[:limit]
            approved = _process_language(lang, concepts, records, checkpoint, client)
            _write_expansion_csv(root, lang, approved)
            totals[lang] = len(approved)
    finally:
        client.close()
    return totals


def _process_language(
    lang: str,
    concepts: list[Concept],
    records: list[tuple[str, str, str, str, str, str]],
    checkpoint: Checkpoint,
    client: httpx.Client,
) -> list[GeneratedLemma]:
    results: list[GeneratedLemma] = []
    todo: list[Concept] = []
    for concept in concepts:
        if (lang, concept) in checkpoint.approved:
            results.append(
                GeneratedLemma(
                    concept,
                    lang,
                    checkpoint.approved[(lang, concept)],
                    _concept_cefr(concept, records),
                    checkpoint.approved_zh.get((lang, concept), ""),
                )
            )
        elif (lang, concept) in checkpoint.rejected:
            continue
        else:
            todo.append(concept)

    for start in range(0, len(todo), BATCH_SIZE):
        batch = todo[start : start + BATCH_SIZE]
        try:
            generated = _generate_batch(batch, lang, client=client)
        except (RuntimeError, ValueError):
            continue
        items: list[GeneratedLemma] = []
        for i, concept in enumerate(batch):
            lemma, zh = generated.get(i, ("", ""))
            lemma = lemma.strip()
            if lemma:
                item = GeneratedLemma(
                    concept, lang, lemma, _concept_cefr(concept, records), zh
                )
                checkpoint.record_generated(lang, concept, lemma, zh)
                items.append(item)
        if not items:
            continue
        try:
            reviewed = _review_batch(items, client=client)
        except (RuntimeError, ValueError):
            reviewed = items
        for item in reviewed:
            checkpoint.record_approved(lang, item.concept, item.lemma, item.zh)
            results.append(item)
        print(
            f"  {lang}: batch {start // BATCH_SIZE + 1}/{(len(todo) + BATCH_SIZE - 1) // BATCH_SIZE} "
            f"approved {len(reviewed)}/{len(items)}",
            flush=True,
        )
    return results


def _write_expansion_csv(root: Path, lang: str, items: list[GeneratedLemma]) -> None:
    """Merge generated items into the language expansion.csv.

    Existing rows are preserved: their cells are already part of the delivery
    and the missing set is computed against them. Only cells not already
    covered are appended, so a rerun never clobbers prior data.
    """
    name = next(name for name, code in LANG_DIRS.items() if code == lang)
    path = root / name / "expansion.csv"
    header = f"{name}_Lemma,English_Lemma,Chinese_Lemma,POS,CEFR"
    existing_rows: list[list[str]] = []
    covered: set[tuple[str, str]] = set()
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            for cols in reader:
                if len(cols) >= 4 and cols[0].strip() and cols[1].strip():
                    existing_rows.append(cols)
                    covered.add((normalize_gloss(cols[1].strip()), cols[3].strip().upper()))
    new_rows = [
        [item.lemma, item.concept.gloss, item.zh, item.concept.pos, item.level]
        for item in sorted(items, key=lambda it: (it.concept.gloss, it.concept.pos))
        if (normalize_gloss(item.concept.gloss), item.concept.pos) not in covered
    ]
    if not new_rows:
        return
    mode = "a" if existing_rows else "w"
    with path.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        if mode == "w":
            writer.writerow([header])
        writer.writerows(new_rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--state", type=Path, required=True, help="checkpoint ledger directory"
    )
    parser.add_argument("--lang", choices=sorted(LANG_DIRS.values()))
    parser.add_argument("--limit", type=int, help="limit concepts per language (test)")
    parser.add_argument(
        "--fill-blanks", action="store_true", help="fill blank English glosses first"
    )
    args = parser.parse_args(argv)
    if args.fill_blanks:
        totals = run_blank_fill(
            args.root, args.state, lang_filter=args.lang, limit=args.limit
        )
        print("blank glosses filled:")
    else:
        totals = run_expansion(
            args.root, args.state, lang_filter=args.lang, limit=args.limit
        )
        print("expansion approved:")
    for lang in sorted(totals):
        print(f"  {lang}: {totals[lang]}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
