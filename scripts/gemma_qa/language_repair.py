from __future__ import annotations

import csv
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from .client import GenerationResult
from .config import INPUT_BATCH_TOKEN_CAP, MODEL_26B, MODEL_31B, MODEL_IDS
from .languages import get_language, has_arabic_script, has_han_script
from .ledger import Ledger
from .packing import TiktokenEstimator, pack_records
from .prompts import (
    LANGUAGE_REPAIR_PROMPT_VERSION,
    LANGUAGE_REPAIR_SYSTEM_PROMPT,
    build_language_repair_adjudication_prompt,
    build_language_repair_generation_prompt,
    build_language_repair_review_prompt,
)
from .schemas import (
    CefrLanguageIssue,
    CefrLanguageRepairItem,
    CefrReviewBatch,
    CefrReviewRow,
    ReviewAction,
    UPOS,
)
from .routing import resolve_adjudication_model
from .semantic_generation import checkpointed_semantic_generate

FORBIDDEN_CEFR_UPOS = {UPOS.PROPN, UPOS.PUNCT, UPOS.SYM, UPOS.X}
MAX_LANGUAGE_REPAIR_RECORDS = 16
MAX_LANGUAGE_REPAIR_OUTPUT_TOKENS = 3_072


class LanguageRepairClient(Protocol):
    def generate(
        self,
        *,
        model: str,
        prompt: str,
        response_model: type[CefrReviewBatch],
        max_output_tokens: int,
    ) -> GenerationResult[CefrReviewBatch]: ...

    def parse_response(
        self,
        response_json: dict[str, object],
        response_model: type[CefrReviewBatch],
    ) -> tuple[CefrReviewBatch, object]: ...


def cefr_row_issues(
    row: CefrReviewRow,
    *,
    lang: str,
) -> list[CefrLanguageIssue]:
    profile = get_language(lang)
    issues: list[CefrLanguageIssue] = []
    if row.action is ReviewAction.DROP:
        issues.append(
            CefrLanguageIssue(
                code="cefr.drop",
                message="Final CEFR row must not use action=drop.",
            )
        )
    if row.upos in FORBIDDEN_CEFR_UPOS:
        issues.append(
            CefrLanguageIssue(
                code="cefr.forbidden_upos",
                message=f"UPOS {row.upos.value} is forbidden in final CEFR rows.",
            )
        )
    if any(character.isspace() for character in row.lemma):
        issues.append(
            CefrLanguageIssue(
                code="cefr.lemma_whitespace",
                message="Target lemma must be a single token without whitespace.",
            )
        )
    if unicodedata.normalize("NFC", row.lemma) != row.lemma:
        issues.append(
            CefrLanguageIssue(
                code="cefr.lemma_not_nfc",
                message="Target lemma must use NFC normalization.",
            )
        )
    if profile.code == "de" and row.upos is UPOS.NOUN and not row.lemma[0].isupper():
        issues.append(
            CefrLanguageIssue(
                code="german.noun_requires_uppercase",
                message="German common-noun citation lemmas must begin uppercase.",
            )
        )
    if (
        profile.code == "de"
        and row.upos not in {UPOS.NOUN, UPOS.PROPN}
        and row.lemma
        and row.lemma[0].isascii()
        and row.lemma[0].isupper()
    ):
        issues.append(
            CefrLanguageIssue(
                code="german.non_noun_capitalized",
                message=(
                    "German non-noun citation lemmas must not begin with an "
                    "uppercase ASCII letter."
                ),
            )
        )
    if profile.code == "en":
        # English lists: column-0 lemma is the citation form and must match the
        # English pivot (base form). Catches half-fixes like dreams/dream.
        if _nfc(row.lemma).casefold() != _nfc(row.english_lemma).casefold():
            issues.append(
                CefrLanguageIssue(
                    code="english.citation_mismatch",
                    message=(
                        "English citation lemma must match english_lemma "
                        "(base/dictionary form)."
                    ),
                )
            )
    elif row.lemma.casefold() == row.english_lemma.casefold():
        issues.append(
            CefrLanguageIssue(
                code="cefr.english_echo",
                message="Target lemma must not exactly echo the English lemma.",
            )
        )
    if (
        profile.code == "de"
        and row.upos is UPOS.VERB
        and not row.lemma.casefold().endswith(("en", "n"))
    ):
        issues.append(
            CefrLanguageIssue(
                code="german.verb_requires_infinitive",
                message="German verb citation lemmas must be infinitives ending in en or n.",
            )
        )
    if profile.code == "ar" and not has_arabic_script(row.lemma):
        issues.append(
            CefrLanguageIssue(
                code="arabic.script_required",
                message="Arabic lemmas must contain Arabic script.",
            )
        )
    if profile.code == "zh" and not has_han_script(row.lemma):
        issues.append(
            CefrLanguageIssue(
                code="chinese.script_required",
                message="Chinese lemmas must contain a Han character.",
            )
        )
    return issues


def german_row_issues(row: CefrReviewRow) -> list[CefrLanguageIssue]:
    return cefr_row_issues(row, lang="german")


def _nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def canonicalize_english_citation(row: CefrReviewRow) -> CefrReviewRow:
    """Force English lemma to the English pivot base form.

    Prefer ``english_lemma`` casing. If they already match case-insensitively,
    return the row unchanged (keep original lemma spelling).
    """
    lemma = _nfc(row.lemma).strip()
    english = _nfc(row.english_lemma).strip()
    if not lemma or not english:
        return row
    if lemma.casefold() == english.casefold():
        return row
    return row.model_copy(
        update={
            "lemma": english,
            "action": (
                ReviewAction.FIX
                if row.action is ReviewAction.KEEP
                else row.action
            ),
        }
    )


def canonicalize_english_review_rows(
    rows: Sequence[CefrReviewRow],
) -> list[CefrReviewRow]:
    return [canonicalize_english_citation(row) for row in rows]


def normalize_english_csv_file(path: Path) -> dict[str, int]:
    """Rewrite English CEFR CSV in place: lemma := english_lemma when they differ.

    Also drops exact duplicate lemma+POS after normalization (first wins).
    Returns counts: rows_in, rewritten, dropped_dupes, rows_out.
    """
    with path.open(encoding="utf-8", newline="") as handle:
        physical = list(csv.reader(handle))
    if not physical:
        raise ValueError(f"empty CSV: {path}")
    header, body = physical[0], physical[1:]
    rewritten = 0
    seen: set[tuple[str, str]] = set()
    out_rows: list[list[str]] = []
    dropped_dupes = 0
    for row in body:
        if len(row) != 4:
            raise ValueError(f"{path}: expected 4 columns, got {row!r}")
        lemma, english, chinese, upos = row
        lemma_n = _nfc(lemma).strip()
        english_n = _nfc(english).strip()
        if lemma_n.casefold() != english_n.casefold() and english_n:
            lemma_n = english_n
            rewritten += 1
        key = (lemma_n.casefold(), upos)
        if key in seen:
            dropped_dupes += 1
            continue
        seen.add(key)
        out_rows.append([lemma_n, english_n, _nfc(chinese).strip(), upos])
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(out_rows)
    temporary.replace(path)
    return {
        "rows_in": len(body),
        "rewritten": rewritten,
        "dropped_dupes": dropped_dupes,
        "rows_out": len(out_rows),
    }


def canonicalize_repaired_german_noun(row: CefrReviewRow) -> CefrReviewRow:
    """Uppercase a repaired NOUN only when noun case is its sole issue."""
    issues = german_row_issues(row)
    if row.upos is not UPOS.NOUN or [issue.code for issue in issues] != [
        "german.noun_requires_uppercase"
    ]:
        return row
    return row.model_copy(update={"lemma": row.lemma[0].upper() + row.lemma[1:]})


def repair_german_rows(
    rows: Sequence[CefrReviewRow],
    *,
    client: LanguageRepairClient,
    ledger: Ledger,
    lang: str,
    level: str,
    pass_number: int,
    single_model: str | None = None,
) -> list[CefrReviewRow]:
    if single_model is not None and single_model not in MODEL_IDS:
        raise ValueError(f"unsupported model: {single_model}")
    items = [
        CefrLanguageRepairItem(row=row, issues=german_row_issues(row)) for row in rows
    ]
    if any(not item.issues for item in items):
        raise ValueError("language repair received a row without language issues")
    packed = pack_records(
        items,
        prompt_overhead=LANGUAGE_REPAIR_SYSTEM_PROMPT,
        cap=INPUT_BATCH_TOKEN_CAP,
        max_records=MAX_LANGUAGE_REPAIR_RECORDS,
    )
    repaired: list[CefrReviewRow] = []
    for batch in packed:
        repaired.extend(
            _repair_language_batch(
                batch,
                client=client,
                ledger=ledger,
                lang=lang,
                level=level,
                pass_number=pass_number,
                single_model=single_model,
            ).rows
        )
    return repaired


def _repair_language_batch(
    items: Sequence[CefrLanguageRepairItem],
    *,
    client: LanguageRepairClient,
    ledger: Ledger,
    lang: str,
    level: str,
    pass_number: int,
    single_model: str | None,
) -> CefrReviewBatch:
    row_ids = [item.row.id for item in items]
    namespace = (
        f"language-repair:{LANGUAGE_REPAIR_PROMPT_VERSION}:{lang}:{level}:"
        f"pass:{pass_number}:{row_ids[0]}..{row_ids[-1]}"
    )
    generation_prompt = build_language_repair_generation_prompt(
        items,
        lang=lang,
        level=level,
        pass_number=pass_number,
    )
    generated = _checkpointed_language_repair(
        client=client,
        ledger=ledger,
        model=single_model or MODEL_31B,
        batch_id=f"{namespace}:generation",
        prompt=generation_prompt,
        row_ids=row_ids,
    )
    if single_model is not None:
        return generated
    review_prompt = build_language_repair_review_prompt(
        items,
        generated,
        lang=lang,
        level=level,
        pass_number=pass_number,
    )
    reviewed = _checkpointed_language_repair(
        client=client,
        ledger=ledger,
        model=MODEL_26B,
        batch_id=f"{namespace}:review",
        prompt=review_prompt,
        row_ids=row_ids,
    )
    if generated.model_dump(mode="json") == reviewed.model_dump(mode="json"):
        return generated
    adjudication_prompt = build_language_repair_adjudication_prompt(
        items,
        generated,
        reviewed,
        lang=lang,
        level=level,
        pass_number=pass_number,
    )
    return _checkpointed_language_repair(
        client=client,
        ledger=ledger,
        model=resolve_adjudication_model(
            client,
            prompt=adjudication_prompt,
            exclude=(MODEL_31B, MODEL_26B),
        ),
        batch_id=f"{namespace}:adjudication",
        prompt=adjudication_prompt,
        row_ids=row_ids,
    )


def _checkpointed_language_repair(
    *,
    client: LanguageRepairClient,
    ledger: Ledger,
    model: str,
    batch_id: str,
    prompt: str,
    row_ids: Sequence[str],
) -> CefrReviewBatch:
    if TiktokenEstimator().count(prompt) > INPUT_BATCH_TOKEN_CAP:
        raise ValueError("language repair prompt exceeds input token cap")
    return checkpointed_semantic_generate(
        client=client,
        ledger=ledger,
        model=model,
        batch_id=batch_id,
        prompt=prompt,
        response_model=CefrReviewBatch,
        max_output_tokens=MAX_LANGUAGE_REPAIR_OUTPUT_TOKENS,
        validate=lambda batch: _validate_language_repair_identity(row_ids, batch),
        expected_identity={"row_ids": list(row_ids)},
    )


def _validate_language_repair_identity(
    row_ids: Sequence[str],
    batch: CefrReviewBatch,
) -> CefrReviewBatch:
    if [row.id for row in batch.rows] != list(row_ids):
        raise ValueError("language repair IDs/order/cardinality differ from input")
    return batch
