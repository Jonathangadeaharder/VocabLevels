from __future__ import annotations

import csv
import unicodedata
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from vocab_schema import TARGETS

from .cefr_refill import (
    RefillClient,
    ReviewRequiredError,
    complete_cefr_rows,
    dedupe_review_rows,
    load_english_refill_concepts,
    load_gap_reject_keys,
    load_gap_rejected_english_keys,
    load_other_level_collision_keys,
)
from .client import GenerationResult
from .config import INPUT_BATCH_TOKEN_CAP, MODEL_26B, MODEL_31B, MODEL_IDS
from .ledger import Ledger
from .language_repair import (
    LanguageRepairClient,
    canonicalize_repaired_german_noun,
    cefr_row_issues,
    german_row_issues,
    repair_german_rows,
)
from .languages import get_language
from .packing import pack_records
from .prompts import SYSTEM_PROMPT, build_adjudication_prompt, build_cefr_prompt
from .schemas import (
    CefrInputRow,
    CefrRefillConcept,
    CefrReviewBatch,
    CefrReviewRow,
    ReviewAction,
    UPOS,
)
from .routing import resolve_adjudication_model
from .semantic_generation import checkpointed_semantic_generate
from .validated import ValidatedStore, validated_store_path

MAX_OUTPUT_TOKENS = 3_072
MAX_RECORDS_PER_BATCH = 36


class CefrClient(Protocol):
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


@dataclass(frozen=True)
class CefrDocument:
    path: Path
    header: tuple[str, str, str, str]
    rows: list[CefrInputRow]


def read_cefr_csv(path: Path, *, lang: str, level: str) -> CefrDocument:
    with path.open(encoding="utf-8", newline="") as handle:
        physical_rows = list(csv.reader(handle))
    if not physical_rows:
        raise ValueError(f"empty CSV: {path}")
    header = physical_rows[0]
    expected = [f"{lang.title()}_Lemma", "English_Lemma", "Chinese_Lemma", "POS"]
    if header != expected:
        raise ValueError(f"unexpected header in {path}: {header!r}")
    rows: list[CefrInputRow] = []
    for row_number, row in enumerate(physical_rows[1:], start=1):
        if len(row) != 4:
            raise ValueError(f"{path}:{row_number + 1}: expected 4 columns")
        rows.append(
            CefrInputRow.model_validate(
                {
                    "id": f"{lang}:{level}:{row_number}",
                    "lemma": row[0],
                    "english_lemma": row[1],
                    "chinese_lemma": row[2] or None,
                    "upos": row[3],
                }
            )
        )
    return CefrDocument(
        path=path,
        header=(header[0], header[1], header[2], header[3]),
        rows=rows,
    )


def validate_review_batch(
    inputs: Sequence[CefrInputRow],
    reviews: CefrReviewBatch,
) -> CefrReviewBatch:
    input_ids = [row.id for row in inputs]
    review_ids = [row.id for row in reviews.rows]
    if review_ids != input_ids:
        raise ValueError("review IDs/order/cardinality differ from input IDs")
    return reviews


def normalize_review(row: CefrReviewRow) -> tuple[str, str, str, str, str, str]:
    def normalize(value: str) -> str:
        return " ".join(unicodedata.normalize("NFC", value).split())

    return (
        row.id,
        normalize(row.lemma),
        normalize(row.english_lemma).casefold(),
        normalize(row.chinese_lemma),
        row.upos.value,
        row.action.value,
    )


def gap_proposed_path(document: CefrDocument) -> Path:
    return document.path.parent / f"{document.path.stem}.gap.proposed.csv"


def input_row_as_review(row: CefrInputRow) -> CefrReviewRow:
    if not row.chinese_lemma:
        raise ValueError(f"{row.id}: Chinese_Lemma is required")
    return CefrReviewRow(
        id=row.id,
        lemma=row.lemma,
        english_lemma=row.english_lemma,
        chinese_lemma=row.chinese_lemma,
        upos=row.upos,
        action=ReviewAction.KEEP,
    )


def split_validated_rows(
    store: ValidatedStore,
    rows: Sequence[CefrInputRow],
    *,
    lang: str,
    level: str,
) -> tuple[list[CefrReviewRow], list[CefrInputRow]]:
    frozen: list[CefrReviewRow] = []
    pending: list[CefrInputRow] = []
    for row in rows:
        if store.contains(
            lang,
            level,
            row.lemma,
            row.english_lemma,
            row.chinese_lemma or "",
            row.upos.value,
        ):
            frozen.append(input_row_as_review(row))
        else:
            pending.append(row)
    return frozen, pending


def committed_rows_as_reviews(document: CefrDocument) -> list[CefrReviewRow]:
    reviews: list[CefrReviewRow] = []
    for row in document.rows:
        if not row.chinese_lemma:
            raise ValueError(f"{document.path}:{row.id}: Chinese_Lemma is required")
        reviews.append(
            CefrReviewRow(
                id=row.id,
                lemma=row.lemma,
                english_lemma=row.english_lemma,
                chinese_lemma=row.chinese_lemma,
                upos=row.upos,
                action=ReviewAction.KEEP,
            )
        )
    return reviews


def write_gap_proposed_csv(
    document: CefrDocument,
    rows: Sequence[CefrReviewRow],
) -> Path:
    if any(row.action is ReviewAction.DROP for row in rows):
        raise ValueError("gap refill rows must not contain drop actions")
    output = gap_proposed_path(document)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(document.header)
        for review in rows:
            writer.writerow(
                [
                    review.lemma,
                    review.english_lemma,
                    review.chinese_lemma,
                    review.upos.value,
                ]
            )
    temporary.replace(output)
    return output


def run_cefr_gap_refill(
    *,
    root: Path,
    lang: str,
    level: str,
    client: CefrClient,
    ledger: Ledger,
    single_model: str | None = None,
    reject_decisions_dir: Path | None = None,
) -> Path:
    if single_model is not None and single_model not in MODEL_IDS:
        raise ValueError(f"unsupported model: {single_model}")
    profile = get_language(lang)
    lang = profile.directory
    validated = ValidatedStore(validated_store_path(root))
    try:
        validated.seed_from_csv(root, lang=lang, level=level)
    finally:
        validated.close()
    document = read_cefr_csv(root / lang / f"{level}.csv", lang=lang, level=level)
    accepted = committed_rows_as_reviews(document)
    target = TARGETS[level]
    if len(accepted) >= target:
        raise ReviewRequiredError(
            f"{lang} {level}: committed row count {len(accepted)} "
            f"already meets target {target}"
        )
    collision_keys = load_other_level_collision_keys(
        root,
        lang=lang,
        level=level,
    )
    skipped_english_keys: set[tuple[str, UPOS]] | None = None
    if reject_decisions_dir is not None:
        collision_keys |= load_gap_reject_keys(reject_decisions_dir)
        skipped_english_keys = load_gap_rejected_english_keys(reject_decisions_dir)
    concepts = load_english_refill_concepts(root, level=level)
    final = complete_cefr_rows(
        accepted,
        concepts=concepts,
        collision_keys=collision_keys,
        target=target,
        lang=lang,
        level=level,
        client=cast(RefillClient, client),
        ledger=ledger,
        single_model=single_model,
        skipped_english_keys=skipped_english_keys,
    )
    gap_rows = final[len(accepted) :]
    expected_gap = target - len(accepted)
    if len(gap_rows) != expected_gap:
        raise ReviewRequiredError(
            f"{lang} {level}: expected {expected_gap} gap rows, got {len(gap_rows)}"
        )
    return write_gap_proposed_csv(document, gap_rows)


def write_reviewed_csv(
    document: CefrDocument,
    reviews: CefrReviewBatch | Sequence[CefrReviewRow],
    *,
    apply: bool,
) -> Path:
    rows = reviews.rows if isinstance(reviews, CefrReviewBatch) else reviews
    if any(row.action is ReviewAction.DROP for row in rows):
        raise ValueError("final reviewed rows must not contain drop actions")
    output = document.path if apply else document.path.with_suffix(".proposed.csv")
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(document.header)
        for review in rows:
            writer.writerow(
                [
                    review.lemma,
                    review.english_lemma,
                    review.chinese_lemma,
                    review.upos.value,
                ]
            )
    temporary.replace(output)
    return output


def run_cefr(
    *,
    root: Path,
    lang: str,
    level: str,
    client: CefrClient,
    ledger: Ledger,
    limit: int | None = None,
    apply: bool = False,
    single_model: str | None = None,
    refill_to_target: bool | None = None,
) -> Path:
    if single_model is not None and single_model not in MODEL_IDS:
        raise ValueError(f"unsupported model: {single_model}")
    profile = get_language(lang)
    lang = profile.directory
    document = read_cefr_csv(root / lang / f"{level}.csv", lang=lang, level=level)
    selected = document.rows if limit is None else document.rows[:limit]
    if not selected:
        raise ValueError("no CEFR rows selected")
    validated = ValidatedStore(validated_store_path(root))
    try:
        frozen, pending = split_validated_rows(
            validated,
            selected,
            lang=lang,
            level=level,
        )
        accepted: list[CefrReviewRow] = list(frozen)
        if pending:
            packed = pack_records(
                pending,
                prompt_overhead=(
                    SYSTEM_PROMPT
                    if profile.code == "de"
                    else build_cefr_prompt((), lang=lang)
                ),
                cap=INPUT_BATCH_TOKEN_CAP,
                max_records=MAX_RECORDS_PER_BATCH,
            )
            models = (single_model,) if single_model else (MODEL_31B, MODEL_26B)
            for batch_rows in packed:
                batch_id = f"{batch_rows[0].id}..{batch_rows[-1].id}"
                prompt = build_cefr_prompt(batch_rows, lang=lang)
                with ThreadPoolExecutor(max_workers=len(models)) as executor:
                    futures = [
                        executor.submit(
                            _checkpointed_generate,
                            client=client,
                            ledger=ledger,
                            model=model,
                            batch_id=batch_id,
                            prompt=prompt,
                            expected_rows=batch_rows,
                        )
                        for model in models
                    ]
                    independent = [future.result() for future in futures]
                for review in independent:
                    validate_review_batch(batch_rows, review)
                chosen = independent[0]
                if len(independent) == 2 and [
                    normalize_review(row) for row in independent[0].rows
                ] != [normalize_review(row) for row in independent[1].rows]:
                    adjudication_prompt = build_adjudication_prompt(
                        batch_rows,
                        independent[0].rows,
                        independent[1].rows,
                        lang=lang,
                    )
                    chosen = _checkpointed_generate(
                        client=client,
                        ledger=ledger,
                        model=resolve_adjudication_model(client),
                        batch_id=f"{batch_id}:adjudication",
                        prompt=adjudication_prompt,
                        expected_rows=batch_rows,
                    )
                    validate_review_batch(batch_rows, chosen)
                accepted.extend(chosen.rows)
        target = TARGETS[level] if limit is None else len(selected)
        collision_keys = load_other_level_collision_keys(
            root,
            lang=lang,
            level=level,
        )
        gate_clean = (
            accepted
            if profile.code == "de"
            else [row for row in accepted if not cefr_row_issues(row, lang=lang)]
        )
        unique = dedupe_review_rows(gate_clean, collision_keys)
        if len(unique) > target:
            raise ReviewRequiredError(
                f"{lang} {level}: {len(unique)} unique review rows exceed target {target}; "
                "manual review required"
            )
        exact_target = (
            profile.code == "de" if refill_to_target is None else refill_to_target
        )
        concepts = []
        if exact_target and len(unique) < target:
            concepts = load_english_refill_concepts(root, level=level)
        if profile.code == "de" and any(
            german_row_issues(row)
            and not validated.contains(
                lang,
                level,
                row.lemma,
                row.english_lemma,
                row.chinese_lemma,
                row.upos.value,
            )
            for row in unique
        ):
            final_rows = _repair_and_refill_german_rows(
                unique,
                concepts=concepts,
                root=root,
                collision_keys=collision_keys,
                target=target,
                lang=lang,
                level=level,
                client=client,
                ledger=ledger,
                single_model=single_model,
                refill_to_target=exact_target,
                validated=validated,
            )
        elif exact_target and len(unique) < target:
            final_rows = complete_cefr_rows(
                unique,
                concepts=concepts,
                collision_keys=collision_keys,
                target=target,
                lang=lang,
                level=level,
                client=cast(RefillClient, client),
                ledger=ledger,
                single_model=single_model,
            )
        else:
            final_rows = unique
        return write_reviewed_csv(
            document,
            final_rows,
            apply=apply,
        )
    finally:
        validated.close()


def _repair_and_refill_german_rows(
    rows: Sequence[CefrReviewRow],
    *,
    concepts: Sequence[CefrRefillConcept],
    root: Path,
    collision_keys: set[tuple[str, UPOS]],
    target: int,
    lang: str,
    level: str,
    client: CefrClient,
    ledger: Ledger,
    single_model: str | None,
    refill_to_target: bool,
    validated: ValidatedStore | None = None,
) -> list[CefrReviewRow]:
    final = list(rows)
    if lang != "german":
        return final
    refill_concepts = list(concepts)
    for pass_number in range(1, 3):
        failing = [
            row
            for row in final
            if german_row_issues(row)
            and (
                validated is None
                or not validated.contains(
                    lang,
                    level,
                    row.lemma,
                    row.english_lemma,
                    row.chinese_lemma,
                    row.upos.value,
                )
            )
        ]
        if not failing:
            return final
        repaired = [
            canonicalize_repaired_german_noun(row)
            for row in repair_german_rows(
                failing,
                client=cast(LanguageRepairClient, client),
                ledger=ledger,
                lang=lang,
                level=level,
                pass_number=pass_number,
                single_model=single_model,
            )
        ]
        replacements = {row.id: row for row in repaired}
        repaired_in_order = [replacements.get(row.id, row) for row in final]
        valid = [row for row in repaired_in_order if not german_row_issues(row)]
        unique = dedupe_review_rows(valid, collision_keys)
        if not refill_to_target:
            final = unique
            continue
        if len(unique) < target and not refill_concepts:
            refill_concepts = load_english_refill_concepts(root, level=level)
        final = complete_cefr_rows(
            unique,
            concepts=refill_concepts,
            collision_keys=collision_keys,
            target=target,
            lang=lang,
            level=level,
            client=cast(RefillClient, client),
            ledger=ledger,
            single_model=single_model,
        )
    remaining = [
        f"{row.id}:{issue.code}"
        for row in final
        for issue in german_row_issues(row)
        if validated is None
        or not validated.contains(
            lang,
            level,
            row.lemma,
            row.english_lemma,
            row.chinese_lemma,
            row.upos.value,
        )
    ]
    if remaining:
        raise ReviewRequiredError(
            f"{lang} {level}: language issues remain after 2 repair/refill passes: "
            + ", ".join(remaining)
        )
    return final


def _checkpointed_generate(
    *,
    client: CefrClient,
    ledger: Ledger,
    model: str,
    batch_id: str,
    prompt: str,
    expected_rows: Sequence[CefrInputRow],
) -> CefrReviewBatch:
    return checkpointed_semantic_generate(
        client=client,
        ledger=ledger,
        model=model,
        batch_id=batch_id,
        prompt=prompt,
        response_model=CefrReviewBatch,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        validate=lambda reviews: validate_review_batch(expected_rows, reviews),
        expected_identity={"row_ids": [row.id for row in expected_rows]},
    )
