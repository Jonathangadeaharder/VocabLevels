from __future__ import annotations

import csv
import unicodedata
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol, cast

from check_quality import SPECIAL_CHARS
from vocab_schema import LEVELS

from .client import GenerationResult
from .config import INPUT_BATCH_TOKEN_CAP, MODEL_26B, MODEL_31B, MODEL_IDS
from .ledger import Ledger
from .language_repair import (
    canonicalize_repaired_german_noun,
    cefr_row_issues,
    german_row_issues,
)
from .languages import get_language
from .packing import TiktokenEstimator, pack_records
from .prompts import (
    NOVEL_PROMPT_VERSION,
    REFILL_SYSTEM_PROMPT,
    build_novel_adjudication_prompt,
    build_novel_generation_prompt,
    build_novel_review_prompt,
    build_refill_adjudication_prompt,
    build_refill_generation_prompt,
    build_refill_review_prompt,
    novel_initial_hint,
)
from .schemas import (
    CefrNovelBatch,
    CefrRefillBatch,
    CefrRefillConcept,
    CefrRefillRow,
    CefrReviewRow,
    ReviewAction,
    UPOS,
)
from .routing import resolve_adjudication_model
from .semantic_generation import checkpointed_semantic_generate

MAX_REFILL_RECORDS = 24
MAX_REFILL_OUTPUT_TOKENS = 2_048
MAX_NOVEL_RECORDS = 10
MAX_NOVEL_ATTEMPTS = 30
MAX_NOVEL_OUTPUT_TOKENS = 2_048


class ReviewRequiredError(ValueError):
    pass


class RefillClient(Protocol):
    def generate(
        self,
        *,
        model: str,
        prompt: str,
        response_model: type[CefrRefillBatch],
        max_output_tokens: int,
    ) -> GenerationResult[CefrRefillBatch]: ...

    def parse_response(
        self,
        response_json: dict[str, object],
        response_model: type[CefrRefillBatch],
    ) -> tuple[CefrRefillBatch, object]: ...


class NovelClient(Protocol):
    def generate(
        self,
        *,
        model: str,
        prompt: str,
        response_model: type[CefrNovelBatch],
        max_output_tokens: int,
    ) -> GenerationResult[CefrNovelBatch]: ...

    def parse_response(
        self,
        response_json: dict[str, object],
        response_model: type[CefrNovelBatch],
    ) -> tuple[CefrNovelBatch, object]: ...


def normalized_key(lemma: str, upos: UPOS) -> tuple[str, UPOS]:
    return unicodedata.normalize("NFC", lemma.strip()).casefold(), upos


def load_other_level_collision_keys(
    root: Path,
    *,
    lang: str,
    level: str,
) -> set[tuple[str, UPOS]]:
    keys: set[tuple[str, UPOS]] = set()
    for other_level in LEVELS:
        if other_level == level:
            continue
        path = root / lang / f"{other_level}.csv"
        if not path.exists():
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
        for row_number, row in enumerate(rows[1:], start=2):
            if len(row) != 4:
                raise ValueError(f"{path}:{row_number}: expected 4 columns")
            keys.add(normalized_key(row[0], UPOS(row[3].strip())))
    return keys


def load_english_refill_concepts(
    root: Path,
    *,
    level: str,
) -> list[CefrRefillConcept]:
    path = root / "english" / f"{level}.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    expected_header = [
        "English_Lemma",
        "English_Lemma",
        "Chinese_Lemma",
        "POS",
    ]
    if not rows or rows[0] != expected_header:
        raise ValueError(f"unexpected English pivot header in {path}")
    concepts: list[CefrRefillConcept] = []
    for row_number, row in enumerate(rows[1:], start=1):
        if len(row) != 4:
            raise ValueError(f"{path}:{row_number + 1}: expected 4 columns")
        concepts.append(
            CefrRefillConcept(
                id=f"english:{level}:{row_number}",
                english_lemma=row[0],
                chinese_lemma=row[2] or None,
                upos=UPOS(row[3]),
            )
        )
    return concepts


def load_gap_reject_keys(decisions_dir: Path) -> set[tuple[str, UPOS]]:
    from .manual_review import read_decisions

    keys: set[tuple[str, UPOS]] = set()
    for decision in read_decisions(decisions_dir):
        keys.add(
            normalized_key(decision.expected.lemma, decision.expected.upos),
        )
    return keys


def load_gap_rejected_english_keys(decisions_dir: Path) -> set[tuple[str, UPOS]]:
    from .manual_review import read_decisions

    keys: set[tuple[str, UPOS]] = set()
    for decision in read_decisions(decisions_dir):
        keys.add(
            normalized_key(
                decision.expected.english_lemma,
                decision.expected.upos,
            ),
        )
    return keys


def complete_cefr_rows(
    accepted: Sequence[CefrReviewRow],
    *,
    concepts: Sequence[CefrRefillConcept],
    collision_keys: set[tuple[str, UPOS]],
    target: int,
    lang: str,
    level: str,
    client: RefillClient,
    ledger: Ledger,
    single_model: str | None = None,
    skipped_english_keys: set[tuple[str, UPOS]] | None = None,
) -> list[CefrReviewRow]:
    if target <= 0:
        raise ValueError("target must be positive")
    if single_model is not None and single_model not in MODEL_IDS:
        raise ValueError(f"unsupported model: {single_model}")
    final = dedupe_review_rows(accepted, collision_keys)
    if len(final) > target:
        raise ReviewRequiredError(
            f"{lang} {level}: {len(final)} unique review rows exceed target {target}; "
            "manual review required"
        )
    if len(final) == target:
        return _assert_language_clean(final, lang=lang, level=level)

    represented_concepts = {
        normalized_key(row.english_lemma, row.upos) for row in final
    }
    if skipped_english_keys:
        represented_concepts |= skipped_english_keys
    unrepresented_concepts = [
        concept
        for concept in concepts
        if normalized_key(concept.english_lemma, concept.upos)
        not in represented_concepts
    ]
    represented_retry_concepts = [
        concept
        for concept in concepts
        if normalized_key(concept.english_lemma, concept.upos) in represented_concepts
    ]
    ordered_concepts = unrepresented_concepts
    used_keys = {normalized_key(row.lemma, row.upos) for row in final}
    profile = get_language(lang)
    if profile.code == "en":
        for concept in ordered_concepts:
            if len(final) >= target:
                break
            if concept.chinese_lemma is None:
                continue
            candidate = CefrReviewRow(
                id=concept.id,
                lemma=concept.english_lemma,
                english_lemma=concept.english_lemma,
                chinese_lemma=concept.chinese_lemma,
                upos=concept.upos,
                action=ReviewAction.KEEP,
            )
            key = normalized_key(candidate.lemma, candidate.upos)
            if (
                key in collision_keys
                or key in used_keys
                or cefr_row_issues(candidate, lang=lang)
            ):
                continue
            final.append(candidate)
            used_keys.add(key)
        ordered_concepts = []
    offset = 0
    accepted_in_unrepresented_pass = 0
    while len(final) < target and offset < len(ordered_concepts):
        remaining = ordered_concepts[offset:]
        deficit = target - len(final)
        batches = pack_records(
            remaining,
            prompt_overhead=(
                REFILL_SYSTEM_PROMPT
                if profile.code == "de"
                else build_refill_generation_prompt((), lang=lang, level=level)
            ),
            cap=INPUT_BATCH_TOKEN_CAP,
            max_records=min(MAX_REFILL_RECORDS, deficit),
        )
        batch = batches[0]
        offset += len(batch)
        reviewed = _run_refill_batch(
            batch,
            lang=lang,
            level=level,
            client=client,
            ledger=ledger,
            single_model=single_model,
        )
        for concept, candidate in zip(batch, reviewed.rows, strict=True):
            accepted_candidate = _accepted_refill_candidate(
                candidate,
                concept,
                lang=lang,
            )
            if accepted_candidate is None:
                continue
            key = normalized_key(accepted_candidate.lemma, accepted_candidate.upos)
            if key in collision_keys or key in used_keys:
                continue
            final.append(accepted_candidate)
            used_keys.add(key)
            accepted_in_unrepresented_pass += 1

    if len(final) < target and accepted_in_unrepresented_pass == 0:
        final = _complete_novel_rows(
            final,
            collision_keys=collision_keys,
            target=target,
            lang=lang,
            level=level,
            client=cast(NovelClient, client),
            ledger=ledger,
            single_model=single_model,
        )
        return _assert_language_clean(final, lang=lang, level=level)

    offset = 0
    ordered_concepts = represented_retry_concepts
    while len(final) < target and offset < len(ordered_concepts):
        remaining = ordered_concepts[offset:]
        deficit = target - len(final)
        batches = pack_records(
            remaining,
            prompt_overhead=(
                REFILL_SYSTEM_PROMPT
                if profile.code == "de"
                else build_refill_generation_prompt((), lang=lang, level=level)
            ),
            cap=INPUT_BATCH_TOKEN_CAP,
            max_records=min(MAX_REFILL_RECORDS, deficit),
        )
        batch = batches[0]
        offset += len(batch)
        reviewed = _run_refill_batch(
            batch,
            lang=lang,
            level=level,
            client=client,
            ledger=ledger,
            single_model=single_model,
        )
        for concept, candidate in zip(batch, reviewed.rows, strict=True):
            accepted_candidate = _accepted_refill_candidate(
                candidate,
                concept,
                lang=lang,
            )
            if accepted_candidate is None:
                continue
            key = normalized_key(accepted_candidate.lemma, accepted_candidate.upos)
            if key in collision_keys or key in used_keys:
                continue
            final.append(accepted_candidate)
            used_keys.add(key)

    if len(final) < target:
        final = _complete_novel_rows(
            final,
            collision_keys=collision_keys,
            target=target,
            lang=lang,
            level=level,
            client=cast(NovelClient, client),
            ledger=ledger,
            single_model=single_model,
        )
    return _assert_language_clean(final, lang=lang, level=level)


def dedupe_review_rows(
    rows: Sequence[CefrReviewRow],
    collision_keys: set[tuple[str, UPOS]],
) -> list[CefrReviewRow]:
    result: list[CefrReviewRow] = []
    seen: set[tuple[str, UPOS]] = set()
    for row in rows:
        if row.action is ReviewAction.DROP:
            continue
        key = normalized_key(row.lemma, row.upos)
        if key in seen or key in collision_keys:
            continue
        result.append(row)
        seen.add(key)
    return result


def _accepted_refill_candidate(
    row: CefrRefillRow,
    concept: CefrRefillConcept,
    *,
    lang: str,
) -> CefrReviewRow | None:
    if row.action is ReviewAction.DROP:
        return None
    if concept.upos in {UPOS.PROPN, UPOS.PUNCT, UPOS.SYM, UPOS.X}:
        return None
    if not row.lemma or any(character.isspace() for character in row.lemma):
        return None
    if unicodedata.normalize("NFC", row.lemma) != row.lemma:
        return None
    if not row.chinese_lemma:
        return None
    candidate = CefrReviewRow(
        id=row.id,
        lemma=row.lemma,
        english_lemma=concept.english_lemma,
        chinese_lemma=row.chinese_lemma,
        upos=concept.upos,
        action=row.action,
    )
    if cefr_row_issues(candidate, lang=lang):
        return None
    return candidate


def _complete_novel_rows(
    final: list[CefrReviewRow],
    *,
    collision_keys: set[tuple[str, UPOS]],
    target: int,
    lang: str,
    level: str,
    client: NovelClient,
    ledger: Ledger,
    single_model: str | None,
) -> list[CefrReviewRow]:
    missing = target - len(final)
    pending_slots = list(range(1, missing + 1))
    used_keys = {normalized_key(row.lemma, row.upos) for row in final}
    represented_english = {normalized_key(row.english_lemma, row.upos) for row in final}
    rejected_exclusions: list[str] = []
    for round_number in range(1, MAX_NOVEL_ATTEMPTS + 1):
        rejected_slots: list[int] = []
        for start in range(0, len(pending_slots), MAX_NOVEL_RECORDS):
            slots = pending_slots[start : start + MAX_NOVEL_RECORDS]
            slot_ids = [
                f"novel:{lang}:{level}:slot:{slot}:round:{round_number}"
                for slot in slots
            ]
            exclusions = _ordered_novel_exclusions(rejected_exclusions, final)
            reviewed = _run_novel_batch(
                slot_ids,
                exclusions=exclusions,
                lang=lang,
                level=level,
                client=client,
                ledger=ledger,
                single_model=single_model,
            )
            for slot, candidate in zip(slots, reviewed.rows, strict=True):
                accepted_candidate = CefrReviewRow.model_validate(
                    candidate.model_dump(mode="json")
                )
                if get_language(lang).code == "de":
                    accepted_candidate = canonicalize_repaired_german_noun(
                        accepted_candidate
                    )
                    if german_row_issues(accepted_candidate):
                        target_key = normalized_key(
                            accepted_candidate.lemma,
                            accepted_candidate.upos,
                        )
                        _remember_rejected_key(rejected_exclusions, target_key)
                        rejected_slots.append(slot)
                        continue
                if cefr_row_issues(accepted_candidate, lang=lang):
                    target_key = normalized_key(
                        accepted_candidate.lemma,
                        accepted_candidate.upos,
                    )
                    _remember_rejected_key(rejected_exclusions, target_key)
                    rejected_slots.append(slot)
                    continue
                target_key = normalized_key(
                    accepted_candidate.lemma,
                    accepted_candidate.upos,
                )
                english_key = normalized_key(
                    accepted_candidate.english_lemma,
                    accepted_candidate.upos,
                )
                if not _novel_candidate_passes(
                    accepted_candidate,
                    required_initial=novel_initial_hint(candidate.id),
                ):
                    _remember_rejected_key(rejected_exclusions, target_key)
                    rejected_slots.append(slot)
                    continue
                if target_key in collision_keys or target_key in used_keys:
                    _remember_rejected_key(rejected_exclusions, target_key)
                    rejected_slots.append(slot)
                    continue
                if english_key in represented_english:
                    _remember_rejected_key(rejected_exclusions, target_key)
                    rejected_slots.append(slot)
                    continue
                final.append(accepted_candidate)
                used_keys.add(target_key)
                represented_english.add(english_key)
        pending_slots = rejected_slots
        if not pending_slots:
            return final
    raise ReviewRequiredError(
        f"{lang} {level}: novel refill exhausted {MAX_NOVEL_ATTEMPTS} attempts "
        f"at {len(final)} of {target}"
    )


def _ordered_novel_exclusions(
    rejected: Sequence[str],
    final: Sequence[CefrReviewRow],
) -> list[str]:
    ordered = list(rejected)
    seen = set(ordered)
    for row in final:
        key = f"{normalized_key(row.lemma, row.upos)[0]}|{row.upos.value}"
        if key not in seen:
            ordered.append(key)
            seen.add(key)
    return ordered


def _remember_rejected_key(
    rejected: list[str],
    key: tuple[str, UPOS],
) -> None:
    if not key[0]:
        return
    exclusion = f"{key[0]}|{key[1].value}"
    if exclusion in rejected:
        rejected.remove(exclusion)
    rejected.insert(0, exclusion)


def _novel_candidate_passes(
    row: CefrReviewRow,
    *,
    required_initial: str | None,
) -> bool:
    if row.action is ReviewAction.DROP:
        return False
    if row.upos in {UPOS.PROPN, UPOS.PUNCT, UPOS.SYM, UPOS.X}:
        return False
    if any(character.isspace() for character in row.lemma):
        return False
    if any(character.isdigit() for character in row.lemma):
        return False
    if SPECIAL_CHARS.search(row.lemma):
        return False
    if required_initial is not None and not row.lemma.casefold().startswith(
        required_initial
    ):
        return False
    return all(
        unicodedata.normalize("NFC", value) == value
        for value in (
            row.lemma,
            row.english_lemma,
            row.chinese_lemma,
        )
    )


def _assert_language_clean(
    rows: list[CefrReviewRow],
    *,
    lang: str,
    level: str,
) -> list[CefrReviewRow]:
    issues = [
        f"{row.id}:{issue.code}"
        for row in rows
        for issue in cefr_row_issues(row, lang=lang)
    ]
    if issues:
        raise ReviewRequiredError(
            f"{lang} {level}: completion contains language issues: " + ", ".join(issues)
        )
    return rows


def _run_novel_batch(
    slot_ids: Sequence[str],
    *,
    exclusions: Sequence[str],
    lang: str,
    level: str,
    client: NovelClient,
    ledger: Ledger,
    single_model: str | None,
) -> CefrNovelBatch:
    identity = {"row_ids": list(slot_ids)}
    profile = get_language(lang)
    prompt_version = (
        NOVEL_PROMPT_VERSION
        if profile.code == "de"
        else f"cefr-novel-{profile.code}-v3"
    )
    namespace = f"novel:{prompt_version}:{lang}:{level}:{slot_ids[0]}..{slot_ids[-1]}"
    generation_prompt = _bounded_novel_prompt(
        lambda included: build_novel_generation_prompt(
            slot_ids,
            lang=lang,
            level=level,
            exclusions=included,
        ),
        exclusions,
    )
    generated = _checkpointed_novel(
        client=client,
        ledger=ledger,
        model=single_model or MODEL_31B,
        batch_id=f"{namespace}:generation",
        prompt=generation_prompt,
        slot_ids=slot_ids,
        identity=identity,
    )
    if single_model is not None:
        return generated
    review_prompt = _bounded_novel_prompt(
        lambda included: build_novel_review_prompt(
            slot_ids,
            generated,
            lang=lang,
            level=level,
            exclusions=included,
        ),
        exclusions,
    )
    reviewed = _checkpointed_novel(
        client=client,
        ledger=ledger,
        model=MODEL_26B,
        batch_id=f"{namespace}:review",
        prompt=review_prompt,
        slot_ids=slot_ids,
        identity=identity,
    )
    if generated.model_dump(mode="json") == reviewed.model_dump(mode="json"):
        return generated
    adjudication_prompt = _bounded_novel_prompt(
        lambda included: build_novel_adjudication_prompt(
            slot_ids,
            generated,
            reviewed,
            lang=lang,
            level=level,
            exclusions=included,
        ),
        exclusions,
    )
    return _checkpointed_novel(
        client=client,
        ledger=ledger,
        model=resolve_adjudication_model(client),
        batch_id=f"{namespace}:adjudication",
        prompt=adjudication_prompt,
        slot_ids=slot_ids,
        identity=identity,
    )


def _bounded_novel_prompt(
    build: Callable[[Sequence[str]], str],
    exclusions: Sequence[str],
) -> str:
    estimator = TiktokenEstimator()
    empty_prompt = build(())
    if estimator.count(empty_prompt) > INPUT_BATCH_TOKEN_CAP:
        raise ReviewRequiredError("novel refill prompt exceeds input token cap")
    lower = 0
    upper = len(exclusions)
    while lower < upper:
        candidate = (lower + upper + 1) // 2
        prompt = build(exclusions[:candidate])
        if estimator.count(prompt) <= INPUT_BATCH_TOKEN_CAP:
            lower = candidate
        else:
            upper = candidate - 1
    return build(exclusions[:lower])


def _checkpointed_novel(
    *,
    client: NovelClient,
    ledger: Ledger,
    model: str,
    batch_id: str,
    prompt: str,
    slot_ids: Sequence[str],
    identity: object,
) -> CefrNovelBatch:
    return checkpointed_semantic_generate(
        client=client,
        ledger=ledger,
        model=model,
        batch_id=batch_id,
        prompt=prompt,
        response_model=CefrNovelBatch,
        max_output_tokens=MAX_NOVEL_OUTPUT_TOKENS,
        validate=lambda batch: _validate_novel_identity(slot_ids, batch),
        expected_identity=identity,
    )


def _validate_novel_identity(
    slot_ids: Sequence[str],
    batch: CefrNovelBatch,
) -> CefrNovelBatch:
    if [row.id for row in batch.rows] != list(slot_ids):
        raise ValueError("novel IDs/order/cardinality differ from slots")
    return batch


def _run_refill_batch(
    concepts: Sequence[CefrRefillConcept],
    *,
    lang: str,
    level: str,
    client: RefillClient,
    ledger: Ledger,
    single_model: str | None,
) -> CefrRefillBatch:
    identity = {"row_ids": [concept.id for concept in concepts]}
    span = f"{concepts[0].id}..{concepts[-1].id}"
    generation_prompt = build_refill_generation_prompt(
        concepts,
        lang=lang,
        level=level,
    )
    profile = get_language(lang)
    prompt_version = (
        "legacy" if profile.code == "de" else f"cefr-refill-{profile.code}-v3"
    )
    namespace = (
        f"refill:{lang}:{level}:{span}"
        if profile.code == "de"
        else f"refill:{prompt_version}:{lang}:{level}:{span}"
    )
    generated = _checkpointed_refill(
        client=client,
        ledger=ledger,
        model=single_model or MODEL_31B,
        batch_id=f"{namespace}:generation",
        prompt=generation_prompt,
        concepts=concepts,
        identity=identity,
    )
    if single_model is not None:
        return generated
    review_prompt = build_refill_review_prompt(
        concepts,
        generated,
        lang=lang,
        level=level,
    )
    reviewed = _checkpointed_refill(
        client=client,
        ledger=ledger,
        model=MODEL_26B,
        batch_id=f"{namespace}:review",
        prompt=review_prompt,
        concepts=concepts,
        identity=identity,
    )
    if generated.model_dump(mode="json") == reviewed.model_dump(mode="json"):
        return generated
    adjudication_prompt = build_refill_adjudication_prompt(
        concepts,
        generated,
        reviewed,
        lang=lang,
        level=level,
    )
    return _checkpointed_refill(
        client=client,
        ledger=ledger,
        model=resolve_adjudication_model(client),
        batch_id=f"{namespace}:adjudication",
        prompt=adjudication_prompt,
        concepts=concepts,
        identity=identity,
    )


def _checkpointed_refill(
    *,
    client: RefillClient,
    ledger: Ledger,
    model: str,
    batch_id: str,
    prompt: str,
    concepts: Sequence[CefrRefillConcept],
    identity: object,
) -> CefrRefillBatch:
    return checkpointed_semantic_generate(
        client=client,
        ledger=ledger,
        model=model,
        batch_id=batch_id,
        prompt=prompt,
        response_model=CefrRefillBatch,
        max_output_tokens=MAX_REFILL_OUTPUT_TOKENS,
        validate=lambda batch: _validate_refill_identity(concepts, batch),
        expected_identity=identity,
    )


def _validate_refill_identity(
    concepts: Sequence[CefrRefillConcept],
    batch: CefrRefillBatch,
) -> CefrRefillBatch:
    expected = [concept.id for concept in concepts]
    actual = [row.id for row in batch.rows]
    if actual != expected:
        raise ValueError("refill IDs/order/cardinality differ from input")
    return batch
