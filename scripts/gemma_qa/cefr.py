from __future__ import annotations

import threading
import time

import csv
import unicodedata
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed, wait
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
from .config import (
    DUAL_POOL,
    INPUT_BATCH_TOKEN_CAP,
    MODEL_IDS,
    default_batch_concurrency,
    probe_optional_models,
)
from .ledger import Ledger
from .language_repair import (
    LanguageRepairClient,
    canonicalize_english_review_rows,
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
from .routing import resolve_adjudication_model, select_dual_models
from .progress import batch_progress_line, print_progress
from .semantic_generation import checkpointed_semantic_generate
from .trace import event
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
    """Align review rows to input IDs/order.

    Models occasionally reorder IDs. Reorder by id to match input order. Raise
    on cardinality mismatch, unknown IDs, or duplicates so the semantic repair
    loop can retry instead of silently accepting garbage.
    """
    input_ids = [row.id for row in inputs]
    review_ids = [row.id for row in reviews.rows]
    if len(reviews.rows) != len(input_ids):
        raise ValueError(
            "IDs/order/cardinality differ from input: "
            f"review_count={len(reviews.rows)} input_count={len(input_ids)}"
        )
    unknown = [rid for rid in review_ids if rid not in set(input_ids)]
    if unknown:
        raise ValueError(
            "IDs/order/cardinality differ from input: "
            f"unknown_ids={unknown[:3]}"
        )
    duplicates = [rid for rid, count in Counter(review_ids).items() if count > 1]
    if duplicates:
        raise ValueError(
            "IDs/order/cardinality differ from input: "
            f"duplicate_ids={duplicates[:3]}"
        )
    by_id: dict[str, CefrReviewRow] = {row.id: row for row in reviews.rows}
    ordered = [by_id[iid] for iid in input_ids]
    repaired = CefrReviewBatch(rows=ordered)
    if review_ids != input_ids:
        from .trace import event

        event(
            "cefr.review_id_repair",
            level="WARN",
            input_count=len(input_ids),
            review_count=len(review_ids),
            filled_keeps=0,
            order_mismatch=True,
        )
    return repaired


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
    event(
        "refill.start",
        lang=lang,
        level_name=level,
        accepted=len(accepted),
        target=target,
        gap=max(0, target - len(accepted)),
        single_model=single_model,
    )
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
    try:
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
            protect_accepted=True,
        )
    except ReviewRequiredError as error:
        # Keep progress when novel slots stall short of exact target.
        if len(error.partial_rows) <= len(accepted):
            event(
                "refill.stall",
                level="WARN",
                lang=lang,
                level_name=level,
                accepted=len(accepted),
                target=target,
                partial=len(error.partial_rows),
                error=str(error).splitlines()[0][:400],
            )
            raise
        final = error.partial_rows
        event(
            "refill.partial",
            level="WARN",
            lang=lang,
            level_name=level,
            accepted=len(accepted),
            target=target,
            final=len(final),
            gap=len(final) - len(accepted),
        )
    gap_rows = final[len(accepted) :]
    if not gap_rows:
        raise ReviewRequiredError(
            f"{lang} {level}: gap refill produced no new rows "
            f"(still {len(accepted)} of {target})"
        )
    output = write_gap_proposed_csv(document, gap_rows)
    event(
        "refill.ok",
        lang=lang,
        level_name=level,
        accepted=len(accepted),
        target=target,
        gap=len(gap_rows),
        output=str(output),
        sample=[
            {"lemma": row.lemma, "upos": row.upos.value, "en": row.english_lemma}
            for row in gap_rows[:12]
        ],
    )
    return output


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
    batch_concurrency: int | None = None,
) -> Path:
    if single_model is not None and single_model not in MODEL_IDS:
        raise ValueError(f"unsupported model: {single_model}")
    concurrency = (
        default_batch_concurrency()
        if batch_concurrency is None
        else max(1, batch_concurrency)
    )
    disabled_optional = probe_optional_models()
    if disabled_optional:
        event(
            "cefr.optional_models_disabled",
            models=disabled_optional,
            reason="not listed on /v1/models",
        )
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
        event(
            "cefr.start",
            lang=lang,
            level_name=level,
            selected=len(selected),
            frozen=len(frozen),
            pending=len(pending),
            single_model=single_model,
            refill_to_target=refill_to_target,
            batch_concurrency=concurrency,
        )
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
            event(
                "cefr.batches",
                lang=lang,
                level_name=level,
                batch_count=len(packed),
                model_pool=list(DUAL_POOL) if single_model is None else [single_model],
                pending=len(pending),
                batch_concurrency=concurrency,
                single_model=single_model,
            )
            reviewed = _review_pending_batches(
                packed,
                client=client,
                ledger=ledger,
                lang=lang,
                level=level,
                single_model=single_model,
                concurrency=concurrency,
            )
            for batch_rows, chosen in zip(packed, reviewed, strict=True):
                accepted.extend(chosen.rows)
        target = TARGETS[level] if limit is None else len(selected)
        collision_keys = load_other_level_collision_keys(
            root,
            lang=lang,
            level=level,
        )
        if profile.code == "en":
            accepted = canonicalize_english_review_rows(accepted)
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


def _review_pending_batches(
    packed: Sequence[Sequence[CefrInputRow]],
    *,
    client: CefrClient,
    ledger: Ledger,
    lang: str,
    level: str,
    single_model: str | None,
    concurrency: int,
) -> list[CefrReviewBatch]:
    """Review CEFR row batches with multi-batch + multi-model parallelism.

    Each batch picks a dual pair from the full pool (Qwen/Gemma/GLM internal
    + external) so concurrent work spreads across models and gateways.
    Results preserve pack order. Ledger checkpoints are restart-safe.
    """
    batch_count = len(packed)
    rows_total = sum(len(batch) for batch in packed)
    if batch_count == 0:
        return []
    workers = max(1, min(concurrency, batch_count))
    progress_lock = threading.Lock()
    batch_durations: list[float] = []
    batches_done = 0
    rows_done = 0
    batch_loop_started = time.time()

    def _progress(
        *,
        batch_index: int,
        rows_in_batch: int,
        status: str,
        wait_s: float | None = None,
    ) -> None:
        with progress_lock:
            print_progress(
                batch_progress_line(
                    lang=lang,
                    level=level,
                    batch_index=batch_index,
                    batch_count=batch_count,
                    rows_in_batch=rows_in_batch,
                    rows_done=rows_done,
                    rows_total=rows_total,
                    durations=list(batch_durations),
                    started_at=batch_loop_started,
                    status=status,
                    wait_s=wait_s,
                    completed=batches_done,
                    concurrency=workers,
                )
            )

    print_progress(
        batch_progress_line(
            lang=lang,
            level=level,
            batch_index=0,
            batch_count=batch_count,
            rows_in_batch=0,
            rows_done=0,
            rows_total=rows_total,
            durations=[],
            started_at=batch_loop_started,
            status="init",
            completed=0,
            concurrency=workers,
        )
    )

    def work(batch_index: int, batch_rows: Sequence[CefrInputRow]) -> CefrReviewBatch:
        nonlocal batches_done, rows_done
        batch_id = f"{batch_rows[0].id}..{batch_rows[-1].id}"
        batch_started = time.time()
        _progress(
            batch_index=batch_index,
            rows_in_batch=len(batch_rows),
            status="running",
            wait_s=0.0,
        )
        # Retriable: optional 422, wall-clock/slot timeout, transport flake.
        # Rotate dual pair each attempt so a hung model does not pin the batch.
        last_error: Exception | None = None
        chosen: CefrReviewBatch | None = None
        for attempt in range(1, 5):
            if single_model is not None:
                models: tuple[str, ...] = (single_model,)
            else:
                models = select_dual_models(batch_index=batch_index + attempt - 1)
            event(
                "cefr.batch_start",
                lang=lang,
                level_name=level,
                batch_id=batch_id,
                attempt=batch_index,
                pair_attempt=attempt,
                rows=len(batch_rows),
                batch_count=batch_count,
                remaining_batches=batch_count - batch_index + 1,
                concurrency=workers,
                dual_models=list(models),
            )
            try:
                chosen = _review_one_batch(
                    batch_rows,
                    client=client,
                    ledger=ledger,
                    lang=lang,
                    models=models,
                    batch_id=batch_id,
                    batch_index=batch_index,
                    batch_count=batch_count,
                    on_wait=lambda wait_s, status: _progress(
                        batch_index=batch_index,
                        rows_in_batch=len(batch_rows),
                        status=status,
                        wait_s=wait_s,
                    ),
                    batch_started=batch_started,
                )
                break
            except Exception as error:  # noqa: BLE001 — dual pair rotation
                last_error = error
                if not _is_retriable_batch_error(error) or attempt >= 4:
                    raise
                event(
                    "cefr.dual_retry",
                    level="WARN",
                    batch_id=batch_id,
                    dual_models=list(models),
                    error=str(error)[:300],
                    pair_attempt=attempt,
                )
                continue
        if chosen is None:
            assert last_error is not None
            raise last_error
        elapsed = time.time() - batch_started
        with progress_lock:
            batches_done += 1
            rows_done += len(batch_rows)
            batch_durations.append(elapsed)
        _progress(
            batch_index=batch_index,
            rows_in_batch=len(batch_rows),
            status="ok",
        )
        return chosen

    results: list[CefrReviewBatch | None] = [None] * batch_count
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures: dict[Future[CefrReviewBatch], int] = {
            pool.submit(work, index + 1, batch_rows): index
            for index, batch_rows in enumerate(packed)
        }
        for future in as_completed(futures):
            index = futures[future]
            results[index] = future.result()
    return [cast(CefrReviewBatch, item) for item in results]


def _is_retriable_batch_error(error: BaseException) -> bool:
    """422 optional models, wall-clock/slot timeouts, transport flakes."""
    if isinstance(error, (TimeoutError, OSError)):
        return True
    try:
        import httpx

        if isinstance(error, (httpx.TimeoutException, httpx.TransportError)):
            return True
    except ImportError:
        pass
    message = str(error).lower()
    needles = (
        "422",
        "unprocessable",
        "timed out",
        "timeout",
        "wall clock",
        "slot acquire",
        "connecterror",
        "remoteprotocolerror",
    )
    return any(part in message for part in needles)


def _dual_wait_ceiling_s() -> float:
    """Max seconds dual legs may wait before failing the pair (rotate models)."""
    from .client import request_wall_clock_s

    # ≤2 wall-clock failures per generate on each dual leg + buffer.
    # Was 2×wall+60 (~420s); that aborted healthy slow B2 duals mid-task.
    return request_wall_clock_s() * 3 + 120.0


def _review_one_batch(
    batch_rows: Sequence[CefrInputRow],
    *,
    client: CefrClient,
    ledger: Ledger,
    lang: str,
    models: Sequence[str],
    batch_id: str,
    batch_index: int,
    batch_count: int,
    on_wait: Callable[[float, str], None] | None = None,
    batch_started: float | None = None,
) -> CefrReviewBatch:
    _ = batch_count
    prompt = build_cefr_prompt(batch_rows, lang=lang)
    started = batch_started if batch_started is not None else time.time()
    ceiling = _dual_wait_ceiling_s()
    # wait=False on shutdown: wall-clock threads must not block pair rotation.
    executor = ThreadPoolExecutor(max_workers=len(models))
    try:
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
        pending_futures = set(futures)
        while pending_futures:
            elapsed = time.time() - started
            if elapsed >= ceiling:
                event(
                    "cefr.dual_wait_ceiling",
                    level="WARN",
                    batch_id=batch_id,
                    wait_s=round(elapsed, 1),
                    ceiling_s=ceiling,
                    dual_models=list(models),
                )
                for future in pending_futures:
                    future.cancel()
                raise TimeoutError(
                    f"dual generate wait ceiling {ceiling:.0f}s exceeded "
                    f"for {batch_id}"
                )
            done_now, pending_futures = wait(pending_futures, timeout=10.0)
            _ = done_now
            if pending_futures and on_wait is not None:
                on_wait(time.time() - started, "waiting")
        independent = [future.result() for future in futures]
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
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
        if on_wait is not None:
            on_wait(time.time() - started, "adjudicating")
        adj_model = resolve_adjudication_model(
            client,
            prompt=adjudication_prompt,
            exclude=tuple(models),
            batch_index=batch_index,
        )
        event(
            "cefr.adjudicate",
            batch_id=batch_id,
            dual_models=list(models),
            adj_model=adj_model,
        )
        chosen = _checkpointed_generate(
            client=client,
            ledger=ledger,
            model=adj_model,
            batch_id=f"{batch_id}:adjudication",
            prompt=adjudication_prompt,
            expected_rows=batch_rows,
        )
        validate_review_batch(batch_rows, chosen)
    return chosen


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
